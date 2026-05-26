# path: src/loaders/LazyFITS.jl
#
# Lazy / memory-mapped FITS arrays.
#
# `load_fits(...; lazy = true)` returns a dataset whose underlying array
# is a `LazyFITSCube` (3D) or `LazyFITSImage` (2D). These types satisfy
# the read-only `AbstractArray` interface that the rest of MANTA needs
# (`size`, `eltype`, `getindex`, plus a `Slicing`-friendly `view` along
# the slowest axis), but they NEVER load the full cube up-front.
#
# Implementation strategy:
#   * Open the FITS file via FITSIO inside a `do` block for every read,
#     using `read(hdu, r1, r2, ...)` to fetch the needed sub-array only.
#   * Cache the last requested slice (cube case) so that successive
#     accesses along the same channel are essentially free.
#   * Convert to `Float32` once, at the boundary, because the rest of
#     MANTA assumes Float32 display arrays.
#
# We deliberately re-open the FITS file every time instead of keeping a
# long-lived handle: FITSIO's `FITS` object is not safe to share across
# tasks, and the viewer is heavily Observable-driven (so reads happen
# from inside `lift` closures on the main thread).

using FITSIO

abstract type AbstractLazyFITS{T,N} <: AbstractArray{T,N} end

# ---- 2D image ---------------------------------------------------------

mutable struct LazyFITSImage{T<:Real} <: AbstractLazyFITS{T,2}
    path::String
    hdu::Int
    sz::NTuple{2,Int}
end

Base.size(L::LazyFITSImage) = L.sz
Base.eltype(::LazyFITSImage{T}) where {T} = T

function Base.getindex(L::LazyFITSImage{T}, i::Int, j::Int) where {T}
    FITS(L.path) do f
        return T(read(f[L.hdu], i:i, j:j)[1])
    end
end

# Full-image materialisation (rare: only when the viewer really needs an Array).
function Base.collect(L::LazyFITSImage{T}) where {T}
    FITS(L.path) do f
        return Array{T,2}(read(f[L.hdu]))
    end
end

# ---- 3D cube ----------------------------------------------------------

mutable struct LazyFITSCube{T<:Real} <: AbstractLazyFITS{T,3}
    path::String
    hdu::Int
    sz::NTuple{3,Int}
    # Last slice cache: keyed on (axis, idx). Invalidated when the user moves.
    cache_axis::Int
    cache_idx::Int
    cache_data::Union{Nothing, Matrix{T}}
end

function LazyFITSCube{T}(path::AbstractString, hdu::Integer,
                         sz::NTuple{3,Integer}) where {T<:Real}
    LazyFITSCube{T}(String(path), Int(hdu), Int.(sz), 0, 0, nothing)
end

Base.size(L::LazyFITSCube) = L.sz
Base.eltype(::LazyFITSCube{T}) where {T} = T

# Single-element access. Slow path — used only by generic Julia code.
function Base.getindex(L::LazyFITSCube{T}, i::Int, j::Int, k::Int) where {T}
    FITS(L.path) do f
        return T(read(f[L.hdu], i:i, j:j, k:k)[1])
    end
end

"""
    read_slice!(L::LazyFITSCube, axis, idx) -> Matrix

Materialise a single 2D slice along `axis` (1, 2, or 3) at index `idx`.
Cached internally so the viewer can hover/zoom without hitting disk.
"""
function read_slice!(L::LazyFITSCube{T}, axis::Integer, idx::Integer) where {T}
    a, k = Int(axis), Int(idx)
    if a == L.cache_axis && k == L.cache_idx && L.cache_data !== nothing
        return L.cache_data
    end
    nx, ny, nz = L.sz
    slice = FITS(L.path) do f
        if a == 1
            1 <= k <= nx || throw(BoundsError(L, (k, :, :)))
            raw = read(f[L.hdu], k:k, 1:ny, 1:nz)
            return Matrix{T}(@view raw[1, :, :])
        elseif a == 2
            1 <= k <= ny || throw(BoundsError(L, (:, k, :)))
            raw = read(f[L.hdu], 1:nx, k:k, 1:nz)
            return Matrix{T}(@view raw[:, 1, :])
        elseif a == 3
            1 <= k <= nz || throw(BoundsError(L, (:, :, k)))
            raw = read(f[L.hdu], 1:nx, 1:ny, k:k)
            return Matrix{T}(@view raw[:, :, 1])
        else
            throw(ArgumentError("axis must be 1, 2 or 3, got $(a)"))
        end
    end
    L.cache_axis = a
    L.cache_idx  = k
    L.cache_data = slice
    return slice
end

"""
    read_spectrum(L::LazyFITSCube, i, j) -> Vector

Read the spectrum at pixel `(i, j)` along axis 3 (channel direction).
"""
function read_spectrum(L::LazyFITSCube{T}, i::Integer, j::Integer) where {T}
    nx, ny, nz = L.sz
    1 <= i <= nx || throw(BoundsError(L, (i, j, :)))
    1 <= j <= ny || throw(BoundsError(L, (i, j, :)))
    return FITS(L.path) do f
        return Vector{T}(vec(read(f[L.hdu], i:i, j:j, 1:nz)))
    end
end

# Fully materialise — used by tests and as an escape hatch.
function Base.collect(L::LazyFITSCube{T}) where {T}
    return FITS(L.path) do f
        Array{T,3}(read(f[L.hdu]))
    end
end

# Generic "view"-shaped slice access: hand off to `read_slice!` so callers
# can write `view(cube, :, :, k)` and get the same caching benefit.
Base.view(L::LazyFITSCube, ::Colon, ::Colon, k::Int) = read_slice!(L, 3, k)
Base.view(L::LazyFITSCube, ::Colon, j::Int, ::Colon) = read_slice!(L, 2, j)
Base.view(L::LazyFITSCube, i::Int, ::Colon, ::Colon) = read_slice!(L, 1, i)

# ---- factory used by load_fits(...; lazy = true) ----------------------

"""
    open_lazy_fits(path; hdu = 1) -> (header, lazy_array, n_hdus)

Inspect the requested HDU once to capture its header + shape, then close
the file. Returns a `LazyFITSImage` or `LazyFITSCube` (Float32) that
defers actual pixel reads.

Throws `HDUSelectionError` if `hdu` is out of range or refers to a non-image
HDU, and `UnsupportedFormatError` for ndims outside {2, 3}.
"""
function open_lazy_fits(path::AbstractString; hdu::Integer = 1)
    require_file(path)
    n_hdus = 0
    header = nothing
    dims = ()
    eltype_T = Float32
    FITS(path) do f
        n_hdus = length(f)
        Int(hdu) <= n_hdus || invalid_hdu(path, Int(hdu), n_hdus)
        h = f[Int(hdu)]
        header = try read_header(h) catch _ nothing end
        # FITSIO exposes the dimensions via `size(h)`.
        dims = try
            tuple(Int.(size(h))...)
        catch e
            rethrow_actionable(e, path;
                format_hint = "L'HDU #$(hdu) ne semble pas être une ImageHDU.")
        end
    end
    if length(dims) == 2
        return (header, LazyFITSImage{eltype_T}(String(path), Int(hdu), dims), n_hdus)
    elseif length(dims) == 3
        return (header, LazyFITSCube{eltype_T}(String(path), Int(hdu), dims), n_hdus)
    else
        throw(DatasetShapeError(
            "le chargement lazy ne supporte que les images 2D et cubes 3D " *
            "(HDU #$(hdu) dans $(path) a $(length(dims)) dimensions).",
            "Désactive `lazy = true` ou choisis une autre HDU."))
    end
end

export LazyFITSImage, LazyFITSCube, AbstractLazyFITS
export read_slice!, read_spectrum, open_lazy_fits
