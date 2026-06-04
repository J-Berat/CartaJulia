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
#   * Keep a small LRU cache of requested slices (cube case) so quick
#     forward/backward scrolling does not repeatedly hit disk.
#   * Convert to `Float32` once, at the boundary, because the rest of
#     MANTA assumes Float32 display arrays.
#
# We deliberately re-open the FITS file every time instead of keeping a
# long-lived handle: FITSIO's `FITS` object is not safe to share across
# tasks, and the viewer is heavily Observable-driven (so reads happen
# from inside `lift` closures on the main thread).
#
# Async prefetch:
#   Call `prefetch_slice!(L, axis, idx)` right after `read_slice!` to start
#   loading the next slice in the background via `@async`. The next
#   `read_slice!` call for that (axis, idx) pair will `take!` the ready
#   result instead of hitting disk again, shaving the full read latency.
#   The design assumes single-threaded cooperative scheduling (the default
#   Julia runtime). If MANTA is ever moved to `Threads.@spawn`, a
#   `ReentrantLock` guard will be needed around the prefetch fields.

using FITSIO

abstract type AbstractLazyFITS{T,N} <: AbstractArray{T,N} end

const DEFAULT_FITS_SLICE_CACHE_CAPACITY = 7

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
    # Most-recent slice mirror, kept for cheap hits and historical introspection.
    cache_axis::Int
    cache_idx::Int
    cache_data::Union{Nothing, Matrix{T}}
    # Small LRU cache keyed on (axis, idx). The last key in cache_lru is MRU.
    cache_capacity::Int
    slice_cache::Dict{Tuple{Int,Int},Matrix{T}}
    cache_lru::Vector{Tuple{Int,Int}}
    # Async prefetch state.
    # `prefetch_ch` holds a Channel{Matrix{T}}(1) that the background task will
    # `put!` its result into. The key (prefetch_axis, prefetch_idx) identifies
    # which slice is being loaded. Set to `nothing` when no prefetch is pending.
    prefetch_axis::Int
    prefetch_idx::Int
    prefetch_ch::Union{Nothing, Channel{Matrix{T}}}
end

function LazyFITSCube{T}(path::AbstractString, hdu::Integer,
                         sz::NTuple{3,Integer};
                         cache_capacity::Integer = DEFAULT_FITS_SLICE_CACHE_CAPACITY) where {T<:Real}
    cap = max(1, Int(cache_capacity))
    LazyFITSCube{T}(
        String(path), Int(hdu), Int.(sz), 0, 0, nothing,
        cap, Dict{Tuple{Int,Int},Matrix{T}}(), Tuple{Int,Int}[],
        0, 0, nothing,
    )
end

Base.size(L::LazyFITSCube) = L.sz
Base.eltype(::LazyFITSCube{T}) where {T} = T

# Single-element access. Slow path — used only by generic Julia code.
function Base.getindex(L::LazyFITSCube{T}, i::Int, j::Int, k::Int) where {T}
    FITS(L.path) do f
        return T(read(f[L.hdu], i:i, j:j, k:k)[1])
    end
end

# Private: open the FITS file and read exactly the requested slice.
# Called both by `read_slice!` (synchronous path) and the prefetch task.
function _read_slice_raw(L::LazyFITSCube{T}, a::Int, k::Int) where {T}
    nx, ny, nz = L.sz
    return FITS(L.path) do f
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
end

function _touch_slice_cache_key!(L::LazyFITSCube, key::Tuple{Int,Int})
    pos = findfirst(==(key), L.cache_lru)
    pos !== nothing && deleteat!(L.cache_lru, pos)
    push!(L.cache_lru, key)
    return nothing
end

function _remember_slice!(L::LazyFITSCube{T}, a::Int, k::Int,
                          slice::Matrix{T}) where {T}
    key = (a, k)
    L.slice_cache[key] = slice
    _touch_slice_cache_key!(L, key)
    while length(L.cache_lru) > L.cache_capacity
        stale = popfirst!(L.cache_lru)
        delete!(L.slice_cache, stale)
    end
    L.cache_axis = a
    L.cache_idx  = k
    L.cache_data = slice
    return slice
end

function _cached_slice!(L::LazyFITSCube{T}, a::Int, k::Int) where {T}
    key = (a, k)
    slice = get(L.slice_cache, key, nothing)
    slice === nothing && return nothing
    _touch_slice_cache_key!(L, key)
    L.cache_axis = a
    L.cache_idx  = k
    L.cache_data = slice
    return slice
end

_has_cached_slice(L::LazyFITSCube, a::Int, k::Int) =
    haskey(L.slice_cache, (a, k))

"""
    read_slice!(L::LazyFITSCube, axis, idx) -> Matrix

Materialise a single 2D slice along `axis` (1, 2, or 3) at index `idx`.
Result is cached in a small LRU window so the viewer can hover/zoom or reverse
scroll direction without hitting disk.

If `prefetch_slice!` was called earlier for the same `(axis, idx)`, the
background load is awaited and the ready result is returned directly —
avoiding the full synchronous disk read latency.
"""
function read_slice!(L::LazyFITSCube{T}, axis::Integer, idx::Integer) where {T}
    a, k = Int(axis), Int(idx)

    # 1. Cache hit — free.
    slice_cached = _cached_slice!(L, a, k)
    slice_cached !== nothing && return slice_cached

    # 2. Prefetch hit — the background task is already loading (or has loaded)
    #    exactly this slice.  `take!` blocks until the result arrives (usually
    #    it is already there by the time the user moves to the next slice).
    slice = if a == L.prefetch_axis && k == L.prefetch_idx &&
               L.prefetch_ch !== nothing
        ch = L.prefetch_ch
        L.prefetch_ch = nothing   # consume the channel; clear the slot
        if isopen(ch)
            try
                take!(ch)
            catch
                # The background task failed (e.g. I/O error). Fall back.
                _read_slice_raw(L, a, k)
            end
        else
            # Channel was closed before being filled — fall back.
            _read_slice_raw(L, a, k)
        end
    else
        # 3. Cold path — cancel any stale prefetch and read synchronously.
        if L.prefetch_ch !== nothing
            close(L.prefetch_ch)
            L.prefetch_ch = nothing
        end
        _read_slice_raw(L, a, k)
    end

    return _remember_slice!(L, a, k, slice)
end

"""
    prefetch_slice!(L::LazyFITSCube, axis, idx)

Start loading the slice at `(axis, idx)` in the background so that the next
`read_slice!(L, axis, idx)` call can return immediately without waiting for
disk I/O.  Safe to call speculatively: if the slice is already cached, or a
prefetch for the same key is already in flight, this is a no-op.

Typical usage in a CubeView `on` callback:

```julia
slice = read_slice!(cube, axis, k)
prefetch_slice!(cube, axis, k + 1)   # speculatively load next channel
```
"""
function prefetch_slice!(L::LazyFITSCube{T},
                         axis::Integer, idx::Integer) where {T}
    a, k = Int(axis), Int(idx)

    # Bounds check — avoid launching a task that will immediately throw.
    a in (1, 2, 3) || return nothing
    1 <= k <= L.sz[a] || return nothing

    # Already cached — nothing to do.
    _has_cached_slice(L, a, k) && return nothing

    # Already prefetching the same slice — nothing to do.
    if a == L.prefetch_axis && k == L.prefetch_idx && L.prefetch_ch !== nothing
        return nothing
    end

    # Cancel any previous (now stale) prefetch.
    if L.prefetch_ch !== nothing
        close(L.prefetch_ch)
    end

    ch = Channel{Matrix{T}}(1)
    L.prefetch_axis = a
    L.prefetch_idx  = k
    L.prefetch_ch   = ch

    @async begin
        try
            put!(ch, _read_slice_raw(L, a, k))
        catch
            # Silently close; `read_slice!` will fall back to the sync path.
            isopen(ch) && close(ch)
        end
    end

    return nothing
end

"""
    prefetch_adjacent!(L::LazyFITSCube, axis, idx_new, last_idx) -> Int

Direction-aware speculative prefetch for cube scrolling. Infers the scroll
direction by comparing `idx_new` with `last_idx` (`+1` ascending, `-1`
descending, defaulting to `+1` when the index is unchanged), then starts an
async [`prefetch_slice!`](@ref) for the neighbouring slice along `axis` when
that neighbour is in bounds.

Returns `idx_new` so the caller can update its "last index" tracker in a
single expression:

```julia
last_idx_ref[] = prefetch_adjacent!(cube, axis[], idx[], last_idx_ref[])
```

An out-of-range `axis`, or a neighbour outside `1:size(L, axis)` (i.e. the
user sits at either end of the cube), makes this a no-op — it still returns
`idx_new` so the tracker stays correct.
"""
function prefetch_adjacent!(L::LazyFITSCube, axis::Integer, idx_new::Integer,
                            last_idx::Integer)
    a  = Int(axis)
    id = Int(idx_new)
    a in (1, 2, 3) || return id
    dir = id > Int(last_idx) ? 1 : id < Int(last_idx) ? -1 : 1
    nbr = id + dir
    1 <= nbr <= L.sz[a] && prefetch_slice!(L, a, nbr)
    return id
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
                format_hint = "HDU #$(hdu) does not appear to be an ImageHDU.")
        end
    end
    if length(dims) == 2
        return (header, LazyFITSImage{eltype_T}(String(path), Int(hdu), dims), n_hdus)
    elseif length(dims) == 3
        return (header, LazyFITSCube{eltype_T}(String(path), Int(hdu), dims), n_hdus)
    else
        throw(DatasetShapeError(
            "lazy loading only supports 2D images and 3D cubes " *
            "(HDU #$(hdu) in $(path) has $(length(dims)) dimensions).",
            "Disable `lazy = true` or choose a different HDU."))
    end
end

# ---- MANTA helper overrides for lazy types ----------------------------
#
# All four overrides live here (rather than in the helpers/ files) because
# `AbstractLazyFITS` is only in scope *after* the loaders are included.
# helpers/ is included first, so adding the methods there would require a
# forward declaration of the type — this is the cleaner alternative.

# 1. as_float32 --------------------------------------------------------
#
# `as_float32(::AbstractArray)` in UIBits.jl calls `Array{Float32}(x)` /
# `Float32.(x)`, both of which materialise the whole array.
# `LazyFITSCube{Float32}` / `LazyFITSImage{Float32}` already satisfy the
# `AbstractArray{Float32}` contract the viewer needs — no copy required.
as_float32(x::AbstractLazyFITS) = x

# 2. Spectrum-shaped Base.view -----------------------------------------
#
# `@views data[i, j, :]` (and its axis-1/axis-2 siblings) in
# SpectrumBundle.jl and CubeView.jl's init block dispatch to
# `Base.view(data, i, j, Colon())`.  Without these methods the fallback
# constructs a `SubArray` that calls `getindex` once per channel —
# O(n_channels) file opens for a single spectrum.  Each method below
# issues exactly ONE FITSIO `read` call.
Base.view(L::LazyFITSCube{T}, i::Int, j::Int, ::Colon) where {T} =
    FITS(L.path) do f
        Vector{T}(vec(read(f[L.hdu], i:i, j:j, 1:L.sz[3])))
    end
Base.view(L::LazyFITSCube{T}, i::Int, ::Colon, k::Int) where {T} =
    FITS(L.path) do f
        Vector{T}(vec(read(f[L.hdu], i:i, 1:L.sz[2], k:k)))
    end
Base.view(L::LazyFITSCube{T}, ::Colon, j::Int, k::Int) where {T} =
    FITS(L.path) do f
        Vector{T}(vec(read(f[L.hdu], 1:L.sz[1], j:j, k:k)))
    end

# 3. moment_map (slice-first) ------------------------------------------
#
# The generic `moment_map` calls `_channel_value(data, axis, u, v, c)`
# which is a single-element `getindex` — O(nx·ny·n_channels) file opens.
# This override iterates channels in the outer loop and reads one 2-D
# slice per channel: O(n_channels) opens, O(nx·ny) peak memory.
#
# nsigma + sigma=nothing note:
#   Per-pixel robust-σ estimation (MAD) requires the full spectrum per
#   pixel, which brings us back to O(nx·ny·nz) reads.  When `nsigma` is
#   requested without an explicit `sigma`, we fall back to the absolute
#   `threshold` and emit a warning so the caller knows.
function moment_map(
    data::AbstractLazyFITS{T,3},
    axis::Integer,
    order::Integer;
    coords    = collect(Float32, 1:size(data, axis)),
    channels  = 1:size(data, axis),
    threshold = 0.0,
    nsigma    = nothing,
    sigma     = nothing,
    dx        = nothing,
    mask::Union{Nothing,AbstractArray{Bool,3}} = nothing,
) where {T}
    1 <= axis <= 3 || throw(ArgumentError("axis must be 1, 2, or 3"))
    order in (0, 1, 2) || throw(ArgumentError("moment order must be 0, 1, or 2"))
    if mask !== nothing && size(mask) != size(data)
        throw(DimensionMismatch(
            "mask size $(size(mask)) must match data size $(size(data))"))
    end

    u_max, v_max = axis == 1 ? (size(data, 2), size(data, 3)) :
                   axis == 2 ? (size(data, 1), size(data, 3)) :
                               (size(data, 1), size(data, 2))
    out      = fill(NaN32, u_max, v_max)
    chan_vec = [c for c in channels if 1 <= c <= size(data, axis)]
    isempty(chan_vec) && return out

    x      = Float32[Float32(coords[c]) for c in chan_vec]
    widths = _channel_widths(x, dx)

    # Threshold resolution.
    thr = if nsigma !== nothing && sigma !== nothing
        Float64(nsigma) * Float64(sigma)
    elseif nsigma !== nothing
        @warn "moment_map on lazy FITS: `nsigma` without explicit `sigma` falls " *
              "back to absolute `threshold=$(threshold)` (per-pixel MAD estimation " *
              "would require O(nx·ny·nz) file opens). Pass `sigma=<value>` to " *
              "enable σ-clipping with lazy data."
        Float64(threshold)
    else
        Float64(threshold)
    end

    # Float64 accumulators for M0 and M1.
    acc0    = zeros(Float64, u_max, v_max)
    acc1    = zeros(Float64, u_max, v_max)
    any_hit = falses(u_max, v_max)
    m1_map  = zeros(Float64, u_max, v_max)   # filled in if order >= 1

    # ---- Pass 1: M0 (and M1 for order ≥ 1) ---- #
    for (n, c) in pairs(chan_vec)
        s  = get_slice_view(data, axis, c)   # one FITSIO read via read_slice!
        xi = Float64(x[n])
        wi = Float64(widths[n])
        @inbounds for v in 1:v_max
            for u in 1:u_max
                if mask !== nothing
                    mval = if axis == 1; mask[c, u, v]
                           elseif axis == 2; mask[u, c, v]
                           else; mask[u, v, c]; end
                    mval || continue
                end
                yi = Float64(Float32(s[u, v]))
                if isfinite(yi) && yi > thr
                    any_hit[u, v] = true
                    acc0[u, v]   += yi * wi
                    order >= 1 && (acc1[u, v] += yi * xi * wi)
                end
            end
        end
    end

    if order == 0
        @inbounds for v in 1:v_max, u in 1:u_max
            any_hit[u, v] && acc0[u, v] != 0.0 &&
                (out[u, v] = Float32(acc0[u, v]))
        end
        return out
    end

    @inbounds for v in 1:v_max, u in 1:u_max
        if any_hit[u, v] && acc0[u, v] != 0.0
            m1_map[u, v] = acc1[u, v] / acc0[u, v]
        end
    end

    if order == 1
        @inbounds for v in 1:v_max, u in 1:u_max
            any_hit[u, v] && acc0[u, v] != 0.0 &&
                (out[u, v] = Float32(m1_map[u, v]))
        end
        return out
    end

    # ---- Pass 2: M2 dispersion ---- #
    acc2 = zeros(Float64, u_max, v_max)
    for (n, c) in pairs(chan_vec)
        s  = get_slice_view(data, axis, c)
        xi = Float64(x[n])
        wi = Float64(widths[n])
        @inbounds for v in 1:v_max
            for u in 1:u_max
                if mask !== nothing
                    mval = if axis == 1; mask[c, u, v]
                           elseif axis == 2; mask[u, c, v]
                           else; mask[u, v, c]; end
                    mval || continue
                end
                yi = Float64(Float32(s[u, v]))
                if isfinite(yi) && yi > thr && any_hit[u, v] && acc0[u, v] != 0.0
                    acc2[u, v] += yi * (xi - m1_map[u, v])^2 * wi
                end
            end
        end
    end
    @inbounds for v in 1:v_max, u in 1:u_max
        if any_hit[u, v] && acc0[u, v] != 0.0
            m2_2 = acc2[u, v] / acc0[u, v]
            out[u, v] = m2_2 >= 0.0 ? Float32(sqrt(m2_2)) : NaN32
        end
    end
    return out
end

# 4. mean_region_spectrum (slice-first) --------------------------------
#
# The generic version iterates `for chan in 1:n, (u,v) in uv_indices` and
# calls `data[u, v, chan]` — O(n_channels · |region|) file opens.
# This override reads one slice per channel: O(n_channels) opens.
function mean_region_spectrum(
    data::AbstractLazyFITS{T,3},
    axis::Integer,
    uv_indices;
    mask::Union{Nothing,AbstractArray{Bool,3}} = nothing,
) where {T}
    1 <= axis <= 3 || throw(ArgumentError("axis must be 1, 2, or 3"))
    if mask !== nothing && size(mask) != size(data)
        throw(DimensionMismatch(
            "mean_region_spectrum: mask size $(size(mask)) does not match data size $(size(data))"))
    end
    n = size(data, axis)
    y = fill(Float32(NaN), n)
    isempty(uv_indices) && return y

    @inbounds for chan in 1:n
        s   = get_slice_view(data, axis, chan)   # one FITSIO read
        acc = 0.0
        cnt = 0
        if mask === nothing
            for (u, v) in uv_indices
                fv = Float32(s[u, v])
                if isfinite(fv)
                    acc += Float64(fv)
                    cnt += 1
                end
            end
        else
            for (u, v) in uv_indices
                mval = if axis == 1; mask[chan, u, v]
                       elseif axis == 2; mask[u, chan, v]
                       else; mask[u, v, chan]; end
                mval || continue
                fv = Float32(s[u, v])
                if isfinite(fv)
                    acc += Float64(fv)
                    cnt += 1
                end
            end
        end
        y[chan] = cnt == 0 ? Float32(NaN) : Float32(acc / cnt)
    end
    return y
end

export LazyFITSImage, LazyFITSCube, AbstractLazyFITS
export read_slice!, prefetch_slice!, read_spectrum, open_lazy_fits
