# path: src/loaders/LazyHDF5.jl
#
# Lazy HDF5 arrays.
#
# `load_hdf5(...; lazy = true)` wraps 2-D images and 3-D cubes without reading
# the full dataset. Reads reopen the file and fetch only the requested
# hyperslab.  For chunked cubes, slice reads cache the chunk-aligned slab along
# the active axis so adjacent channel navigation can reuse the same HDF5 chunk
# work when the file layout supports it.

using HDF5

abstract type AbstractLazyHDF5{T,N} <: AbstractArray{T,N} end

mutable struct LazyHDF5Image{T<:Real} <: AbstractLazyHDF5{T,2}
    path::String
    address::String
    sz::NTuple{2,Int}
    chunk::Union{Nothing,NTuple{2,Int}}
end

mutable struct LazyHDF5Cube{T<:Real} <: AbstractLazyHDF5{T,3}
    path::String
    address::String
    sz::NTuple{3,Int}
    chunk::Union{Nothing,NTuple{3,Int}}
    cache_axis::Int
    cache_range::UnitRange{Int}
    cache_data::Union{Nothing,Array{T,3}}
    prefetch_axis::Int
    prefetch_range::UnitRange{Int}
    prefetch_ch::Union{Nothing,Channel{Array{T,3}}}
end

LazyHDF5Cube{T}(path::AbstractString, address::AbstractString,
                sz::NTuple{3,Integer},
                chunk::Union{Nothing,NTuple{3,Int}} = nothing) where {T<:Real} =
    LazyHDF5Cube{T}(String(path), String(address), Int.(sz), chunk,
                    0, 1:0, nothing, 0, 1:0, nothing)

Base.size(L::LazyHDF5Image) = L.sz
Base.eltype(::LazyHDF5Image{T}) where {T} = T
Base.size(L::LazyHDF5Cube) = L.sz
Base.eltype(::LazyHDF5Cube{T}) where {T} = T

function Base.getindex(L::LazyHDF5Image{T}, i::Int, j::Int) where {T}
    h5open(L.path, "r") do f
        return T(f[L.address][i, j])
    end
end

function Base.collect(L::LazyHDF5Image{T}) where {T}
    h5open(L.path, "r") do f
        return Array{T,2}(read(f[L.address]))
    end
end

function Base.getindex(L::LazyHDF5Cube{T}, i::Int, j::Int, k::Int) where {T}
    h5open(L.path, "r") do f
        return T(f[L.address][i, j, k])
    end
end

function _hdf5_slice_range(L::LazyHDF5Cube, axis::Int, idx::Int)
    ch = L.chunk
    ch === nothing && return idx:idx
    width = ch[axis]
    # Contiguous datasets, or chunks spanning the whole active axis, would
    # make this cache materialise too much.  In those cases read one slice and
    # let HDF5's own chunk cache do the lower-level work.
    if width <= 1 || width >= L.sz[axis]
        return idx:idx
    end
    lo = ((idx - 1) ÷ width) * width + 1
    hi = min(lo + width - 1, L.sz[axis])
    return lo:hi
end

function _read_hdf5_slab(L::LazyHDF5Cube{T}, axis::Int, r::UnitRange{Int}) where {T}
    nx, ny, nz = L.sz
    h5open(L.path, "r") do f
        ds = f[L.address]
        raw = if axis == 1
            ds[r, 1:ny, 1:nz]
        elseif axis == 2
            ds[1:nx, r, 1:nz]
        elseif axis == 3
            ds[1:nx, 1:ny, r]
        else
            throw(ArgumentError("axis must be 1, 2 or 3, got $(axis)"))
        end
        return Array{T,3}(raw)
    end
end

function _hdf5_slice_from_block(L::LazyHDF5Cube{T}, axis::Int, idx::Int,
                                r::UnitRange{Int}, block::Array{T,3}) where {T}
    offset = idx - first(r) + 1
    if axis == 1
        return Matrix{T}(@view block[offset, :, :])
    elseif axis == 2
        return Matrix{T}(@view block[:, offset, :])
    else
        return Matrix{T}(@view block[:, :, offset])
    end
end

"""
    read_slice!(L::LazyHDF5Cube, axis, idx) -> Matrix

Materialise a 2-D cube slice from an HDF5 dataset.  Chunked datasets cache a
chunk-aligned slab along the active axis when doing so keeps the cached block
smaller than the full axis. If a matching prefetch is in flight, this waits for
that slab and promotes it into the normal cache.
"""
function read_slice!(L::LazyHDF5Cube{T}, axis::Integer, idx::Integer) where {T}
    a, k = Int(axis), Int(idx)
    a in (1, 2, 3) || throw(ArgumentError("axis must be 1, 2 or 3, got $(a)"))
    1 <= k <= L.sz[a] || throw(BoundsError(L, (a, k)))

    if a == L.cache_axis && k in L.cache_range && L.cache_data !== nothing
        return _hdf5_slice_from_block(L, a, k, L.cache_range, L.cache_data)
    end

    r = _hdf5_slice_range(L, a, k)
    block = if a == L.prefetch_axis && r == L.prefetch_range &&
               L.prefetch_ch !== nothing
        ch = L.prefetch_ch
        L.prefetch_ch = nothing
        if isopen(ch)
            try
                take!(ch)
            catch
                _read_hdf5_slab(L, a, r)
            end
        else
            _read_hdf5_slab(L, a, r)
        end
    else
        if L.prefetch_ch !== nothing
            close(L.prefetch_ch)
            L.prefetch_ch = nothing
        end
        _read_hdf5_slab(L, a, r)
    end
    L.cache_axis = a
    L.cache_range = r
    L.cache_data = block
    return _hdf5_slice_from_block(L, a, k, r, block)
end

"""
    prefetch_slice!(L::LazyHDF5Cube, axis, idx)

Start loading the chunk-aligned slab containing `(axis, idx)` in the background.
Safe to call speculatively: already cached slabs, duplicate prefetches, invalid
axes, and out-of-bounds indices are no-ops.
"""
function prefetch_slice!(L::LazyHDF5Cube{T},
                         axis::Integer, idx::Integer) where {T}
    a, k = Int(axis), Int(idx)
    a in (1, 2, 3) || return nothing
    1 <= k <= L.sz[a] || return nothing

    r = _hdf5_slice_range(L, a, k)
    a == L.cache_axis && r == L.cache_range && L.cache_data !== nothing &&
        return nothing

    if a == L.prefetch_axis && r == L.prefetch_range && L.prefetch_ch !== nothing
        return nothing
    end

    if L.prefetch_ch !== nothing
        close(L.prefetch_ch)
    end

    ch = Channel{Array{T,3}}(1)
    L.prefetch_axis = a
    L.prefetch_range = r
    L.prefetch_ch = ch

    @async begin
        try
            put!(ch, _read_hdf5_slab(L, a, r))
        catch
            isopen(ch) && close(ch)
        end
    end

    return nothing
end

"""
    prefetch_adjacent!(L::LazyHDF5Cube, axis, idx_new, last_idx) -> Int

Direction-aware speculative prefetch for HDF5 cube scrolling. Infers the scroll
direction from `idx_new` and `last_idx`, then starts an async prefetch for the
neighbouring slice when it is in bounds.
"""
function prefetch_adjacent!(L::LazyHDF5Cube, axis::Integer, idx_new::Integer,
                            last_idx::Integer)
    a = Int(axis)
    id = Int(idx_new)
    a in (1, 2, 3) || return id
    dir = id > Int(last_idx) ? 1 : id < Int(last_idx) ? -1 : 1
    nbr = id + dir
    1 <= nbr <= L.sz[a] && prefetch_slice!(L, a, nbr)
    return id
end

Base.view(L::LazyHDF5Cube, ::Colon, ::Colon, k::Int) = read_slice!(L, 3, k)
Base.view(L::LazyHDF5Cube, ::Colon, j::Int, ::Colon) = read_slice!(L, 2, j)
Base.view(L::LazyHDF5Cube, i::Int, ::Colon, ::Colon) = read_slice!(L, 1, i)

Base.view(L::LazyHDF5Cube{T}, i::Int, j::Int, ::Colon) where {T} =
    h5open(L.path, "r") do f
        Vector{T}(vec(f[L.address][i:i, j:j, 1:L.sz[3]]))
    end
Base.view(L::LazyHDF5Cube{T}, i::Int, ::Colon, k::Int) where {T} =
    h5open(L.path, "r") do f
        Vector{T}(vec(f[L.address][i:i, 1:L.sz[2], k:k]))
    end
Base.view(L::LazyHDF5Cube{T}, ::Colon, j::Int, k::Int) where {T} =
    h5open(L.path, "r") do f
        Vector{T}(vec(f[L.address][1:L.sz[1], j:j, k:k]))
    end

function read_spectrum(L::LazyHDF5Cube{T}, i::Integer, j::Integer) where {T}
    nx, ny, nz = L.sz
    1 <= i <= nx || throw(BoundsError(L, (i, j, :)))
    1 <= j <= ny || throw(BoundsError(L, (i, j, :)))
    h5open(L.path, "r") do f
        Vector{T}(vec(f[L.address][i:i, j:j, 1:nz]))
    end
end

function Base.collect(L::LazyHDF5Cube{T}) where {T}
    h5open(L.path, "r") do f
        Array{T,3}(read(f[L.address]))
    end
end

as_float32(x::AbstractLazyHDF5) = x

function moment_map(
    data::AbstractLazyHDF5{T,3},
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
    out = fill(NaN32, u_max, v_max)
    chan_vec = [c for c in channels if 1 <= c <= size(data, axis)]
    isempty(chan_vec) && return out

    x = Float32[Float32(coords[c]) for c in chan_vec]
    widths = _channel_widths(x, dx)
    thr = if nsigma !== nothing && sigma !== nothing
        Float64(nsigma) * Float64(sigma)
    elseif nsigma !== nothing
        @warn "moment_map on lazy HDF5: `nsigma` without explicit `sigma` falls " *
              "back to absolute `threshold=$(threshold)` (per-pixel MAD estimation " *
              "would require scanning full spectra for every pixel). Pass " *
              "`sigma=<value>` to enable σ-clipping with lazy data."
        Float64(threshold)
    else
        Float64(threshold)
    end

    acc0 = zeros(Float64, u_max, v_max)
    acc1 = zeros(Float64, u_max, v_max)
    any_hit = falses(u_max, v_max)
    m1_map = zeros(Float64, u_max, v_max)

    for (n, c) in pairs(chan_vec)
        s = get_slice_view(data, axis, c)
        xi = Float64(x[n])
        wi = Float64(widths[n])
        @inbounds for v in 1:v_max, u in 1:u_max
            if mask !== nothing
                mval = if axis == 1; mask[c, u, v]
                       elseif axis == 2; mask[u, c, v]
                       else; mask[u, v, c]; end
                mval || continue
            end
            yi = Float64(Float32(s[u, v]))
            if isfinite(yi) && yi > thr
                any_hit[u, v] = true
                acc0[u, v] += yi * wi
                order >= 1 && (acc1[u, v] += yi * xi * wi)
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

    acc2 = zeros(Float64, u_max, v_max)
    for (n, c) in pairs(chan_vec)
        s = get_slice_view(data, axis, c)
        xi = Float64(x[n])
        wi = Float64(widths[n])
        @inbounds for v in 1:v_max, u in 1:u_max
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
    @inbounds for v in 1:v_max, u in 1:u_max
        if any_hit[u, v] && acc0[u, v] != 0.0
            m2_2 = acc2[u, v] / acc0[u, v]
            out[u, v] = m2_2 >= 0.0 ? Float32(sqrt(m2_2)) : NaN32
        end
    end
    return out
end

function mean_region_spectrum(
    data::AbstractLazyHDF5{T,3},
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
        s = get_slice_view(data, axis, chan)
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

export LazyHDF5Image, LazyHDF5Cube, AbstractLazyHDF5
