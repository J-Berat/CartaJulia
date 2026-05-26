# path: src/helpers/Slicing.jl
#
# Cube → slice mapping primitives (`ijk_to_uv`, `uv_to_ijk`, `get_slice_view`,
# `get_slice_copy`, `get_slice`), region selection (`region_uv_indices`,
# `mean_region_spectrum`), small stats helper (`finite_mean_std`) and the
# dual-view comparison product (`dual_view_product`) and visual resampling
# helpers. Extracted from helpers/Helpers.jl.


############################
# Mapping / Slicing
############################

"""
    ijk_to_uv(i, j, k, axis) -> (u, v)

Map 3D voxel → 2D slice coords.
axis=1 ⇒ (u=j, v=k), axis=2 ⇒ (u=i, v=k), axis=3 ⇒ (u=i, v=j).
"""
@inline function ijk_to_uv(i::Int, j::Int, k::Int, axis::Int)
    axis == 1 && return (j, k)  # (y,z)
    axis == 2 && return (i, k)  # (x,z)
    return (i, j)               # (x,y)
end

"""
    uv_to_ijk(u, v, axis, idx) -> (i, j, k)

Inverse: 2D coords + slice index → 3D voxel.
"""
@inline function uv_to_ijk(u::Int, v::Int, axis::Int, idx::Int)
    axis == 1 && return (idx, u, v)
    axis == 2 && return (u, idx, v)
    return (u, v, idx)
end

"""
    get_slice_view(data::AbstractArray{T,3}, axis, idx) -> SubArray

Return a non-allocating 2D view into `data` along `axis` at index `idx`.
Orientation is consistent with [`ijk_to_uv`](@ref): `axis==1` ⇒ (y,z),
`axis==2` ⇒ (x,z), `axis==3` ⇒ (x,y).

Mutating the returned view will mutate the underlying cube. Use
[`get_slice_copy`](@ref) when an independent buffer is required.
"""
function get_slice_view(data::AbstractArray{T,3}, axis::Integer, idx::Integer) where {T}
    1 <= axis <= 3 || throw(ArgumentError("MANTA: axis must be 1, 2 or 3, got $(axis)"))
    1 <= idx <= size(data, axis) || throw(BoundsError(data, (axis, idx)))
    if axis == 1
        return @view data[idx, :, :]   # (y, z)
    elseif axis == 2
        return @view data[:, idx, :]   # (x, z)
    else
        return @view data[:, :, idx]   # (x, y)
    end
end

"""
    get_slice_copy(data::AbstractArray{T,3}, axis, idx) -> Array

Return a freshly allocated 2D slice. Equivalent to `copy(get_slice_view(...))`.
Use this when the caller needs to mutate the slice without touching the cube.
"""
get_slice_copy(data::AbstractArray, axis::Integer, idx::Integer) =
    copy(get_slice_view(data, axis, idx))

"""
    get_slice(data, axis, idx) -> AbstractMatrix

Backwards-compatible slice accessor. Historically copying; kept as a thin
alias over [`get_slice_copy`](@ref) so existing callers in the cube viewer
keep their independent buffers. New code should pick `get_slice_view` when
no mutation is needed.
"""
get_slice(data::AbstractArray, axis::Integer, idx::Integer) =
    get_slice_copy(data, axis, idx)

"""
    region_uv_indices(u_max, v_max, x0, y0, x1, y1, shape) -> Vector{Tuple{Int,Int}}

Return `(u, v)` pixels inside a drawn region. `x` maps to `v`, `y` maps
to `u`, matching the image axis convention.
"""
function region_uv_indices(
    u_max::Int,
    v_max::Int,
    x0::Real,
    y0::Real,
    x1::Real,
    y1::Real,
    shape::Symbol,
)
    if u_max < 1 || v_max < 1
        return Tuple{Int,Int}[]
    end
    if shape === :circle
        cx, cy = Float64(x0), Float64(y0)
        r = hypot(Float64(x1) - cx, Float64(y1) - cy)
        r < 0.5 && return Tuple{Int,Int}[]
        umin = clamp(Int(floor(cy - r)), 1, u_max)
        umax = clamp(Int(ceil(cy + r)), 1, u_max)
        vmin = clamp(Int(floor(cx - r)), 1, v_max)
        vmax = clamp(Int(ceil(cx + r)), 1, v_max)
        rr = r * r
        return [(u, v) for u in umin:umax for v in vmin:vmax if (v - cx)^2 + (u - cy)^2 <= rr]
    else
        xmin, xmax = minmax(Float64(x0), Float64(x1))
        ymin, ymax = minmax(Float64(y0), Float64(y1))
        if abs(xmax - xmin) < 0.5 || abs(ymax - ymin) < 0.5
            return Tuple{Int,Int}[]
        end
        umin = clamp(Int(round(ymin)), 1, u_max)
        umax = clamp(Int(round(ymax)), 1, u_max)
        vmin = clamp(Int(round(xmin)), 1, v_max)
        vmax = clamp(Int(round(xmax)), 1, v_max)
        return [(u, v) for u in umin:umax for v in vmin:vmax]
    end
end

"""
    mean_region_spectrum(data, axis, uv_indices; mask=nothing) -> Vector{Float32}

Average the spectrum along `axis` over a set of `(u, v)` pixels in the
current slice plane. Non-finite voxels are ignored channel by channel.

When `mask` is provided, it must be a 3-D `AbstractArray{Bool}` with the
same shape as `data`. Voxels where the mask is `false` are skipped
(treated identically to a non-finite value), so the returned average only
incorporates voxels accepted by both the finiteness test AND the mask.
Passing `nothing` (the default) preserves the legacy unmasked behaviour
exactly.
"""
function mean_region_spectrum(data::AbstractArray{T,3}, axis::Integer, uv_indices;
                              mask::Union{Nothing,AbstractArray{Bool,3}} = nothing) where {T}
    1 <= axis <= 3 || throw(ArgumentError("axis must be 1, 2, or 3"))
    if mask !== nothing && size(mask) != size(data)
        throw(DimensionMismatch(
            "mean_region_spectrum: mask size $(size(mask)) does not match data size $(size(data))"))
    end
    n = size(data, axis)
    y = fill(Float32(NaN), n)
    isempty(uv_indices) && return y
    if mask === nothing
        @inbounds for chan in 1:n
            acc = 0.0
            cnt = 0
            for (u, v) in uv_indices
                val = if axis == 1
                    data[chan, u, v]
                elseif axis == 2
                    data[u, chan, v]
                else
                    data[u, v, chan]
                end
                fv = Float32(val)
                if isfinite(fv)
                    acc += Float64(fv)
                    cnt += 1
                end
            end
            y[chan] = cnt == 0 ? Float32(NaN) : Float32(acc / cnt)
        end
    else
        @inbounds for chan in 1:n
            acc = 0.0
            cnt = 0
            for (u, v) in uv_indices
                m, val = if axis == 1
                    (mask[chan, u, v], data[chan, u, v])
                elseif axis == 2
                    (mask[u, chan, v], data[u, chan, v])
                else
                    (mask[u, v, chan], data[u, v, chan])
                end
                m || continue
                fv = Float32(val)
                if isfinite(fv)
                    acc += Float64(fv)
                    cnt += 1
                end
            end
            y[chan] = cnt == 0 ? Float32(NaN) : Float32(acc / cnt)
        end
    end
    return y
end

function finite_mean_std(vals)
    acc = 0.0
    cnt = 0
    @inbounds for v in vals
        x = Float64(v)
        if isfinite(x)
            acc += x
            cnt += 1
        end
    end
    cnt == 0 && return (NaN, NaN)
    μ = acc / cnt
    acc2 = 0.0
    @inbounds for v in vals
        x = Float64(v)
        if isfinite(x)
            acc2 += (x - μ)^2
        end
    end
    σ = cnt <= 1 ? 0.0 : sqrt(acc2 / (cnt - 1))
    return (μ, σ)
end

"""
    dual_view_product(a, b, mode) -> Matrix{Float32}

Compute the right-hand dual-view product from primary slice `a` and secondary
slice `b`. `mode` accepts `:A`, `:B`, `:diff`, `:ratio`, or `:residuals`.
When the slices have different sizes, `b` is linearly resampled to `a`'s
grid for visual comparison.
"""
function dual_view_product(a::AbstractMatrix, b::AbstractMatrix, mode::Symbol)
    b_view = size(a) == size(b) ? b : resample_matrix_linear(b, size(a))
    out = similar(Float32.(a))
    if mode === :A
        copyto!(out, Float32.(a))
    elseif mode === :B
        copyto!(out, Float32.(b_view))
    elseif mode === :diff
        @inbounds for i in eachindex(out, a, b_view)
            out[i] = Float32(a[i]) - Float32(b_view[i])
        end
    elseif mode === :ratio
        @inbounds for i in eachindex(out, a, b_view)
            den = Float32(b_view[i])
            num = Float32(a[i])
            out[i] = isfinite(num) && isfinite(den) && den != 0f0 ? num / den : NaN32
        end
    elseif mode === :residuals
        diff = similar(out)
        @inbounds for i in eachindex(diff, a, b_view)
            diff[i] = Float32(a[i]) - Float32(b_view[i])
        end
        μ, σ = finite_mean_std(diff)
        if !isfinite(σ) || σ <= 0
            fill!(out, NaN32)
        else
            @inbounds for i in eachindex(out, diff)
                x = Float64(diff[i])
                out[i] = isfinite(x) ? Float32((x - μ) / σ) : NaN32
            end
        end
    else
        throw(ArgumentError("unknown dual view mode: $(mode)"))
    end
    return out
end

"""
    resample_matrix_linear(A, target_size) -> Matrix{Float32}

Linearly resample a 2-D array to `target_size`, matching endpoints along
each axis. Non-finite source pixels are ignored and yield `NaN32` only when
all contributing pixels are non-finite or the sampled coordinate falls
outside the source image.
"""
function resample_matrix_linear(A::AbstractMatrix{<:Real}, target_size::Tuple{<:Integer,<:Integer})
    tx, ty = Int.(target_size)
    tx >= 1 && ty >= 1 || throw(ArgumentError("target_size entries must be positive"))
    sx, sy = size(A)
    out = Matrix{Float32}(undef, tx, ty)
    @inbounds for j in 1:ty
        y = _resample_coord(j, sy, ty)
        for i in 1:tx
            x = _resample_coord(i, sx, tx)
            out[i, j] = _linear_sample_2d(A, x, y)
        end
    end
    return out
end

"""
    resample_cube_linear(A, target_size) -> Array{Float32,3}

Trilinearly resample a cube to `target_size`, matching endpoints along every
axis. This is intended for visual comparison products, not flux-conserving
scientific reprojection.
"""
function resample_cube_linear(A::AbstractArray{<:Real,3}, target_size::NTuple{3,<:Integer})
    tx, ty, tz = Int.(target_size)
    tx >= 1 && ty >= 1 && tz >= 1 || throw(ArgumentError("target_size entries must be positive"))
    sx, sy, sz = size(A)
    out = Array{Float32,3}(undef, tx, ty, tz)
    @inbounds for k in 1:tz
        z = _resample_coord(k, sz, tz)
        for j in 1:ty
            y = _resample_coord(j, sy, ty)
            for i in 1:tx
                x = _resample_coord(i, sx, tx)
                out[i, j, k] = _linear_sample_3d(A, x, y, z)
            end
        end
    end
    return out
end

"""
    reproject_cube_linear(A, source_wcs, reference_wcs, target_size)

Resample `A` onto the reference cube's pixel grid using the lightweight
per-axis linear WCS (`CRVAL/CRPIX/CDELT`). This deliberately ignores CD/PC
cross-terms; callers should use it only when the axes are separable.
"""
function reproject_cube_linear(
    A::AbstractArray{<:Real,3},
    source_wcs,
    reference_wcs,
    target_size::NTuple{3,<:Integer},
)
    tx, ty, tz = Int.(target_size)
    tx >= 1 && ty >= 1 && tz >= 1 || throw(ArgumentError("target_size entries must be positive"))
    out = Array{Float32,3}(undef, tx, ty, tz)
    @inbounds for k in 1:tz
        z = source_pixel_coord(source_wcs, 3, world_coord(reference_wcs, 3, k))
        for j in 1:ty
            y = source_pixel_coord(source_wcs, 2, world_coord(reference_wcs, 2, j))
            for i in 1:tx
                x = source_pixel_coord(source_wcs, 1, world_coord(reference_wcs, 1, i))
                out[i, j, k] = _linear_sample_3d(A, x, y, z)
            end
        end
    end
    return out
end

source_pixel_coord(wcs, dim::Integer, world::Real) =
    has_wcs(wcs, dim) ? (Float64(world) - wcs[dim].crval) / wcs[dim].cdelt + wcs[dim].crpix : Float64(world)

function _resample_coord(i::Integer, source_n::Integer, target_n::Integer)
    source_n <= 1 && return 1.0
    target_n <= 1 && return 1.0
    return 1.0 + (Float64(i) - 1.0) * (Float64(source_n) - 1.0) / (Float64(target_n) - 1.0)
end

function _linear_sample_2d(A, x::Real, y::Real)
    sx, sy = size(A)
    (1.0 <= x <= sx && 1.0 <= y <= sy) || return NaN32
    x0 = clamp(Int(floor(x)), 1, sx)
    y0 = clamp(Int(floor(y)), 1, sy)
    x1 = min(x0 + 1, sx)
    y1 = min(y0 + 1, sy)
    fx = Float64(x) - x0
    fy = Float64(y) - y0
    acc = 0.0
    wsum = 0.0
    @inbounds for (ii, wx) in ((x0, 1.0 - fx), (x1, fx))
        wx == 0.0 && continue
        for (jj, wy) in ((y0, 1.0 - fy), (y1, fy))
            w = wx * wy
            w == 0.0 && continue
            v = Float64(A[ii, jj])
            isfinite(v) || continue
            acc += w * v
            wsum += w
        end
    end
    return wsum == 0.0 ? NaN32 : Float32(acc / wsum)
end

function _linear_sample_3d(A, x::Real, y::Real, z::Real)
    sx, sy, sz = size(A)
    (1.0 <= x <= sx && 1.0 <= y <= sy && 1.0 <= z <= sz) || return NaN32
    x0 = clamp(Int(floor(x)), 1, sx)
    y0 = clamp(Int(floor(y)), 1, sy)
    z0 = clamp(Int(floor(z)), 1, sz)
    x1 = min(x0 + 1, sx)
    y1 = min(y0 + 1, sy)
    z1 = min(z0 + 1, sz)
    fx = Float64(x) - x0
    fy = Float64(y) - y0
    fz = Float64(z) - z0
    acc = 0.0
    wsum = 0.0
    @inbounds for (ii, wx) in ((x0, 1.0 - fx), (x1, fx))
        wx == 0.0 && continue
        for (jj, wy) in ((y0, 1.0 - fy), (y1, fy))
            wy == 0.0 && continue
            for (kk, wz) in ((z0, 1.0 - fz), (z1, fz))
                w = wx * wy * wz
                w == 0.0 && continue
                v = Float64(A[ii, jj, kk])
                isfinite(v) || continue
                acc += w * v
                wsum += w
            end
        end
    end
    return wsum == 0.0 ? NaN32 : Float32(acc / wsum)
end
