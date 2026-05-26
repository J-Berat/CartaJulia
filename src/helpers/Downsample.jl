# path: src/helpers/Downsample.jl
#
# Decimation helpers for display. GLMakie heatmaps stutter on very large
# 2D arrays (>~ 8k × 8k); when the user only has 1920 px of screen, there
# is no point shipping the full array to the GPU.
#
# Strategy:
#   * Pick an integer stride so that each dim ≤ `max_pixels` (default 4096).
#   * Use block-mean by default (preserves photometry better than naive
#     subsampling) and fall back to subsampling when the dtype isn't
#     floating-point or NaN-aware reduction is too expensive.
#
# The functions here are pure and side-effect free, so they can be unit
# tested with `activate_gl=false`.

"""
    downsample_factor(sz, max_pixels)

Integer stride that brings every dimension of `sz` ≤ `max_pixels`.
Returns `1` (no downsampling needed) when the array already fits.
"""
function downsample_factor(sz::Tuple{Vararg{Integer}}, max_pixels::Integer)
    max_pixels > 0 || throw(ArgumentError("max_pixels must be positive"))
    isempty(sz) && return 1
    m = maximum(Int.(sz))
    m <= Int(max_pixels) && return 1
    return cld(m, Int(max_pixels))
end

downsample_factor(A::AbstractArray, max_pixels::Integer) =
    downsample_factor(size(A), max_pixels)

"""
    downsample_block_mean(A, stride)

Block-mean downsampling of a 2D array `A` with the given integer `stride`.
Trailing partial blocks are reduced over whatever remains, so the output
size is `cld.(size(A), stride)`. NaNs are ignored.

Returns a new `Matrix{Float32}` (so the GPU upload is always small).
"""
function downsample_block_mean(A::AbstractMatrix{<:Real}, stride::Integer)
    s = Int(stride)
    s >= 1 || throw(ArgumentError("stride must be ≥ 1"))
    s == 1 && return Float32.(A)
    nx, ny = size(A)
    ox, oy = cld(nx, s), cld(ny, s)
    out = Matrix{Float32}(undef, ox, oy)
    @inbounds for j in 1:oy
        j0 = (j - 1) * s + 1
        j1 = min(j * s, ny)
        for i in 1:ox
            i0 = (i - 1) * s + 1
            i1 = min(i * s, nx)
            sumv = 0.0
            n = 0
            for jj in j0:j1, ii in i0:i1
                v = Float64(A[ii, jj])
                if isfinite(v)
                    sumv += v
                    n += 1
                end
            end
            out[i, j] = n == 0 ? Float32(NaN) : Float32(sumv / n)
        end
    end
    return out
end

"""
    downsample_subsample(A, stride)

Cheaper alternative: stride-based subsampling (no averaging). Useful when
photometric accuracy isn't required (e.g. for a quick contrast preview).
"""
function downsample_subsample(A::AbstractMatrix{<:Real}, stride::Integer)
    s = Int(stride)
    s >= 1 || throw(ArgumentError("stride must be ≥ 1"))
    s == 1 && return Float32.(A)
    return Float32.(@view A[1:s:end, 1:s:end])
end

"""
    auto_downsample(A; max_pixels=4096, mode=:block_mean)

Convenience entry point used by the 2D view. Returns the downsampled array
(or `A` itself if no downsampling is needed) together with the stride used,
so the caller can label the axes appropriately.

`mode` is `:block_mean` (default) or `:subsample`.
"""
function auto_downsample(A::AbstractMatrix{<:Real};
                         max_pixels::Integer = 4096,
                         mode::Symbol = :block_mean)
    s = downsample_factor(A, max_pixels)
    s == 1 && return (Float32.(A), 1)
    out = mode === :subsample ? downsample_subsample(A, s) :
                                downsample_block_mean(A, s)
    return (out, s)
end

"""
    auto_downsample(A::AbstractArray{T,3}; ...)

3D variant: downsample only the first two dims (spectral axis is preserved
because a slice viewer needs every channel intact).
"""
function auto_downsample(A::AbstractArray{<:Real,3};
                         max_pixels::Integer = 4096,
                         mode::Symbol = :block_mean)
    s = downsample_factor(size(A)[1:2], max_pixels)
    s == 1 && return (Float32.(A), 1)
    nx, ny, nz = size(A)
    ox, oy = cld(nx, s), cld(ny, s)
    out = Array{Float32,3}(undef, ox, oy, nz)
    for k in 1:nz
        out[:, :, k] = mode === :subsample ?
            downsample_subsample(@view(A[:, :, k]), s) :
            downsample_block_mean(@view(A[:, :, k]), s)
    end
    return (out, s)
end

export downsample_factor, downsample_block_mean, downsample_subsample
export auto_downsample
