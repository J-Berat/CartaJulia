# path: src/helpers/Images.jl
#
# NaN-aware Gaussian filtering and RGB image / pixel helpers. Extracted from
# helpers/Helpers.jl as part of the modular split — included from
# helpers/Helpers.jl. Public names (`nan_gaussian_filter`, `is_rgb_like`,
# `rgb_image`, `as_rgb_image`, `as_rgb_pixels`) are exported from there.

############################
# Filtering
############################

"""
    nan_gaussian_filter(img, sigma) -> Matrix{Float32}

Gaussian smoothing for projected maps with NaNs. Finite values are filtered and
renormalized by a filtered validity mask, so invalid/outside pixels do not bleed
into the map. Invalid output pixels stay `NaN32`.
"""
function nan_gaussian_filter(img::AbstractMatrix, sigma::Real)
    σ = Float32(sigma)
    σ <= 0 && return Float32.(img)
    values = similar(img, Float32)
    weights = similar(img, Float32)
    @inbounds for i in eachindex(img)
        v = Float32(img[i])
        if isfinite(v)
            values[i] = v
            weights[i] = 1f0
        else
            values[i] = 0f0
            weights[i] = 0f0
        end
    end
    k = ImageFiltering.Kernel.gaussian((σ, σ))
    smooth_values = imfilter(values, k)
    smooth_weights = imfilter(weights, k)
    out = similar(values, Float32)
    @inbounds for i in eachindex(out)
        w = smooth_weights[i]
        out[i] = w > 1f-6 ? Float32(smooth_values[i] / w) : NaN32
    end
    return out
end

############################
# RGB helpers
############################

is_rgb_like(x) = x isa AbstractArray && (
    eltype(x) <: Colorant ||
    (ndims(x) == 3 && (size(x, 1) in (3, 4) || size(x, 3) in (3, 4))) ||
    (ndims(x) == 2 && (size(x, 1) in (3, 4) || size(x, 2) in (3, 4)) && valid_healpix_npix(maximum(size(x))) > 0)
)

_unit_channel(v) = begin
    x = Float32(v)
    isfinite(x) ? clamp(x, 0f0, 1f0) : 0f0
end

function _normalize_rgb_channel(ch, mode::Symbol)
    out = similar(ch, Float32)
    if mode === :none
        @inbounds for i in eachindex(ch)
            out[i] = _unit_channel(ch[i])
        end
        return out
    elseif mode === :symmetric
        m = 0f0
        @inbounds for v in ch
            fv = Float32(v)
            isfinite(fv) && (m = max(m, abs(fv)))
        end
        if m == 0f0
            fill!(out, 0.5f0)
        else
            @inbounds for i in eachindex(ch)
                fv = Float32(ch[i])
                out[i] = isfinite(fv) ? clamp(0.5f0 + 0.5f0 * fv / m, 0f0, 1f0) : 0f0
            end
        end
        return out
    elseif mode === :minmax
        lo, hi = clamped_extrema(ch)
        span = hi - lo
        if span == 0f0
            fill!(out, 0.5f0)
        else
            @inbounds for i in eachindex(ch)
                fv = Float32(ch[i])
                out[i] = isfinite(fv) ? clamp((fv - lo) / span, 0f0, 1f0) : 0f0
            end
        end
        return out
    else
        throw(ArgumentError("RGB normalization must be :none, :minmax, or :symmetric."))
    end
end

"""
    rgb_image(r, g, b; normalize=:symmetric) -> Matrix{RGBf}

Build a display-ready RGB image from three scalar channels of the same size.
`normalize` can be `:symmetric`, `:minmax`, or `:none`.
"""
function rgb_image(r::AbstractMatrix, g::AbstractMatrix, b::AbstractMatrix; normalize::Symbol = :symmetric)
    (size(r) == size(g) && size(r) == size(b)) || throw(ArgumentError("RGB channels must have identical sizes."))
    R = _normalize_rgb_channel(r, normalize)
    G = _normalize_rgb_channel(g, normalize)
    B = _normalize_rgb_channel(b, normalize)
    out = Matrix{RGBf}(undef, size(R))
    @inbounds for i in eachindex(R)
        out[i] = RGBf(R[i], G[i], B[i])
    end
    return out
end

"""
    as_rgb_image(img) -> AbstractMatrix{<:Colorant}

Accept either a 2D colorant matrix or a numeric 3/4-channel stack with channels
in the first or last dimension. Numeric channels are interpreted in `[0, 1]`.
"""
function as_rgb_image(img::AbstractMatrix{<:Colorant})
    return img
end

function as_rgb_image(img::AbstractArray)
    ndims(img) == 3 || throw(ArgumentError("RGB image must be a color matrix or a 3D stack with 3/4 channels."))
    if size(img, 1) in (3, 4)
        rows, cols = size(img, 2), size(img, 3)
        if size(img, 1) == 3
            return RGBf[
                RGBf(_unit_channel(img[1, i, j]), _unit_channel(img[2, i, j]), _unit_channel(img[3, i, j]))
                for i in 1:rows, j in 1:cols
            ]
        else
            return RGBAf[
                RGBAf(_unit_channel(img[1, i, j]), _unit_channel(img[2, i, j]), _unit_channel(img[3, i, j]), _unit_channel(img[4, i, j]))
                for i in 1:rows, j in 1:cols
            ]
        end
    elseif size(img, 3) in (3, 4)
        rows, cols = size(img, 1), size(img, 2)
        if size(img, 3) == 3
            return RGBf[
                RGBf(_unit_channel(img[i, j, 1]), _unit_channel(img[i, j, 2]), _unit_channel(img[i, j, 3]))
                for i in 1:rows, j in 1:cols
            ]
        else
            return RGBAf[
                RGBAf(_unit_channel(img[i, j, 1]), _unit_channel(img[i, j, 2]), _unit_channel(img[i, j, 3]), _unit_channel(img[i, j, 4]))
                for i in 1:rows, j in 1:cols
            ]
        end
    else
        throw(ArgumentError("RGB stack must have 3 or 4 channels in the first or last dimension."))
    end
end

"""
    as_rgb_pixels(pixels) -> Vector{<:Colorant}

Accept a HEALPix RGB vector or a numeric `npix×3`, `npix×4`, `3×npix`, or
`4×npix` array. Numeric channels are interpreted in `[0, 1]`.
"""
function as_rgb_pixels(pixels::AbstractVector{<:Colorant})
    valid_healpix_npix(length(pixels)) > 0 || throw(ArgumentError("RGB HEALPix vector length must be 12*nside^2."))
    return pixels
end

function as_rgb_pixels(pixels::AbstractMatrix)
    rows, cols = size(pixels)
    if cols in (3, 4) && valid_healpix_npix(rows) > 0
        if cols == 3
            return RGBf[RGBf(_unit_channel(pixels[i, 1]), _unit_channel(pixels[i, 2]), _unit_channel(pixels[i, 3])) for i in 1:rows]
        else
            return RGBAf[RGBAf(_unit_channel(pixels[i, 1]), _unit_channel(pixels[i, 2]), _unit_channel(pixels[i, 3]), _unit_channel(pixels[i, 4])) for i in 1:rows]
        end
    elseif rows in (3, 4) && valid_healpix_npix(cols) > 0
        if rows == 3
            return RGBf[RGBf(_unit_channel(pixels[1, i]), _unit_channel(pixels[2, i]), _unit_channel(pixels[3, i])) for i in 1:cols]
        else
            return RGBAf[RGBAf(_unit_channel(pixels[1, i]), _unit_channel(pixels[2, i]), _unit_channel(pixels[3, i]), _unit_channel(pixels[4, i])) for i in 1:cols]
        end
    else
        throw(ArgumentError("RGB HEALPix pixels must be a vector or a numeric npix×3/4 or 3/4×npix matrix."))
    end
end
