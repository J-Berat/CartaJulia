# path: src/helpers/Moments.jl
#
# Spectral moment integrals along a cube axis (`moments`, `moments_map`,
# `moment_map`, `moment_vectors`, `filtered_cube_by_slice`) and their
# numerical kernels (`_channel_value`, `_channel_widths`, `_robust_sigma`,
# `_resolve_threshold`, `_pixel_moment`). Implements the M0 = Σ y·Δx
# integral contract. Extracted from helpers/Helpers.jl.


function _channel_value(data, axis::Integer, u::Integer, v::Integer, chan::Integer)
    if axis == 1
        return data[chan, u, v]
    elseif axis == 2
        return data[u, chan, v]
    else
        return data[u, v, chan]
    end
end

"""
    _channel_widths(x, dx) -> Vector{Float64}

Resolve effective channel widths `Δx_i` used for moment integration.

* `dx === nothing`: infer from `x` using centered differences for interior
  channels and a half-step for the edges (one-sided for the boundary case).
  Falls back to `ones(length(x))` if `length(x) <= 1`.
* `dx` is a `Number`: uniform width applied to every channel.
* `dx` is an iterable: per-channel width vector (must have `length(x)` entries).
"""
function _channel_widths(x, dx)
    n = length(x)
    if dx === nothing
        n == 0 && return Float64[]
        n == 1 && return Float64[1.0]
        widths = Vector{Float64}(undef, n)
        widths[1] = abs(Float64(x[2]) - Float64(x[1]))
        widths[n] = abs(Float64(x[n]) - Float64(x[n-1]))
        @inbounds for i in 2:n-1
            widths[i] = 0.5 * abs(Float64(x[i+1]) - Float64(x[i-1]))
        end
        return widths
    elseif dx isa Number
        return fill(Float64(dx), n)
    else
        v = collect(Float64, dx)
        length(v) == n || throw(DimensionMismatch("dx length must match x length"))
        return v
    end
end

"""
    _robust_sigma(y) -> Float64

MAD-based robust standard-deviation estimator. Returns `NaN` if `y` has no
finite samples. The 1.4826 factor converts the median absolute deviation to
a Gaussian-equivalent σ.
"""
function _robust_sigma(y)
    finite = Float64[]
    @inbounds for v in y
        fv = Float64(v); isfinite(fv) && push!(finite, fv)
    end
    isempty(finite) && return NaN
    med = median(finite)
    mad = median(abs.(finite .- med))
    return 1.4826 * mad
end

"""
    _resolve_threshold(y, threshold, nsigma, sigma) -> Float64

Combine an absolute floor (`threshold`) with an optional sigma-clipping
specification. When `nsigma` is given, the effective threshold becomes
`nsigma * σ` where σ is either the user-supplied `sigma` or a MAD-based
robust estimate of σ from `y`. If σ cannot be estimated, the absolute
`threshold` is returned.
"""
function _resolve_threshold(y, threshold, nsigma, sigma)
    if nsigma === nothing
        return Float64(threshold)
    end
    σ = sigma === nothing ? _robust_sigma(y) : Float64(sigma)
    return isfinite(σ) && σ > 0 ? Float64(nsigma) * σ : Float64(threshold)
end

"""
    moments(y; x=1:length(y), threshold=0.0, nsigma=nothing, sigma=nothing,
            dx=nothing, channels=nothing) -> (m0, m1, m2)

Compute the zeroth, first and second moments of `y(x)` as proper
integrals along the spectral axis:

```
M0 = Σ y_i Δx_i                 (e.g. K·km/s when y is K and Δx is km/s)
M1 = Σ y_i x_i Δx_i / M0
M2 = sqrt( Σ y_i (x_i - M1)^2 Δx_i / M0 )
```

Only samples with `y_i > threshold` contribute. Set `nsigma` to clip below
`nsigma * σ` instead of an absolute value (σ defaults to a MAD-based robust
estimate of the spectrum, or the user-provided `sigma`). Pass `channels`
to restrict the sum to a specific spectral window. `dx` overrides the
auto-detected channel widths (scalar = uniform Δx, vector = per-channel).
"""
function moments(y; x = 1:length(y), threshold = 0.0, nsigma = nothing,
                 sigma = nothing, dx = nothing, channels = nothing)
    length(x) == length(y) || throw(DimensionMismatch("x and y must have the same length"))
    widths = _channel_widths(x, dx)
    in_window = if channels === nothing
        nothing
    else
        m = falses(length(y))
        for c in channels
            (c isa Integer && 1 <= c <= length(y)) && (m[c] = true)
        end
        m
    end
    thr = _resolve_threshold(y, threshold, nsigma, sigma)

    acc0 = 0.0
    acc1 = 0.0
    any_sample = false
    @inbounds for i in eachindex(y, x)
        in_window !== nothing && !in_window[i] && continue
        yi = Float64(y[i])
        if isfinite(yi) && yi > thr
            any_sample = true
            xi = Float64(x[i])
            w = widths[i]
            acc0 += yi * w
            acc1 += yi * xi * w
        end
    end
    (!any_sample || acc0 == 0.0) && return (NaN, NaN, NaN)
    m0 = acc0
    m1 = acc1 / m0
    acc2 = 0.0
    @inbounds for i in eachindex(y, x)
        in_window !== nothing && !in_window[i] && continue
        yi = Float64(y[i])
        if isfinite(yi) && yi > thr
            xi = Float64(x[i])
            w = widths[i]
            acc2 += yi * (xi - m1)^2 * w
        end
    end
    m2_2 = acc2 / m0
    m2 = m2_2 >= 0 ? sqrt(m2_2) : NaN
    return (m0, m1, m2)
end

"""
    moments_map(data, array; threshold=0.0, nsigma=nothing, sigma=nothing,
                dx=nothing, channels=nothing) -> (M0, M1, M2)

Compute moment 0/1/2 maps along the third dimension of `data`. Forwards all
threshold / Δx / channel-window controls to [`moments`](@ref).
"""
function moments_map(data::AbstractArray{T,3}, array;
                     threshold = 0.0, nsigma = nothing, sigma = nothing,
                     dx = nothing, channels = nothing) where {T}
    length(array) == size(data, 3) || throw(DimensionMismatch("array length must match size(data, 3)"))
    widths = _channel_widths(array, dx)
    M0 = Matrix{Float32}(undef, size(data, 1), size(data, 2))
    M1 = similar(M0)
    M2 = similar(M0)
    @inbounds for i in 1:size(data, 1), j in 1:size(data, 2)
        m0, m1, m2 = moments(@view(data[i, j, :]); x = array, threshold = threshold,
                             nsigma = nsigma, sigma = sigma, dx = widths,
                             channels = channels)
        M0[i, j] = Float32(m0)
        M1[i, j] = Float32(m1)
        M2[i, j] = Float32(m2)
    end
    return M0, M1, M2
end

"""
    _pixel_moment(order, y, x, widths, threshold, nsigma, sigma) -> Float64

Specialised single-moment kernel used by [`moment_map`](@ref). Returns just
the requested moment (0, 1 or 2), skipping the additional passes that
`moments` would do for the higher-order moments when they're not needed.

Semantics match [`moments`](@ref): same `Float64` accumulators, same
`_resolve_threshold` logic, same `acc2 / m0` formula for the dispersion.
"""
@inline function _pixel_moment(order::Integer, y, x, widths,
                               threshold, nsigma, sigma)::Float64
    thr = _resolve_threshold(y, threshold, nsigma, sigma)
    acc0 = 0.0
    acc1 = 0.0
    any_sample = false
    @inbounds for i in eachindex(y, x)
        yi = Float64(y[i])
        if isfinite(yi) && yi > thr
            any_sample = true
            xi = Float64(x[i])
            w = widths[i]
            acc0 += yi * w
            if order >= 1
                acc1 += yi * xi * w
            end
        end
    end
    (!any_sample || acc0 == 0.0) && return NaN
    order == 0 && return acc0
    m0 = acc0
    m1 = acc1 / m0
    order == 1 && return m1
    acc2 = 0.0
    @inbounds for i in eachindex(y, x)
        yi = Float64(y[i])
        if isfinite(yi) && yi > thr
            xi = Float64(x[i])
            w = widths[i]
            acc2 += yi * (xi - m1)^2 * w
        end
    end
    m2_2 = acc2 / m0
    return m2_2 >= 0 ? sqrt(m2_2) : NaN
end

"""
    moment_map(data, axis, order; coords=1:size(data, axis),
               channels=1:size(data, axis),
               threshold=0.0, nsigma=nothing, sigma=nothing, dx=nothing,
               mask=nothing)

Compute moment 0, 1, or 2 along `axis`, returning a 2D map in the same
orientation as `get_slice`. The moment definition is delegated to
[`moments`](@ref) — see its docstring for the integration convention and
the threshold / channel-window options.

`mask`, when provided, is a 3-D `AbstractArray{Bool}` with the same shape
as `data`. Voxels where the mask is `false` are NaN-ed out before entering
the moment accumulator, so they are skipped both by the `isfinite` test
and by the threshold predicate inside [`_pixel_moment`](@ref). Passing
`nothing` (the default) preserves the legacy unmasked behaviour exactly.
"""
function moment_map(
    data::AbstractArray{T,3},
    axis::Integer,
    order::Integer;
    coords = collect(Float32, 1:size(data, axis)),
    channels = 1:size(data, axis),
    threshold = 0.0,
    nsigma = nothing,
    sigma = nothing,
    dx = nothing,
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
    y = Vector{Float32}(undef, length(chan_vec))
    if mask === nothing
        @inbounds for u in 1:u_max, v in 1:v_max
            for (n, c) in pairs(chan_vec)
                y[n] = Float32(_channel_value(data, axis, u, v, c))
            end
            # why: only the requested order is computed (one pass for 0/1, two for 2),
            # instead of always running the full moments() triple and discarding 2/3.
            out[u, v] = Float32(_pixel_moment(order, y, x, widths, threshold, nsigma, sigma))
        end
    else
        # Masked path: NaN out voxels rejected by the mask. `_pixel_moment`
        # already discards non-finite samples, so this is the cheapest way to
        # honour the mask without forking the inner kernel.
        @inbounds for u in 1:u_max, v in 1:v_max
            for (n, c) in pairs(chan_vec)
                m = if axis == 1
                    mask[c, u, v]
                elseif axis == 2
                    mask[u, c, v]
                else
                    mask[u, v, c]
                end
                y[n] = m ? Float32(_channel_value(data, axis, u, v, c)) : Float32(NaN)
            end
            out[u, v] = Float32(_pixel_moment(order, y, x, widths, threshold, nsigma, sigma))
        end
    end
    return out
end

"""
    moment_vectors(data, x; threshold=0.0, nsigma=nothing, sigma=nothing,
                   dx=nothing, channels=nothing) -> (M0, M1, M2)

Per-row moments for a 2D `(npix, nchan)` matrix (typical HEALPix-PPV
layout). Forwards all options to [`moments`](@ref).
"""
function moment_vectors(data::AbstractMatrix, x;
                        threshold = 0.0, nsigma = nothing, sigma = nothing,
                        dx = nothing, channels = nothing)
    length(x) == size(data, 2) || throw(DimensionMismatch("x length must match size(data, 2)"))
    widths = _channel_widths(x, dx)
    M0 = Vector{Float32}(undef, size(data, 1))
    M1 = similar(M0)
    M2 = similar(M0)
    @inbounds for i in 1:size(data, 1)
        m0, m1, m2 = moments(@view(data[i, :]); x = x, threshold = threshold,
                             nsigma = nsigma, sigma = sigma, dx = widths,
                             channels = channels)
        M0[i] = Float32(m0)
        M1[i] = Float32(m1)
        M2[i] = Float32(m2)
    end
    return M0, M1, M2
end

"""
    filtered_cube_by_slice(data, axis, sigma) -> Array{Float32,3}

Apply the viewer's 2D Gaussian filter independently to every slice along
`axis`.
"""
function filtered_cube_by_slice(data::AbstractArray{T,3}, axis::Integer, sigma::Real) where {T}
    1 <= axis <= 3 || throw(ArgumentError("axis must be 1, 2, or 3"))
    σ = Float32(sigma)
    # why: avoid `Float32.(data)` / `similar(Float32.(data))` which would
    # materialise a lazy FITS cube in full.  Allocate the output directly and
    # fill it slice-by-slice via `get_slice_view`, which dispatches to the
    # cached `read_slice!` path for lazy arrays and to a no-alloc SubArray
    # view for dense arrays.
    out = Array{Float32,3}(undef, size(data))
    for idx in 1:size(data, axis)
        s_raw = get_slice_view(data, axis, idx)
        s = σ > 0 ? nan_gaussian_filter(s_raw, σ) : Float32.(s_raw)
        if axis == 1
            @views out[idx, :, :] .= s
        elseif axis == 2
            @views out[:, idx, :] .= s
        else
            @views out[:, :, idx] .= s
        end
    end
    return out
end
