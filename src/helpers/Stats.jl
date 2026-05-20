# path: src/helpers/Stats.jl
#
# Finite-aware extrema, percentile contrast limits and histogram primitives
# (`clamped_extrema`, `percentile_clims`, `histogram_counts`,
# `histogram_profile`, `histogram_ylabel`, `_smooth_histogram_density`,
# `normalize_histogram_mode`, `finite_float_values`). Extracted from
# helpers/Helpers.jl.


"""
    clamped_extrema(vals) -> (Float32, Float32)

Ignore NaN, expand zero ranges, fallback to (0,1).
"""
function clamped_extrema(vals)::Tuple{Float32,Float32}
    found = false
    mn = 0f0
    mx = 0f0
    @inbounds for v in vals
        fv = Float32(v)
        if isfinite(fv)
            if !found
                mn = fv
                mx = fv
                found = true
            else
                mn = min(mn, fv)
                mx = max(mx, fv)
            end
        end
    end
    if !found
        return (0f0, 1f0)
    end
    if mn == mx
        return (prevfloat(mn), nextfloat(mx))
    end
    return (mn, mx)
end

finite_float_values(vals) = begin
    out = Float32[]
    for v in vals
        fv = Float32(v)
        isfinite(fv) && push!(out, fv)
    end
    out
end

"""
    percentile_clims(vals, lo_pct, hi_pct) -> (Float32, Float32)

Return finite-value percentile limits, expanding degenerate ranges.
Percentiles are in `[0, 100]`.
"""
function percentile_clims(vals, lo_pct::Real, hi_pct::Real)::Tuple{Float32,Float32}
    xs = finite_float_values(vals)
    isempty(xs) && return (0f0, 1f0)
    lo = clamp(Float64(lo_pct), 0.0, 100.0) / 100.0
    hi = clamp(Float64(hi_pct), 0.0, 100.0) / 100.0
    lo > hi && ((lo, hi) = (hi, lo))
    qlo = Float32(quantile(xs, lo))
    qhi = Float32(quantile(xs, hi))
    if qlo == qhi
        return (prevfloat(qlo), nextfloat(qhi))
    end
    return (qlo, qhi)
end

"""
    histogram_counts(vals; bins=48, limits=nothing) -> (centers, counts)

Small finite-value histogram for UI display.
"""
function histogram_counts(vals; bins::Int = 48, limits = nothing)
    nb = max(1, bins)
    # why: avoid materializing `xs = finite_float_values(vals)` just to derive
    # extrema and a count. Single pass finds finite extrema + count.
    found_finite = false
    mn = Inf32
    mx = -Inf32
    n_finite = 0
    @inbounds for v in vals
        fv = Float32(v)
        if isfinite(fv)
            found_finite = true
            mn = mn < fv ? mn : fv
            mx = mx > fv ? mx : fv
            n_finite += 1
        end
    end
    found_finite || return (Float32[], Float32[])

    # Replicate clamped_extrema's degenerate-range expansion.
    auto_lo, auto_hi = mn == mx ? (prevfloat(mn), nextfloat(mx)) : (mn, mx)

    if limits === nothing
        lo, hi = auto_lo, auto_hi
    else
        lo, hi = Float32(first(limits)), Float32(last(limits))
        if !(isfinite(lo) && isfinite(hi)) || lo == hi
            lo, hi = auto_lo, auto_hi
        end
    end

    lo > hi && ((lo, hi) = (hi, lo))
    width = (hi - lo) / nb
    if width <= 0
        return (Float32[lo], Float32[Float32(n_finite)])
    end
    counts = zeros(Float32, nb)
    @inbounds for v in vals
        x = Float32(v)
        isfinite(x) || continue
        if lo <= x <= hi
            b = clamp(Int(floor((x - lo) / width)) + 1, 1, nb)
            counts[b] += 1f0
        end
    end
    centers = Float32[lo + (i - 0.5f0) * width for i in 1:nb]
    return (centers, counts)
end

normalize_histogram_mode(mode)::Symbol = begin
    m = Symbol(lowercase(String(mode)))
    if m in (:bar, :bars, :hist, :histogram)
        :bars
    elseif m in (:kde, :density)
        :kde
    else
        :bars
    end
end

histogram_ylabel(mode) = normalize_histogram_mode(mode) === :kde ? L"\text{density}" : L"\text{count}"

function _smooth_histogram_density(counts::Vector{Float32}, width::Float32)
    n = length(counts)
    n == 0 && return Float32[]
    total = sum(counts)
    if total <= 0f0 || width <= 0f0
        return zeros(Float32, n)
    end
    σ = Float32(max(1.0, n / 64))
    radius = max(1, ceil(Int, 3σ))
    offsets = -radius:radius
    kernel = Float32[exp(-0.5f0 * (Float32(k) / σ)^2) for k in offsets]
    kernel ./= sum(kernel)
    smoothed = zeros(Float32, n)
    @inbounds for i in 1:n
        acc = 0f0
        for (j, k) in enumerate(offsets)
            idx = i + k
            if 1 <= idx <= n
                acc += counts[idx] * kernel[j]
            end
        end
        smoothed[i] = acc / (total * width)
    end
    smoothed
end

"""
    histogram_profile(vals; bins=48, limits=nothing, mode=:bars)
        -> (x, y, width, mode)

Histogram data prepared for UI display. `mode=:bars` returns bin counts;
`mode=:kde` returns a binned Gaussian density estimate on the same x grid.
"""
function histogram_profile(vals; bins::Int = 48, limits = nothing, mode = :bars)
    nb = max(1, bins)
    centers, counts = histogram_counts(vals; bins = nb, limits = limits)
    if isempty(centers)
        return (x = Float32[], y = Float32[], width = 1f0, mode = normalize_histogram_mode(mode))
    end
    width = length(centers) > 1 ? Float32(centers[2] - centers[1]) : 1f0
    mode_sym = normalize_histogram_mode(mode)
    y = mode_sym === :kde ? _smooth_histogram_density(counts, width) : counts
    return (x = centers, y = y, width = width, mode = mode_sym)
end
