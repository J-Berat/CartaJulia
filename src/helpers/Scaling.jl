# path: src/helpers/Scaling.jl
#
# Display-scale transforms (`apply_scale`, `apply_scale_display`) used by the
# viewers to convert raw arrays into log / lin display buffers. Extracted from
# helpers/Helpers.jl as part of the modular split.

############################
# Scaling / Extrema
############################

"""
    apply_scale(x, mode::Symbol) -> Array{Float32}

Display modes: :lin | :log10 | :ln.
In log mode, values ≤ 0 become NaN to avoid -Inf/+Inf.
"""
function apply_scale(x::AbstractArray, mode::Symbol)
    if mode === :lin
        return x isa AbstractArray{Float32} ? x : Float32.(x)
    elseif mode === :log10
        y = similar(x, Float32)
        @inbounds @fastmath for i in eachindex(x)
            xi = x[i]
            y[i] = xi > 0 ? Float32(log10(xi)) : Float32(NaN32)  # why: UI-safe
        end
        return y
    elseif mode === :ln
        y = similar(x, Float32)
        @inbounds @fastmath for i in eachindex(x)
            xi = x[i]
            y[i] = xi > 0 ? Float32(log(xi)) : Float32(NaN32)    # why: UI-safe
        end
        return y
    else
        return Float32.(x)
    end
end

"""
    apply_scale_display(x, mode) -> Matrix{Float32}

Display-oriented `apply_scale`: always returns a freshly allocated `Float32`
buffer where non-finite values are replaced by `0f0`. Equivalent to

```
A = apply_scale(x, mode); out = similar(A, Float32)
for i in eachindex(A); out[i] = isfinite(A[i]) ? Float32(A[i]) : 0f0; end
```

but fused into a single pass — saves one allocation per slice update on the
viewer's hot path. The replacement-by-zero is intentional: NaN/Inf would
break heatmap rendering downstream.
"""
function apply_scale_display(x::AbstractArray, mode::Symbol)
    out = similar(x, Float32)
    if mode === :log10
        @inbounds @fastmath for i in eachindex(x)
            xi = x[i]
            v = xi > 0 ? Float32(log10(xi)) : NaN32
            out[i] = isfinite(v) ? v : 0f0
        end
    elseif mode === :ln
        @inbounds @fastmath for i in eachindex(x)
            xi = x[i]
            v = xi > 0 ? Float32(log(xi)) : NaN32
            out[i] = isfinite(v) ? v : 0f0
        end
    else
        # :lin and unknown modes — match apply_scale's :lin / fallback paths
        @inbounds for i in eachindex(x)
            v = Float32(x[i])
            out[i] = isfinite(v) ? v : 0f0
        end
    end
    return out
end
