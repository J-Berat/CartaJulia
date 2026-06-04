# path: src/helpers/Scaling.jl
#
# Display-scale transforms (`apply_scale`, `apply_scale_display`) used by the
# viewers to convert raw arrays into log / lin display buffers. Extracted from
# helpers/Helpers.jl as part of the modular split.

############################
# Scaling / Extrema
############################

# Softening length used by the :asinh stretch when no explicit value is
# supplied. The transform is asinh(x / a): for |x| ≪ a it is ~linear, for
# |x| ≫ a it is ~logarithmic, so a single knob slides between the two regimes.
# `a` is clamped to be strictly positive (≤ 0 falls back to this default) so
# the division can never blow up.
const ASINH_SOFTENING_DEFAULT = 1.0f0

# Internal: sanitize the asinh softening parameter to a strictly positive
# Float32. Non-finite or ≤ 0 inputs collapse to ASINH_SOFTENING_DEFAULT so
# callers never have to guard the value themselves.
@inline function _asinh_softening(a)::Float32
    av = Float32(a)
    (isfinite(av) && av > 0f0) ? av : ASINH_SOFTENING_DEFAULT
end

"""
    apply_scale(x, mode::Symbol; asinh_softening=ASINH_SOFTENING_DEFAULT) -> Array{Float32}

Display modes: `:lin` | `:log10` | `:ln` | `:asinh` | `:sqrt`.

In `:log10`, `:ln` and `:sqrt` modes, values ≤ 0 become `NaN` to avoid
`-Inf`/domain errors — the contrast limits then renormalise the finite range.
`:asinh` applies `asinh(x / asinh_softening)` and keeps negative values
(asinh is defined on all of ℝ), which makes it the stretch of choice for
low-SNR images with both positive and negative pixels. The softening length
`asinh_softening` (clamped > 0) slides between a near-linear core and
logarithmic wings.
"""
function apply_scale(x::AbstractArray, mode::Symbol;
                     asinh_softening = ASINH_SOFTENING_DEFAULT)
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
    elseif mode === :asinh
        inv_a = 1f0 / _asinh_softening(asinh_softening)
        y = similar(x, Float32)
        @inbounds for i in eachindex(x)
            y[i] = Float32(asinh(Float32(x[i]) * inv_a))         # why: keeps sign
        end
        return y
    elseif mode === :sqrt
        y = similar(x, Float32)
        @inbounds @fastmath for i in eachindex(x)
            xi = x[i]
            y[i] = xi > 0 ? Float32(sqrt(xi)) : Float32(NaN32)   # why: UI-safe
        end
        return y
    else
        return Float32.(x)
    end
end

# Canonical ordering of the display stretches. Used both to populate the
# scale menus (`scale_menu_options`) and to drive the keyboard cycle
# (`cycle_scale_mode`) so every view stays in sync with a single source.
const SCALE_MODES = (:lin, :log10, :ln, :asinh, :sqrt)

"""
    scale_menu_options() -> Vector{String}

String labels for the scale `Menu` widgets, in canonical [`SCALE_MODES`](@ref)
order. Centralised so a new stretch is added in exactly one place.
"""
scale_menu_options() = String[String(m) for m in SCALE_MODES]

"""
    cycle_scale_mode(mode::Symbol) -> Symbol

Return the next stretch after `mode` in [`SCALE_MODES`](@ref), wrapping around.
Drives the "cycle scale" keyboard shortcut shared by all viewers. Unknown
modes restart the cycle at `:lin`.
"""
function cycle_scale_mode(mode::Symbol)
    i = findfirst(==(mode), SCALE_MODES)
    i === nothing && return first(SCALE_MODES)
    return SCALE_MODES[mod1(i + 1, length(SCALE_MODES))]
end

"""
    scale_label_tex(mode::Symbol, unit::AbstractString) -> LaTeXString

LaTeX axis/colorbar label for a value carrying physical unit `unit` displayed
under stretch `mode`: e.g. `:log10` → ``\\log_{10}\\,\\mathrm{unit}``,
`:asinh` → ``\\mathrm{asinh}\\,\\mathrm{unit}``, `:sqrt` →
``\\sqrt{\\mathrm{unit}}``. `:lin` and unknown modes return the bare unit.
"""
function scale_label_tex(mode::Symbol, unit::AbstractString)
    u = latex_safe(unit)
    if mode === :log10
        return latexstring("\\log_{10}\\,\\text{", u, "}")
    elseif mode === :ln
        return latexstring("\\ln\\,\\text{", u, "}")
    elseif mode === :asinh
        return latexstring("\\mathrm{asinh}\\,\\text{", u, "}")
    elseif mode === :sqrt
        return latexstring("\\sqrt{\\text{", u, "}}")
    else
        return latexstring("\\text{", u, "}")
    end
end

"""
    apply_scale_display(x, mode; asinh_softening=ASINH_SOFTENING_DEFAULT) -> Matrix{Float32}

Display-oriented `apply_scale`: always returns a freshly allocated `Float32`
buffer where non-finite values are replaced by `0f0`. Equivalent to

```
A = apply_scale(x, mode); out = similar(A, Float32)
for i in eachindex(A); out[i] = isfinite(A[i]) ? Float32(A[i]) : 0f0; end
```

but fused into a single pass — saves one allocation per slice update on the
viewer's hot path. The replacement-by-zero is intentional: NaN/Inf would
break heatmap rendering downstream. Supports the same five modes as
[`apply_scale`](@ref); see there for the `asinh_softening` semantics.
"""
function apply_scale_display(x::AbstractArray, mode::Symbol;
                             asinh_softening = ASINH_SOFTENING_DEFAULT)
    out = similar(x, Float32)
    if mode === :log10
        @inbounds for i in eachindex(x)
            xi = x[i]
            v = xi > 0 ? Float32(log10(xi)) : NaN32
            out[i] = isfinite(v) ? v : 0f0
        end
    elseif mode === :ln
        @inbounds for i in eachindex(x)
            xi = x[i]
            v = xi > 0 ? Float32(log(xi)) : NaN32
            out[i] = isfinite(v) ? v : 0f0
        end
    elseif mode === :asinh
        inv_a = 1f0 / _asinh_softening(asinh_softening)
        @inbounds for i in eachindex(x)
            v = Float32(asinh(Float32(x[i]) * inv_a))
            out[i] = isfinite(v) ? v : 0f0
        end
    elseif mode === :sqrt
        @inbounds for i in eachindex(x)
            xi = x[i]
            v = xi > 0 ? Float32(sqrt(xi)) : NaN32
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
