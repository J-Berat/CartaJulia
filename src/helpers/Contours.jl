# path: src/helpers/Contours.jl
#
# Automatic / manual contour-level parsing and color resolution
# (`automatic_contour_levels`, `parse_contour_levels`, `parse_contour_specs`,
# `format_contour_specs`, `contour_color_values`, plus the private
# `_try_*_color` helpers). Extracted from helpers/Helpers.jl.


"""
    automatic_contour_levels(vals; n=7) -> Vector{Float32}

Robust automatic contour levels from the finite 5th to 95th percentiles.
"""
function automatic_contour_levels(vals; n::Int = 7)
    nlev = max(2, n)
    lo, hi = percentile_clims(vals, 5, 95)
    if !(isfinite(lo) && isfinite(hi)) || lo == hi
        lo, hi = clamped_extrema(vals)
    end
    lo == hi && return Float32[]
    return collect(Float32, LinRange(lo, hi, nlev))
end

"""
    parse_contour_levels(txt; fallback=Float32[]) -> (ok, use_manual, levels, message)

Parse comma/space/semicolon-separated contour levels. Empty text means auto.
"""
function parse_contour_levels(txt::AbstractString; fallback = Float32[])
    ok, use_manual, levels, _colors, message = parse_contour_specs(txt; fallback_levels = fallback)
    return (ok, use_manual, levels, message)
end

function _try_hex_contour_color(s::AbstractString)
    h = startswith(s, "#") ? s[2:end] : s
    if !(length(h) in (6, 8)) || any(c -> !(c in '0':'9' || c in 'a':'f' || c in 'A':'F'), h)
        return (false, RGBAf(0, 0, 0, 1))
    end
    r = parse(Int, h[1:2]; base = 16) / 255
    g = parse(Int, h[3:4]; base = 16) / 255
    b = parse(Int, h[5:6]; base = 16) / 255
    a = length(h) == 8 ? parse(Int, h[7:8]; base = 16) / 255 : 1.0
    return (true, RGBAf(r, g, b, a))
end

function _try_contour_color(token::AbstractString)
    s = strip(String(token))
    isempty(s) && return (true, nothing)
    if startswith(s, "#")
        return _try_hex_contour_color(s)
    end
    for key in (Symbol(s), Symbol(lowercase(s)))
        try
            return (true, Makie.to_color(key))
        catch
        end
    end
    return (false, nothing)
end

"""
    parse_contour_specs(txt; fallback_levels=Float32[], fallback_colors=String[])

Parse manual contours. Empty text means automatic contours. Each entry can be
just a level (`1, 2, 3`) or a level with a color (`1:red, 2:#00ffaa`).
"""
function parse_contour_specs(
    txt::AbstractString;
    fallback_levels = Float32[],
    fallback_colors = String[],
)
    s = strip(String(txt))
    isempty(s) && return (
        true,
        false,
        Float32.(fallback_levels),
        String.(fallback_colors),
        "Automatic contour levels enabled.",
    )
    tokens = if occursin(r"[:=]", s)
        filter(!isempty, strip.(split(s, r"[,;\n]+")))
    else
        filter(!isempty, split(s, r"[,;\s]+"))
    end
    isempty(tokens) && return (
        true,
        false,
        Float32.(fallback_levels),
        String.(fallback_colors),
        "Automatic contour levels enabled.",
    )
    vals = Float32[]
    colors = String[]
    for tok in tokens
        parts = split(tok, r"\s*[:=]\s*"; limit = 2)
        level_txt = strip(first(parts))
        color_txt = length(parts) == 2 ? strip(last(parts)) : ""
        v = tryparse(Float32, level_txt)
        v === nothing && return (
            false,
            true,
            Float32.(fallback_levels),
            String.(fallback_colors),
            "Contour levels must be valid numbers.",
        )
        isfinite(v) || return (
            false,
            true,
            Float32.(fallback_levels),
            String.(fallback_colors),
            "Contour levels must be finite.",
        )
        ok_color, _ = _try_contour_color(color_txt)
        ok_color || return (
            false,
            true,
            Float32.(fallback_levels),
            String.(fallback_colors),
            "Contour colors must be names like red/blue or hex codes like #00ffaa.",
        )
        push!(vals, v)
        push!(colors, color_txt)
    end
    order = sortperm(vals)
    vals = vals[order]
    colors = colors[order]
    keep = trues(length(vals))
    for i in 2:length(vals)
        if vals[i] == vals[i - 1]
            keep[i - 1] = false
        end
    end
    vals = vals[keep]
    colors = colors[keep]
    length(vals) < 1 && return (
        false,
        true,
        Float32.(fallback_levels),
        String.(fallback_colors),
        "Provide at least one contour level.",
    )
    colored = any(!isempty, colors)
    msg = colored ? "Manual contour levels and colors applied." : "Manual contour levels applied."
    return (true, true, vals, colors, msg)
end

function _format_level(x::Real)
    xf = Float64(x)
    r = round(xf)
    return abs(xf - r) < 1e-8 ? string(Int(r)) : string(x)
end

"""
    format_contour_specs(levels, colors) -> String

Format contour levels back into the UI textbox syntax.
"""
function format_contour_specs(levels, colors = String[])
    out = String[]
    for (i, level) in enumerate(levels)
        color = i <= length(colors) ? strip(String(colors[i])) : ""
        push!(out, isempty(color) ? _format_level(level) : string(_format_level(level), ":", color))
    end
    return join(out, ", ")
end

"""
    contour_color_values(colors, n, default_color) -> Vector

Return Makie-ready color values for `n` contour levels.
"""
function contour_color_values(colors, n::Integer, default_color)
    out = Any[]
    for i in 1:max(0, n)
        token = i <= length(colors) ? strip(String(colors[i])) : ""
        if isempty(token)
            push!(out, default_color)
        else
            ok, c = _try_contour_color(token)
            push!(out, ok && c !== nothing ? c : default_color)
        end
    end
    return out
end
