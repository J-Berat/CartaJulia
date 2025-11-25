#       API stable: apply_scale, clamped_extrema, ijk_to_uv, uv_to_ijk, get_slice,
#                   make_info_tex, to_cmap, get_box_str, _pick_fig_size,
#                   latex_safe, make_main_title, make_slice_title, make_spec_title

############################
# Exports
############################
export apply_scale, clamped_extrema
export ijk_to_uv, uv_to_ijk, get_slice
export make_info_tex
export to_cmap, get_box_str, _pick_fig_size
export latex_safe, make_main_title, make_slice_title, make_spec_title

############################
# Deps
############################
using Makie
using GLFW
using LaTeXStrings          
using MathTeXEngine         

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
        return Float32.(x)
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
    clamped_extrema(vals) -> (Float32, Float32)

Ignore NaN, expand zero ranges, fallback to (0,1).
"""
function clamped_extrema(vals)::Tuple{Float32,Float32}
    f = filter(!isnan, Float32.(vals))
    if isempty(f)
        return (0f0, 1f0)
    end
    mn, mx = extrema(f)
    if mn == mx
        return (prevfloat(mn), nextfloat(mx))
    end
    return (mn, mx)
end

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
    get_slice(data::Array{T,3}, axis, idx) -> Array{Float32,2}

Returns a 2D view as Float32, orientation consistent with `ijk_to_uv`.
"""
function get_slice(data::AbstractArray{T,3}, axis::Integer, idx::Integer) where {T}
    @assert 1 ≤ axis ≤ 3 "axis must be 1,2,3"
    if axis == 1
        @views return Float32.(data[idx, :, :])  # (y,z)
    elseif axis == 2
        @views return Float32.(data[:, idx, :])  # (x,z)
    else
        @views return Float32.(data[:, :, idx])  # (x,y)
    end
end

############################
# LaTeX helpers (safe)
############################

"""
    latex_safe(s) -> String

Escape special LaTeX characters.
"""
function latex_safe(s::AbstractString)
    t = String(s)
    t = replace(t, "\\" => "\\textbackslash{}")
    t = replace(t, "_" => "\\_")
    t = replace(t, "%" => "\\%")
    t = replace(t, "&" => "\\&")
    t = replace(t, "#" => "\\#")
    t = replace(t, "\$" => "\\\$")
    t = replace(t, "{" => "\\{")
    t = replace(t, "}" => "\\}")
    t = replace(t, "^" => "\\^{}")
    t = replace(t, "~" => "\\~{}")
    return t
end

"""
    make_main_title(fname) -> LaTeXString
"""
make_main_title(fname::AbstractString) = latexstring("\\text{", latex_safe(fname), "}")

"""
    make_slice_title(fname, axis, idx) -> LaTeXString
"""
make_slice_title(fname::AbstractString, axis::Int, idx::Int) =
    latexstring("\\text{", latex_safe(fname), " — slice axis $(axis), index $(idx)}")

"""
    make_spec_title(i,j,k) -> LaTeXString
"""
make_spec_title(i::Int, j::Int, k::Int) =
    latexstring("\\text{Spectrum at pixel }(i,j,k) = ($i,$j,$k)")

"""
    make_info_tex(i,j,k,u,v,val) -> LaTeXString

Inline format; no line breaks to keep layout stable.
"""
make_info_tex(i::Int, j::Int, k::Int, u::Int, v::Int, val::Real) = latexstring(
    "\\text{pixel }(i,j,k) = ($i,$j,$k)\\,\\text{ ; slice }(\\text{row},\\text{col}) = ($u,$v)\\,\\text{ ; value }= ",
    isnan(val) ? "NaN" : string(round(Float32(val); digits=4))
)

############################
# IO / UI helpers
############################

"""
    to_cmap(name::Union{Symbol,String}) -> colormap

Resolve to a Makie colormap.
"""
to_cmap(name::Union{Symbol,String}) = Makie.to_colormap(Symbol(name))

"""
    get_box_str(textbox) -> String

Read the content of a Makie Textbox robustly.
"""
function get_box_str(tb)
    s = try
        tb.stored_string[]
    catch
        nothing
    end
    if s === nothing || (s isa AbstractString && isempty(s))
        s2 = try
            tb.displayed_string[]
        catch
            ""
        end
        return strip(String(s2))
    else
        return strip(String(s))
    end
end

############################
# Window size
############################

"""
    _pick_fig_size(sizeopt) -> (w::Int, h::Int)

Use `sizeopt` if provided, else default.
"""
@inline function _pick_fig_size(sizeopt)
    sizeopt !== nothing ? (Int(sizeopt[1]), Int(sizeopt[2])) : (1200, 800)
end