# =========================
# path: src/helpers/scaling.jl
# =========================
export apply_scale, clamped_extrema

"""
apply_scale(x, mode::Symbol) -> Array{Float32}
Scales data for display. Non-positive inputs become NaN for log modes.
"""
function apply_scale(x::AbstractArray, mode::Symbol)
    if mode === :lin
        return Float32.(x)
    elseif mode === :log10
        y = similar(x, Float32)
        @inbounds @fastmath for i in eachindex(x)
            xi = x[i]
            y[i] = xi > 0 ? Float32(log10(xi)) : Float32(NaN32)
        end
        return y
    elseif mode === :ln
        y = similar(x, Float32)
        @inbounds @fastmath for i in eachindex(x)
            xi = x[i]
            y[i] = xi > 0 ? Float32(log(xi)) : Float32(NaN32)
        end
        return y
    else
        return Float32.(x)
    end
end

"""
clamped_extrema(vals) -> (Float32, Float32)
Returns safe extrema; widens zero-width ranges and handles all-NaN.
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


# =========================
# path: src/helpers/mapping.jl
# =========================
export ijk_to_uv, uv_to_ijk, get_slice

"""
ijk_to_uv(i, j, k, axis) -> (u, v)
Maps 3D voxel indices to 2D slice coords for the chosen axis.
"""
@inline function ijk_to_uv(i::Int, j::Int, k::Int, axis::Int)
    axis == 1 && return (j, k)  # slice is (y,z)
    axis == 2 && return (i, k)  # slice is (x,z)
    return (i, j)               # axis==3 -> (x,y)
end

"""
uv_to_ijk(u, v, axis, idx) -> (i, j, k)
Inverse mapping: 2D slice coords + slice index -> 3D voxel.
"""
@inline function uv_to_ijk(u::Int, v::Int, axis::Int, idx::Int)
    axis == 1 && return (idx, u, v)
    axis == 2 && return (u, idx, v)
    return (u, v, idx)
end

"""
get_slice(data, axis, idx) -> Array{Float32,2}
Returns a Float32 2D view/slice for rendering.
"""
function get_slice(data::AbstractArray{T,3}, axis::Integer, idx::Integer) where {T}
    @assert 1 ≤ axis ≤ 3 "axis must be 1,2,3"
    if axis == 1
        @views return Float32.(data[idx, :, :])
    elseif axis == 2
        @views return Float32.(data[:, idx, :])
    else
        @views return Float32.(data[:, :, idx])
    end
end


# =========================
# path: src/helpers/latex.jl
# =========================
export make_info_tex, make_slice_title, make_spec_title
using LaTeXStrings

# NOTE: Inline LaTeX only. No line breaks (no `\\` newlines). Thin spaces via `\\,`.

"""
make_info_tex(i,j,k,u,v,val) -> LaTeXString
Inline status text. No line breaks, just thin spaces.
"""
make_info_tex(i::Int, j::Int, k::Int, u::Int, v::Int, val::Float32) = latexstring(
    "\\text{pixel }(i,j,k) = ($i,$j,$k)\\,\\text{ ; slice }(\\text{row},\\text{col}) = ($u,$v)\\,\\text{ ; value }= $(isnan(val) ? "NaN" : string(round(val; digits=4)))"
)

"""
make_slice_title(fname, axis, idx) -> LaTeXString
Title for saved slice figures, inline.
"""
make_slice_title(fname::AbstractString, axis::Int, idx::Int) = latexstring(
    "$(fname)\\,\\text{ — slice axis } $(axis),\\, \\text{index } $(idx)"
)

"""
make_spec_title(i,j,k) -> LaTeXString
Title for saved spectrum figure, inline.
"""
make_spec_title(i::Int, j::Int, k::Int) = latexstring(
    "\\text{Spectrum at pixel }(i,j,k) = ($i,$j,$k)"
)


# =========================
# path: src/helpers/io.jl
# =========================
export to_cmap, get_box_str
using Makie

"""
to_cmap(name) -> colormap
Resolves a Makie colormap from Symbol/String.
"""
to_cmap(name::Union{Symbol,String}) = Makie.to_colormap(Symbol(name))

"""
get_box_str(textbox) -> String
Returns trimmed content from a Makie Textbox.
"""
get_box_str(tb) = strip(String(tb.stored_string[]))


# =========================
# path: src/helpers/ui.jl
# =========================
export _pick_fig_size
using GLFW

"""
_pick_fig_size(fullscreen::Bool, sizeopt) -> (w,h)
Decide figure size from explicit size or primary monitor.
"""
@inline function _pick_fig_size(fullscreen::Bool, sizeopt)
    if sizeopt !== nothing
        return sizeopt
    elseif fullscreen
        try
            mon  = GLFW.GetPrimaryMonitor()
            mode = GLFW.GetVideoMode(mon)
            return (mode.width, mode.height)
        catch
            return (1920, 1080)  # fallback
        end
    else
        return (1800, 900)
    end
end
