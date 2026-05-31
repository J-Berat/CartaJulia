# path: src/helpers/UIBits.jl
#
# Grab-bag of UI / IO helpers shared across viewers: LaTeX escaping
# (`latex_safe`, `make_*_title`, `make_info_tex`), `as_float32`, path spec
# parsing (`parse_path_spec`), colormap registry (`to_cmap`,
# `MANTA_COLORMAP_OPTIONS`, `ui_colormap_options`), text-box helper
# (`get_box_str`), figure-size / screen detection
# (`_DEFAULT_FIG_SIZE`, `_MIN_FIG_SIZE`, `_detect_screen_size`,
# `_pick_fig_size`, `_axis_render_height`), input validation parsers
# (`parse_textbox`, `parse_manual_clims`, `parse_histogram_bins`,
# `parse_histogram_xlimits`, `parse_histogram_ylimits`,
# `parse_spectrum_ylimits`, `parse_gif_request`)
# and viewer settings TOML I/O (`save_viewer_settings`,
# `load_viewer_settings`). Extracted from helpers/Helpers.jl.


############################
# LaTeX helpers (safe)
############################

"""
    latex_safe(s) -> String

Escape special LaTeX characters for MathTeXEngine (the backend used by
Makie). MathTeXEngine is not a full LaTeX engine: it has no text mode and
does not understand the standard text-escape commands `\\^{}` / `\\~{}` —
those would crash the parser even when wrapped inside `\\text{...}`.

We therefore leave `^` and `~` raw: MathTeXEngine interprets them as
superscript / sim operators in both `\\text{...}` and `\\mathrm{...}`,
which produces the desired rendering for unit strings like `rad/m^2`
(rendered as `rad/m²`).
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
    # NB: `^` and `~` intentionally left raw — see docstring above.
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
    "\\mathbf{pixel}\\,(i,j,k)=($i,$j,$k)\\quad\\mathbf{slice}\\,(\\text{row},\\text{col})=($u,$v)\\quad\\mathbf{intensity}= ",
    isnan(val) ? "NaN" : string(round(Float32(val); digits=4))
)

"""
    latex_tick(v::Real) -> LaTeXString

Produce a Makie-ready LaTeX tick label for a numeric tick `v`:
  - values with `|v| < 1e-10` are flattened to "0";
  - values close to an integer (`|v - round(v)| < 1e-8`) are rendered without
    a decimal point;
  - otherwise rounded to 2 decimals.

The output is wrapped in `\\mathrm{...}` so axis ticks render with an upright
font, matching the rest of MANTA's labels. Strings are escaped through
`latex_safe` so values like `-1.23` are LaTeX-safe.
"""
latex_tick(v::Real) = begin
    x = abs(Float64(v)) < 1e-10 ? 0.0 : Float64(v)
    r = round(x)
    s = if abs(x - r) < 1e-8
        string(Int(r))
    else
        string(round(x; digits = 2))
    end
    latexstring("\\mathrm{", latex_safe(s), "}")
end

"""
    latex_tick_formatter(vals) -> Vector{LaTeXString}

Vectorized version of [`latex_tick`](@ref), suitable for passing directly to
`Axis(...; xtickformat = latex_tick_formatter, ytickformat = latex_tick_formatter)`.
"""
latex_tick_formatter(vals) = [latex_tick(v) for v in vals]

############################
# Type conversion helpers
############################

"""
    as_float32(x) -> AbstractArray{Float32}

Return `x` unchanged if it is already a dense `Array{Float32}`, otherwise
allocate a fresh `Float32` copy. Centralizes the "make-it-display-ready"
conversion used by loaders and the cube viewer so that we avoid a redundant
allocation every time the data already has the right type.
"""
@inline as_float32(x::Array{Float32}) = x
@inline as_float32(x::AbstractArray) = eltype(x) === Float32 ? Array{Float32}(x) : Float32.(x)

############################
# Path spec parsing
############################

"""
    parse_path_spec(s) -> (kind, path[, address])

Inspect a path-like string and dispatch to the appropriate loader family:

- `"file.fits"`, `"file.fit"`, `"file.fits.gz"` → `(:fits, path)`
- `"file.h5"`, `"file.hdf5"`, `"file.he5"`     → `(:hdf5, path, "/")`
- `"file.h5:/group/dataset"`                    → `(:hdf5, "file.h5", "/group/dataset")`
- otherwise                                     → `(:unknown, path)`

The HDF5 `path:address` form splits on the LAST `:` only when the prefix has
an HDF5 extension and the suffix begins with `/`. This rejects Windows drive
letters (`C:/path/file.h5`) and plain FITS paths.
"""
function parse_path_spec(s::AbstractString)
    str = String(s)
    # Try HDF5 group-address form first.
    idx = findlast(==(':'), str)
    if idx !== nothing
        prefix = str[1:idx-1]
        suffix = str[idx+1:end]
        if !isempty(suffix) && startswith(suffix, "/") &&
           lowercase(splitext(prefix)[2]) ∈ (".h5", ".hdf5", ".he5")
            return (:hdf5, String(prefix), String(suffix))
        end
    end
    lower = lowercase(str)
    if endswith(lower, ".fits") || endswith(lower, ".fit") || endswith(lower, ".fits.gz")
        return (:fits, str)
    elseif endswith(lower, ".h5") || endswith(lower, ".hdf5") || endswith(lower, ".he5")
        return (:hdf5, str, "/")
    else
        return (:unknown, str)
    end
end

############################
# IO / UI helpers
############################

"""
    to_cmap(name::Union{Symbol,String}) -> colormap

Resolve to a Makie colormap.
"""
function to_cmap(name::Union{Symbol,String})
    cmap_name = Symbol(name)
    cmap_name = cmap_name in (:gray, :grey) ? :grayC : cmap_name
    return Makie.to_colormap(cmap_name)
end

############################
# Micro-icons
############################

"""
    MANTA_ICONS

Unicode micro-icons for UI labels and button prefixes.
All code-points are in the Basic Multilingual Plane and covered by
DejaVu Sans (GLMakie's bundled font).

| key       | char | meaning                              |
|-----------|------|--------------------------------------|
| save      | ⬇   | downward arrow → write to disk       |
| contrast  | ◐   | circle left-half black → half-tone   |
| selection | ⊕   | circled plus → crosshair / target    |
| histogram | ▂▄▇ | ascending blocks → bar chart profile |
| mask      | ⊟   | squared minus → stencil / filter     |
"""
const MANTA_ICONS = (
    save      = "⬇",    # U+2B07
    contrast  = "◐",    # U+25D0
    selection = "⊕",    # U+2295
    histogram = "▂▄▇",  # U+2582 U+2584 U+2587
    mask      = "⊟",    # U+229F
)

# Core sequential / perceptual colormaps.
# "okabe_ito" — 8-colour Okabe-Ito palette (colorblind-safe, ColorSchemes.jl).
# "tab10"     — Tableau-10 palette (colorblind-safe), registered in Makie's
#               gradient registry under the matplotlib-compatible alias :tab10.
#               (ColorSchemes.jl's "tableau_10_medium" is *not* in Makie's
#               gradient registry and would error via to_colormap(::Symbol).)
# Both are categorical by origin but Makie interpolates them smoothly when
# used as continuous colormaps, which works well for 2-D intensity maps.
const MANTA_COLORMAP_OPTIONS = (
    "viridis", "cividis", "magma", "inferno", "plasma", "gray",
    "okabe_ito", "tab10",
)

ui_colormap_options() = collect(MANTA_COLORMAP_OPTIONS)

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

# Defaults used when no display is available (CI / headless / Docker).
const _DEFAULT_FIG_SIZE   = (1500, 900)
# Minimum we still consider "usable" so we never collapse below this.
const _MIN_FIG_SIZE       = (1100, 720)
# Maximum fraction of the screen we want a default figure to occupy. Tuned to
# leave room for the OS chrome (title bar, dock/taskbar) on small laptops.
const _FIG_SCREEN_FRAC_W  = 0.92
const _FIG_SCREEN_FRAC_H  = 0.88

# Cache so repeated calls (panels, dual view, …) don't hit GLFW each time and
# so headless runs (`activate_gl=false`) never even try to initialize it.
const _SCREEN_SIZE_CACHE = Ref{Union{Nothing,Tuple{Int,Int}}}(nothing)
const _SCREEN_SIZE_PROBED = Ref(false)

"""
    _detect_screen_size() -> Union{Nothing,Tuple{Int,Int}}

Best-effort query of the primary monitor work area (px). Resolution order:

  1. environment override `MANTA_SCREEN_W` / `MANTA_SCREEN_H` (useful for
     Docker / VNC where GLFW often misreports the workarea),
  2. early bail-out on Linux when neither `DISPLAY` nor `WAYLAND_DISPLAY` is
     set (avoids initializing GLFW in headless containers),
  3. `GLFW.GetMonitorWorkarea` on the primary monitor, falling back to the
     full video-mode size if the workarea API isn't available.

Returns `nothing` if no size could be obtained — callers must handle this
gracefully (see `_pick_fig_size` which falls back to `_DEFAULT_FIG_SIZE`).

Result is cached: the first call probes, subsequent calls reuse the value.
"""
function _detect_screen_size()::Union{Nothing,Tuple{Int,Int}}
    _SCREEN_SIZE_PROBED[] && return _SCREEN_SIZE_CACHE[]
    _SCREEN_SIZE_PROBED[] = true

    # Environment override (useful for Docker / VNC where GLFW reports wrong values).
    env_w = tryparse(Int, get(ENV, "MANTA_SCREEN_W", ""))
    env_h = tryparse(Int, get(ENV, "MANTA_SCREEN_H", ""))
    if env_w !== nothing && env_h !== nothing && env_w > 0 && env_h > 0
        _SCREEN_SIZE_CACHE[] = (env_w, env_h)
        return _SCREEN_SIZE_CACHE[]
    end

    # On Linux without DISPLAY there is no usable screen; don't probe GLFW
    # at all so we don't risk an Init() error in headless containers.
    if Sys.islinux() && isempty(get(ENV, "DISPLAY", "")) && isempty(get(ENV, "WAYLAND_DISPLAY", ""))
        _SCREEN_SIZE_CACHE[] = nothing
        return nothing
    end

    # GLFW probe. Wrapped in try/catch because:
    #   - GLFW may already be initialized by GLMakie (Init is idempotent),
    #   - the platform may not expose a primary monitor,
    #   - the workarea API may not be available on some drivers.
    val = try
        try; GLFW.Init(); catch; end
        mon = GLFW.GetPrimaryMonitor()
        # On some bindings the null monitor has a zero handle. Treat that as
        # "no monitor".
        if mon === nothing || (hasproperty(mon, :handle) && mon.handle == C_NULL)
            nothing
        else
            # GetMonitorWorkarea returns (x, y, w, h) of the usable area
            # (i.e. screen minus dock / taskbar / menu bar). Falls back to
            # the video mode if the workarea entry-point is missing.
            wa = try
                GLFW.GetMonitorWorkarea(mon)
            catch
                nothing
            end
            if wa !== nothing
                (Int(wa[3]), Int(wa[4]))
            else
                vm = GLFW.GetVideoMode(mon)
                (Int(vm.width), Int(vm.height))
            end
        end
    catch
        nothing
    end

    _SCREEN_SIZE_CACHE[] = val
    return val
end

"""
    _pick_fig_size(sizeopt) -> (w::Int, h::Int)

Resolve the figure size for a viewer:

  * if `sizeopt` is a `(w, h)` tuple it is used verbatim,
  * otherwise the primary monitor work area is queried and the result is
    capped to `_FIG_SCREEN_FRAC_W / _FIG_SCREEN_FRAC_H` of that area,
  * if no screen can be detected (headless / CI / Docker), the conservative
    `_DEFAULT_FIG_SIZE` fallback is returned.

In all cases the output is clamped above `_MIN_FIG_SIZE` so a tiny screen
doesn't yield an unusable layout.
"""
@inline function _pick_fig_size(sizeopt)
    if sizeopt !== nothing
        return (Int(sizeopt[1]), Int(sizeopt[2]))
    end
    scr = _detect_screen_size()
    if scr === nothing
        return _DEFAULT_FIG_SIZE
    end
    sw, sh = scr
    w = round(Int, sw * _FIG_SCREEN_FRAC_W)
    h = round(Int, sh * _FIG_SCREEN_FRAC_H)
    w = max(w, _MIN_FIG_SIZE[1])
    h = max(h, _MIN_FIG_SIZE[2])
    return (w, h)
end

"""
    _axis_render_height(axis)

Return an observable height matching the axis' rendered data viewport.
Useful for keeping adjacent colorbars the same height as `DataAspect()` images.
"""
_axis_render_height(axis) = lift(axis.scene.viewport) do rect
    max(1, rect.widths[2])
end

############################
# Input validation
############################

"""
    parse_textbox(s; type, fallback[, hint_ok, hint_empty, hint_fail])
      -> (ok::Bool, value::T, hint::String)

Single-field textbox parser: strip → empty-check → `tryparse`.

| input      | `ok`  | `value`    | `hint`       |
|------------|-------|------------|--------------|
| empty      | true  | `fallback` | `hint_empty` |
| bad parse  | false | `fallback` | `hint_fail`  |
| good parse | true  | parsed     | `hint_ok`    |

Leave `hint_ok` at its default `""` when the caller builds the final
message from the returned value; the empty-vs-success branch can then be
detected by `isempty(hint)` on an `ok == true` result.
"""
function parse_textbox(
    s::AbstractString;
    type::Type{T},
    fallback::T,
    hint_ok::AbstractString    = "",
    hint_empty::AbstractString = "Value unchanged.",
    hint_fail::AbstractString  = "Invalid number.",
) where {T}
    raw = strip(String(s))
    isempty(raw) && return (true, fallback, String(hint_empty))
    val = tryparse(T, raw)
    val === nothing && return (false, fallback, String(hint_fail))
    return (true, val, String(hint_ok))
end

"""
    parse_manual_clims(min_txt, max_txt; fallback=(0f0, 1f0))
      -> (ok, use_manual, clims, message)

Validate and normalize user-provided contrast limits.
"""
parse_manual_clims(
    min_txt::AbstractString,
    max_txt::AbstractString;
    fallback::Tuple{Float32,Float32} = (0f0, 1f0),
) = _parse_axis_limits(min_txt, max_txt;
        fallback = fallback, axis_name = "contrast", check_finite = false)

"""
    parse_histogram_bins(txt; fallback=64, min_bins=4, max_bins=512)
      -> (ok, bins, message)

Validate the histogram bin count entered in the UI.
"""
function parse_histogram_bins(
    txt::AbstractString;
    fallback::Int = 64,
    min_bins::Int = 4,
    max_bins::Int = 512,
)
    ok, raw, msg = parse_textbox(String(txt);
        type       = Int,
        fallback   = fallback,
        hint_empty = "Histogram bin count unchanged.",
        hint_fail  = "Histogram bins must be an integer.",
    )
    ok || return (false, fallback, msg)
    isempty(msg) || return (true, clamp(fallback, min_bins, max_bins), msg)  # empty input
    bins = clamp(raw, min_bins, max_bins)
    bins != raw && return (true, bins, "Histogram bins were clamped to $(bins).")
    return (true, bins, "Histogram bins set to $(bins).")
end

"""
    parse_histogram_xlimits(min_txt, max_txt; fallback=(0f0, 1f0))
      -> (ok, use_manual, limits, message)

Validate user-provided histogram x-axis limits. Empty fields restore automatic
limits, which follow the current color scale limits.
"""
parse_histogram_xlimits(
    min_txt::AbstractString,
    max_txt::AbstractString;
    fallback::Tuple{Float32,Float32} = (0f0, 1f0),
) = _parse_axis_limits(min_txt, max_txt; fallback = fallback, axis_name = "histogram x-axis")

# Generic backend shared by all axis-limit parsers (xlimits / ylimits /
# spectrum ylimits / contrast clims). Pass `check_finite = false` for cases
# where ±Inf inputs should not be rejected outright (e.g. contrast limits).
function _parse_axis_limits(
    min_txt::AbstractString,
    max_txt::AbstractString;
    fallback::Tuple{Float32,Float32} = (0f0, 1f0),
    axis_name::AbstractString = "axis",
    check_finite::Bool = true,
)
    smin = strip(String(min_txt))
    smax = strip(String(max_txt))
    if isempty(smin) && isempty(smax)
        return (true, false, fallback, "Automatic $(axis_name) enabled.")
    end
    if isempty(smin) ⊻ isempty(smax)
        return (false, false, fallback, "Fill both $(axis_name) min and max, or clear both for auto mode.")
    end
    # Both non-empty: parse each field through the common combinator.
    name     = uppercasefirst(axis_name)
    fail_msg = "$(name) limits must be valid numbers."
    ok_lo, lo, _ = parse_textbox(smin; type = Float32, fallback = fallback[1], hint_fail = fail_msg)
    ok_hi, hi, _ = parse_textbox(smax; type = Float32, fallback = fallback[2], hint_fail = fail_msg)
    (ok_lo && ok_hi) || return (false, false, fallback, fail_msg)
    if check_finite && !(isfinite(lo) && isfinite(hi))
        return (false, false, fallback, "$(name) limits must be finite numbers.")
    end
    if lo > hi
        lo, hi = hi, lo
        return (true, true, (lo, hi), "$(name) limits were swapped because min > max.")
    end
    if lo == hi
        lo = prevfloat(lo)
        hi = nextfloat(hi)
        return (true, true, (lo, hi), "Expanded equal $(axis_name) limits to avoid zero width.")
    end
    return (true, true, (lo, hi), "Manual $(axis_name) applied.")
end

"""
    parse_histogram_ylimits(min_txt, max_txt; fallback=(0f0, 1f0))
      -> (ok, use_manual, limits, message)

Validate user-provided histogram y-axis limits. Empty fields restore automatic
limits.
"""
parse_histogram_ylimits(min_txt::AbstractString, max_txt::AbstractString; fallback::Tuple{Float32,Float32} = (0f0, 1f0)) =
    _parse_axis_limits(min_txt, max_txt; fallback = fallback, axis_name = "histogram y-axis")

"""
    parse_spectrum_ylimits(min_txt, max_txt; fallback=(0f0, 1f0))
      -> (ok, use_manual, limits, message)

Validate user-provided spectrum y-axis limits. Empty fields restore automatic
limits.
"""
parse_spectrum_ylimits(min_txt::AbstractString, max_txt::AbstractString; fallback::Tuple{Float32,Float32} = (0f0, 1f0)) =
    _parse_axis_limits(min_txt, max_txt; fallback = fallback, axis_name = "spectrum y-axis")

"""
    parse_gif_request(start_txt, stop_txt, step_txt, fps_txt, amax; pingpong=false)
      -> (ok, frames, fps, message)

Validate and normalize GIF export parameters.
"""
function parse_gif_request(
    start_txt::AbstractString,
    stop_txt::AbstractString,
    step_txt::AbstractString,
    fps_txt::AbstractString,
    amax::Int;
    pingpong::Bool = false
)
    amax < 1 && return (false, Int[], 12, "Cannot export GIF: axis length must be >= 1.")

    # Parse a single integer GIF field: empty → use default; bad parse → typemin(Int) sentinel.
    function _gif_field(txt::AbstractString, default::Int)
        raw = strip(txt)
        isempty(raw) && return default
        _, val, _ = parse_textbox(raw; type = Int, fallback = typemin(Int))
        return val
    end

    startv = _gif_field(start_txt, 1)
    stopv  = _gif_field(stop_txt, amax)
    stepv  = _gif_field(step_txt, 1)
    fpsv   = _gif_field(fps_txt, 12)

    if startv == typemin(Int) || stopv == typemin(Int) || stepv == typemin(Int) || fpsv == typemin(Int)
        return (false, Int[], 12, "GIF fields must be integers.")
    end
    if stepv <= 0
        return (false, Int[], 12, "GIF step must be >= 1.")
    end
    if fpsv <= 0
        return (false, Int[], 12, "GIF fps must be >= 1.")
    end

    swapped = false
    if startv > stopv
        startv, stopv = stopv, startv
        swapped = true
    end

    startv = clamp(startv, 1, amax)
    stopv  = clamp(stopv, 1, amax)
    frames = collect(startv:stepv:stopv)
    isempty(frames) && return (false, Int[], fpsv, "No GIF frame generated from the selected range.")

    if pingpong && length(frames) >= 2
        frames = vcat(frames, reverse(frames[2:end-1]))
    end

    if swapped
        return (true, frames, fpsv, "GIF start/stop were swapped because start > stop.")
    end
    return (true, frames, fpsv, "GIF settings applied.")
end

############################
# Reproducible Julia snippets
############################

"""
    manta_recipe(source; kwargs...) -> String

Build a copy-pasteable Julia call of the form
`MANTA.manta(source; kwargs...)`. Values are rendered as Julia literals for
the scalar, tuple, vector and dictionary types used by viewer state.
"""
function manta_recipe(source; kwargs...)
    pairs = collect(kwargs)
    isempty(pairs) && return "MANTA.manta($(_julia_literal(source)))"
    lines = ["MANTA.manta($(_julia_literal(source));"]
    for (i, (k, v)) in enumerate(pairs)
        comma = i == length(pairs) ? "" : ","
        push!(lines, "    $(String(k)) = $(_julia_literal(v))$(comma)")
    end
    push!(lines, ")")
    return join(lines, "\n")
end

struct JuliaExpr
    code::String
end

JuliaExpr(code::AbstractString) = JuliaExpr(String(code))

_julia_literal(x::JuliaExpr) = x.code
_julia_literal(x::Nothing) = "nothing"
_julia_literal(x::Bool) = x ? "true" : "false"
_julia_literal(x::Symbol) = ":" * String(x)
_julia_literal(x::AbstractString) = repr(String(x))
_julia_literal(x::Integer) = string(x)
function _julia_literal(x::AbstractFloat)
    isnan(x) && return "NaN"
    isinf(x) && return x > 0 ? "Inf" : "-Inf"
    return string(x)
end
_julia_literal(x::Real) = string(x)
_julia_literal(x::Tuple) = "(" * join((_julia_literal(v) for v in x), ", ") * (length(x) == 1 ? "," : "") * ")"
_julia_literal(x::NamedTuple) =
    "(" * join(("$(String(k)) = $(_julia_literal(v))" for (k, v) in pairs(x)), ", ") * ")"
_julia_literal(x::AbstractVector) =
    "Any[" * join((_julia_literal(v) for v in x), ", ") * "]"
function _julia_literal(x::AbstractDict)
    entries = ["$(_julia_literal(k)) => $(_julia_literal(v))" for (k, v) in sort!(collect(x); by = p -> string(p.first))]
    return "Dict{Any,Any}(" * join(entries, ", ") * ")"
end
_julia_literal(x) = repr(x)

function copy_text_to_clipboard(txt::AbstractString)
    try
        if isdefined(Base, :clipboard)
            Base.invokelatest(getfield(Base, :clipboard), String(txt))
            return true
        end
    catch e
        @debug "Clipboard copy failed" exception=e
    end
    return false
end

############################
# Settings I/O
############################

"""
    save_viewer_settings(path, settings)

Write a viewer settings dict to TOML.
"""
function save_viewer_settings(path::AbstractString, settings::AbstractDict{<:AbstractString,<:Any})
    open(path, "w") do io
        TOML.print(io, Dict{String,Any}(settings))
    end
    return nothing
end

"""
    load_viewer_settings(path) -> Dict{String, Any}

Read a viewer settings dict from TOML.
"""
function load_viewer_settings(path::AbstractString)::Dict{String,Any}
    return Dict{String,Any}(TOML.parsefile(path))
end
