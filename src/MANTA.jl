# path: src/MANTA.jl
module MANTA

# ---------------------------------------------------------------------------
# Observable / Figure lifecycle registry
#
# Problem: Julia's GC is a tracing collector.  A GLMakie `Figure` owns the
# root of an `Observable` graph — slice index, colormap, contrast limits,
# etc.  Once `display(fig)` returns, *user code* no longer holds any local
# variable pointing at `fig`, so the GC is free to collect it along with
# every `Observable` and `on(...)` listener registered on it.  GLMakie's
# rendering thread holds internal back-references to the *scene*, but not
# to the Julia `Figure` wrapper nor to the Observables layer above it,
# which means interactive callbacks silently die the next time the GC runs.
#
# Solution: pin each `Figure` in the module-level `_KEEP_ALIVE` array as
# soon as it is built.  The array is the GC root that prevents collection.
# This is *intentional* — it is not a leak.
#
# Cleanup: every viewer registers a single listener on
# `fig.scene.events.window_open`.  When GLMakie sets that Observable to
# `false` (window closed by the user), `forget!(fig)` removes the figure
# from `_KEEP_ALIVE`.  With no remaining strong reference the GC can then
# collect the `Figure` together with its entire `Observable` graph and all
# GPU-side buffers that GLMakie releases on scene destruction.
#
# Invariant: keepalive!(fig) is always called *before* display(fig), and
# forget!(fig) is always called from the window_open callback, never
# earlier.  Callers that set display_fig=false (headless tests, exports)
# still call keepalive! so the figure stays valid for the duration of any
# post-build operations, and forget! is still triggered on window close
# (which is a no-op when the figure was never displayed).
# ---------------------------------------------------------------------------
const _KEEP_ALIVE = Any[]
const _WINDOW_CLOSE_EVENTS = IdDict{Any,Any}()

"""
    keepalive!(x)

Pin `x` (typically a `Figure`) in the module-level GC-root registry so it
is not collected while its window is open.  Returns `x` unchanged.

See the `_KEEP_ALIVE` block comment above for the full rationale.
Use `forget!(x)` to release the reference (called automatically from the
`window_open` event listener installed by every viewer entry point).
"""
keepalive!(x) = (push!(_KEEP_ALIVE, x); x)

"""
    forget!(x)

Remove `x` from the `_KEEP_ALIVE` registry, allowing the GC to collect it
once no other reference exists.  Idempotent: calling it twice is safe.

This is called automatically from the `on(fig.scene.events.window_open)`
listener installed by every MANTA viewer when the window is closed.
"""
forget!(x) = (filter!(y -> y !== x, _KEEP_ALIVE); nothing)

"""
    register_window_close!(fig[, on_close])

Pin `fig` while its GLMakie window is open and release it when
`fig.scene.events.window_open` becomes `false`.  `on_close` is called exactly
once before the figure is forgotten.
"""
function register_window_close!(fig, on_close::Function = () -> nothing)
    keepalive!(fig)
    close_event = get!(_WINDOW_CLOSE_EVENTS, fig) do
        Base.Event()
    end
    did_close = Ref(false)

    function finish_close!()
        did_close[] && return nothing
        did_close[] = true
        try
            on_close()
        catch e
            @warn "MANTA: window close callback failed" exception=(e, catch_backtrace())
        end
        forget!(fig)
        pop!(_WINDOW_CLOSE_EVENTS, fig, nothing)
        notify(close_event)
        return nothing
    end

    on(fig.scene.events.window_open) do is_open
        is_open || finish_close!()
    end
    try
        fig.scene.events.window_open[] || finish_close!()
    catch
        # Some non-GL/headless figures may not expose a usable window state.
    end
    return fig
end

register_window_close!(on_close::Function, fig) = register_window_close!(fig, on_close)

"""
    wait_until_closed(fig)

Block until the GLMakie window backing `fig` is closed by the user.  Prefer
this over polling `isopen(fig.scene)`, which can stay true after the native
window has already disappeared on some GLMakie/GLFW combinations.
"""
function wait_until_closed(fig)
    haskey(_WINDOW_CLOSE_EVENTS, fig) || register_window_close!(fig)
    close_event = get!(_WINDOW_CLOSE_EVENTS, fig) do
        Base.Event()
    end
    try
        fig.scene.events.window_open[] || return nothing
    catch
        return nothing
    end
    wait(close_event)
    return nothing
end

using GLMakie
using CairoMakie
using Makie
using Observables
using ImageFiltering
using LaTeXStrings
using FITSIO
using ColorTypes
using FFTW
using Healpix

# ---- helpers ----
include("helpers/Helpers.jl")
include("helpers/UITheme.jl")
include("helpers/UIConstants.jl")

# ---- datasets ----
include("datasets/Datasets.jl")

# ---- masking ----
# Persistent mask system shared by the cube viewer and the pure helpers
# (`moment_map`, `mean_region_spectrum`). Declared early so the symbols are
# in scope when CubeView.jl is compiled.
include("masking/Mask.jl")

# ---- HEALPix viewer ----
import Statistics: quantile
include("MANTAHealpix.jl")
export manta_healpix, manta_healpix_cube, is_healpix_fits,
       read_healpix_map, mollweide_grid, mollweide_color_grid,
       valid_healpix_npix, manta_healpix_panels

# ---- loaders ----
include("loaders/LazyFITS.jl")
include("loaders/LazyHDF5.jl")
include("loaders/FITSLoader.jl")
include("loaders/HDF5Loader.jl")
include("loaders/InMemoryLoader.jl")
include("datasets/LoadDataset.jl")

# ---- views ----
include("views/HealpixMapView.jl")
# Cube-viewer support: pure helpers shared by `_view_cube` and any future
# cube-related entry points. Included before CubeView.jl so the symbols
# are available when the closure-heavy view body is compiled.
include("views/cube/CubeViewState.jl")
include("views/cube/CubeLayout.jl")
include("views/cube/MaskBundle.jl")
include("views/cube/CompareBundle.jl")
include("views/cube/KeyboardBundle.jl")
include("views/cube/ExportBundle.jl")
include("views/cube/PSWindowBundle.jl")
include("views/cube/PowerSpectrumBundle.jl")
include("views/cube/AnimationRequest.jl")
include("views/cube/SlicePipelineBundle.jl")
include("views/cube/SpectrumBundle.jl")
include("views/cube/UICallbacksBundle.jl")
include("views/cube/SettingsBundle.jl")
include("views/CubeView.jl")
include("views/VectorView.jl")

export load_dataset
export AbstractMANTADataset, AbstractCartaDataset
export VectorDataset, ImageDataset, CubeDataset
export MultiChannelDataset, HealpixMapDataset, HealpixCubeDataset
export get_slice_view, get_slice_copy, as_float32, parse_path_spec
export stable_source_id
export view_cube
# Mask system (persistent voxel masks for cube viewers)
export MaskSource, NoMaskSource, FiniteSource, ThresholdSource, RectangleSource
export GradientSource, MorphologySource
export AndSource, OrSource, NotSource
export MANTAMask, make_mask, build_mask, validate_bounds, mask_count, mask_total, mask_fraction
export mask_source_to_toml, mask_source_from_toml

spawn_safely(f::Function) = @async try f() catch e
    @error "Background task failed" exception=(e, catch_backtrace())
end

export set_dark_mode!, is_dark_mode, dark_ui_theme, current_ui_theme
export wait_until_closed
export manta, manta_panels, manta_batch

"""
    manta(filepath::String; kwargs...)

Interactive FITS viewer. **Dispatches automatically** based on file content :

- **3D cube** → slice + per-voxel spectrum viewer (default behavior).
- **HEALPix map** (header has `PIXTYPE = 'HEALPIX'`) → Mollweide
  projection with right-drag zoom (delegates to `manta_healpix`).

Common kwargs:
- `cmap`, `vmin`, `vmax`, `invert`, `figsize`, `save_dir`,
  `activate_gl`, `display_fig`.

Cube-only kwargs:
- `settings_path`.

HEALPix-only kwargs:
- `column` : column index in the BinTable (default 1).
- `nx`, `ny` : Mollweide grid resolution (default 1400×700).
- `scale` : `:lin | :log10 | :ln` (default `:lin`).

Notes:
- Manual contrast limits when `vmin` & `vmax` set (also sync spectrum Y for
  cubes).
- Window sized by explicit `figsize=(w,h)` or a fallback default.
- Export directory configurable via `save_dir`; defaults to your Desktop
  if it exists, otherwise the current working directory.
- `activate_gl=false` allows smoke tests without requiring an OpenGL
  context.
- `display_fig=false` skips window display (useful for automated tests).

Lifecycle:
The returned `Figure` is pinned in the module-level `_KEEP_ALIVE` registry
immediately after construction.  This prevents Julia's GC from collecting
the figure (and its Observable graph) while the window is open.  When the
user closes the window, GLMakie fires `fig.scene.events.window_open = false`
and MANTA's internal listener calls `forget!(fig)` to release the strong
reference.  You do not need to do anything special: just call `manta(...)`,
interact with the window, and close it when done.  Holding the returned
`Figure` in a variable is optional and does not change memory behaviour.
"""
function manta(
    filepath::String;
    cmap::Symbol = :viridis,
    vmin = nothing,
    vmax = nothing,
    invert::Bool = false,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    settings_path::Union{Nothing,AbstractString} = nothing,
    rgb::Bool = false,
    # HEALPix-specific options (ignored for 3D cubes)
    column::Int = 1,
    nx::Int = 1400,
    ny::Int = 700,
    scale::Symbol = :lin,
    asinh_softening::Real = ASINH_SOFTENING_DEFAULT,
    hist_mode::Symbol = :bars,
    hist_bins::Int = HIST_BINS_DEFAULT,
    hist_xlimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    hist_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    # HEALPix PPV cube (npix×nv) — spectral axis for the spectrum panel
    spec_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    v0::Real = 0.0,
    dv::Real = 1.0,
    vunit::AbstractString = "km/s",
    # Moment-map controls (cubes & HEALPix-PPV).
    moment_threshold::Real = 0.0,
    moment_nsigma::Union{Nothing,Real} = nothing,
    moment_channels::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    # Cube-only: preload a second FITS cube for side-by-side comparison.
    compare::Union{Nothing,AbstractString} = nothing,
    # Cube-only: declarative viewer state, as produced by "Copy code" or
    # `save_viewer_settings`.
    state = nothing,
    # FITS-only: choose the HDU (1 = primary, 0 = auto-pick first non-empty).
    hdu::Integer = 1,
    # FITS-only: memory-map the pixels (read each slice on demand instead of
    # the whole cube up-front). Quietly ignored for non-FITS inputs.
    lazy::Bool = false,
    )
    # User startup defaults (~/.manta/defaults.toml). Per the configured policy
    # the file *wins over* explicit keyword arguments, so a user can pin their
    # preferred look once and not re-pass it on every launch. Missing or
    # malformed files degrade to a no-op (see `load_manta_defaults`).
    _defaults = load_manta_defaults()
    if !isempty(_defaults)
        cmap            = _manta_default_symbol(_defaults, :cmap, cmap)
        invert          = _manta_default_bool(_defaults, :invert, invert)
        scale           = _manta_default_symbol(_defaults, :scale, scale)
        asinh_softening = _manta_default_real(_defaults, :asinh_softening, asinh_softening)
        hist_mode       = _manta_default_symbol(_defaults, :hist_mode, hist_mode)
        hist_bins       = _manta_default_int(_defaults, :hist_bins, hist_bins)
        column          = _manta_default_int(_defaults, :column, column)
        nx              = _manta_default_int(_defaults, :nx, nx)
        ny              = _manta_default_int(_defaults, :ny, ny)
        figsize         = _manta_default_figsize(_defaults, :figsize, figsize)
        save_dir        = _manta_default_string(_defaults, :save_dir, save_dir)
    end
    ds = load_dataset(filepath; column = column, v0 = v0, dv = dv, vunit = vunit,
                      hdu = hdu, lazy = lazy)

    if ds isa HealpixMapDataset
        return manta(ds;
            cmap = cmap === :viridis ? :inferno : cmap,
            vmin = vmin, vmax = vmax, invert = invert, scale = scale,
            asinh_softening = asinh_softening,
            hist_mode = hist_mode, hist_bins = hist_bins,
            hist_xlimits = hist_xlimits, hist_ylimits = hist_ylimits,
            nx = nx, ny = ny, figsize = figsize, save_dir = save_dir,
            activate_gl = activate_gl, display_fig = display_fig)
    elseif ds isa HealpixCubeDataset
        return manta(ds;
            cmap = cmap === :viridis ? :inferno : cmap,
            vmin = vmin, vmax = vmax, invert = invert, scale = scale,
            asinh_softening = asinh_softening,
            hist_mode = hist_mode, hist_bins = hist_bins,
            hist_xlimits = hist_xlimits, hist_ylimits = hist_ylimits,
            spec_ylimits = spec_ylimits,
            nx = nx, ny = ny, figsize = figsize, save_dir = save_dir,
            activate_gl = activate_gl, display_fig = display_fig,
            rgb = rgb,
            moment_threshold = moment_threshold,
            moment_nsigma = moment_nsigma,
            moment_channels = moment_channels)
    elseif ds isa CubeDataset
        return manta(ds;
            cmap = cmap, vmin = vmin, vmax = vmax, invert = invert,
            scale = scale,
            asinh_softening = asinh_softening,
            figsize = figsize, save_dir = save_dir,
            activate_gl = activate_gl, display_fig = display_fig,
            settings_path = settings_path,
            hist_mode = hist_mode, hist_bins = hist_bins,
            hist_xlimits = hist_xlimits, hist_ylimits = hist_ylimits,
            spec_ylimits = spec_ylimits,
            rgb = rgb,
            moment_threshold = moment_threshold,
            moment_nsigma = moment_nsigma,
            moment_channels = moment_channels,
            compare = compare,
            state = state)
    elseif ds isa ImageDataset
        return manta(ds;
            cmap = cmap, vmin = vmin, vmax = vmax, invert = invert,
            asinh_softening = asinh_softening,
            hist_mode = hist_mode, hist_bins = hist_bins,
            hist_xlimits = hist_xlimits, hist_ylimits = hist_ylimits,
            scale = scale, figsize = figsize, save_dir = save_dir,
            activate_gl = activate_gl, display_fig = display_fig)
    else
        return manta(ds; activate_gl = activate_gl, display_fig = display_fig)
    end
end

function manta(
    img::AbstractMatrix{<:Real};
    title::AbstractString = "2D image",
    cmap::Symbol = :viridis,
    vmin = nothing,
    vmax = nothing,
    invert::Bool = false,
    scale::Symbol = :lin,
    asinh_softening::Real = ASINH_SOFTENING_DEFAULT,
    hist_mode::Symbol = :bars,
    hist_bins::Int = HIST_BINS_DEFAULT,
    hist_xlimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    hist_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    unit_label::AbstractString = "value",
    wcs::AbstractVector{SimpleWCSAxis} = SimpleWCSAxis[],
    wcs_transform::Union{Nothing,WCSTransform} = nothing,
)
    data2d = Float32.(img)
    unit_label_tex = latexstring("\\text{", latex_safe(unit_label), "}")

    cmap_name = Observable(cmap)
    invert_cmap = Observable(invert)
    cm_obs = lift(cmap_name, invert_cmap) do name, inv
        base = to_cmap(name); inv ? reverse(base) : base
    end
    scale_mode = Observable(scale)
    asinh_a = Observable(Float32(asinh_softening))
    img_disp = lift(scale_mode, asinh_a) do m, a
        A = apply_scale(data2d, m; asinh_softening = a)
        out = similar(A, Float32)
        @inbounds for i in eachindex(A)
            x = A[i]
            out[i] = isfinite(x) ? Float32(x) : 0f0
        end
        out
    end
    clims_auto = lift(img_disp) do im
        clamped_extrema(im)
    end
    clims_manual = Observable((0f0, 1f0))
    use_manual = Observable(false)
    if vmin !== nothing && vmax !== nothing
        a, b = Float32(vmin), Float32(vmax)
        a == b && (a = prevfloat(a); b = nextfloat(b))
        clims_manual[] = (a, b)
        use_manual[] = true
    end
    clims_obs = lift(use_manual, clims_auto, clims_manual) do um, ca, cm
        um ? cm : ca
    end
    clims_safe = lift(clims_obs) do (lo, hi)
        (isfinite(lo) && isfinite(hi) && lo != hi) ? (lo, hi) : (0f0, 1f0)
    end
    hist_mode_obs = Observable(normalize_histogram_mode(hist_mode))
    hist_bins_obs = Observable(clamp(hist_bins, HIST_BINS_MIN, HIST_BINS_MAX))
    hist_xlimits_manual = Observable(hist_xlimits !== nothing)
    hist_xlimits_manual_value = Observable(hist_xlimits === nothing ?
        (0f0, 1f0) :
        parse_histogram_xlimits(string(first(hist_xlimits)), string(last(hist_xlimits)))[3])
    hist_ylimits_manual = Observable(hist_ylimits !== nothing)
    hist_ylimits_manual_value = Observable(hist_ylimits === nothing ?
        (0f0, 1f0) :
        parse_histogram_ylimits(string(first(hist_ylimits)), string(last(hist_ylimits)))[3])
    hist_limits_obs = lift(hist_xlimits_manual, hist_xlimits_manual_value, clims_safe) do manual, xlim, clim
        manual ? xlim : clim
    end
    hist_pair_obs = lift(img_disp, hist_limits_obs, hist_bins_obs, hist_mode_obs) do im, lim, bins, mode
        histogram_profile(im; bins = bins, limits = lim, mode = mode)
    end
    hist_x_obs = lift(p -> p.x, hist_pair_obs)
    hist_y_obs = lift(p -> p.y, hist_pair_obs)
    hist_width_obs = lift(p -> p.width, hist_pair_obs)
    hist_bars_visible = lift(m -> m === :bars, hist_mode_obs)
    hist_kde_visible = lift(m -> m === :kde, hist_mode_obs)
    hist_ylabel_obs = lift(histogram_ylabel, hist_mode_obs)

    ui_theme = current_ui_theme()
    fig_bg_panels = ui_theme.background
    pick_backend!(activate_gl)
    fig = Figure(size = _pick_fig_size(figsize), backgroundcolor = fig_bg_panels)
    grid = fig[1, 1] = GridLayout()
    colgap!(grid, 16); rowgap!(grid, 14)
    # halign/tellwidth: prevents img_grid from expanding beyond its natural
    # content width (image + colorbar) when the controls below are wider.
    img_grid = grid[1, 1] = GridLayout(; halign = :center, tellwidth = false)
    colgap!(img_grid, -8)
    rowgap!(img_grid, 14)   # gap between image and histogram (same as rowgap of grid)
    ax = Axis(
        img_grid[1, 1];
        title = make_main_title(title),
        xlabel = L"\text{pixel x}",
        ylabel = L"\text{pixel y}",
        aspect = DataAspect(),
    )
    hm = heatmap!(ax, img_disp; colormap = cm_obs, colorrange = clims_safe)
    Colorbar(
        img_grid[1, 2],
        hm;
        label = unit_label_tex,
        width = 20,
        height = _axis_render_height(ax),
        tellheight = false,
        valign = :center,
    )

    # Celestial WCS graticule overlaid on the image. Computed by contouring the
    # per-pixel sky grids (same deprojection as the cursor readout); a no-op
    # with empty `plots` when the input carries no identified sky axis pair, so
    # the headless precompile path (wcs = SimpleWCSAxis[]) is unaffected.
    grat_src = wcs_transform === nothing ? wcs : wcs_transform
    show_graticule = Observable(false)
    graticule_handle = (wcs_transform !== nothing || !isempty(wcs)) ?
        draw_wcs_graticule!(ax, grat_src, size(data2d, 1), size(data2d, 2); visible = false) :
        nothing
    has_grat = graticule_handle !== nothing && !isempty(graticule_handle.plots)
    on(show_graticule) do v
        graticule_handle === nothing || set_wcs_graticule_visible!(graticule_handle, v)
    end

    ui_accent = ui_theme.accent
    ui_text_muted = ui_theme.text_muted

    # Histogram placed in img_grid[2, 1:2]: it automatically inherits the
    # same width as the image+colorbar pair (columns 1 and 2).
    ax_hist = Axis(
        img_grid[2, 1:2];
        title = L"\text{Image histogram}",
        xlabel = unit_label_tex,
        ylabel = hist_ylabel_obs,
        height = 130,
    )
    barplot!(ax_hist, hist_x_obs, hist_y_obs; width = hist_width_obs, color = (ui_accent, HIST_BAR_ALPHA), strokecolor = ui_accent, strokewidth = HIST_BAR_STROKE_LW, visible = hist_bars_visible)
    lines!(ax_hist, hist_x_obs, hist_y_obs; color = ui_accent, linewidth = HIST_KDE_LW, visible = hist_kde_visible)
    vlines!(ax_hist, lift(lim -> [first(lim), last(lim)], clims_safe); color = (ui_text_muted, HIST_LIMITS_ALPHA), linewidth = HIST_LIMITS_LW, linestyle = :dash)

    ctrl = grid[2, 1] = GridLayout(; alignmode = Outside())
    Label(ctrl[1, 1], text = "Image", halign = :left, tellwidth = false, fontsize = 14, color = ui_text_muted)
    scale_menu = Menu(ctrl[1, 2]; options = scale_menu_options(), prompt = String(scale), width = 96)
    Label(ctrl[1, 3], text = "Colormap", halign = :left, tellwidth = false, fontsize = 14, color = ui_text_muted)
    cmap_menu = colormap_selector!(ctrl[1, 4]; cmap = cmap, width = 112, theme = ui_theme)
    invert_chk = Checkbox(ctrl[1, 5])
    Label(ctrl[1, 6], text = "Invert", halign = :left, tellwidth = false, fontsize = 14, color = ui_theme.text)
    help_btn = Button(ctrl[1, 7]; label = MANTA_ICONS.help, width = 46, height = 32)
    # Graticule toggle: only surfaced when the image actually carries a sky WCS
    # (otherwise `has_grat` is false and the control would be a dead no-op).
    grat_chk = nothing
    if has_grat
        grat_chk = Checkbox(ctrl[1, 8]; checked = false)
        Label(ctrl[1, 9], text = "Graticule", halign = :left, tellwidth = false, fontsize = 14, color = ui_theme.text)
        on(grat_chk.checked) do c
            show_graticule[] = c
        end
    end
    Label(ctrl[2, 1], text = "Contrast", halign = :left, tellwidth = false, fontsize = 14, color = ui_text_muted)
    clim_min_box = Textbox(ctrl[2, 2]; placeholder = "min", width = 110, height = 32)
    clim_max_box = Textbox(ctrl[2, 3]; placeholder = "max", width = 110, height = 32)
    apply_btn = Button(ctrl[2, 4]; label = "Apply", width = 82, height = 32)
    auto_btn = Button(ctrl[2, 5]; label = "Auto", width = 78, height = 32)
    p1_btn = Button(ctrl[2, 6]; label = "p1-p99", width = 92, height = 32)
    p5_btn = Button(ctrl[2, 7]; label = "p5-p95", width = 92, height = 32)
    save_btn = Button(ctrl[2, 8]; label = "$(MANTA_ICONS.export_icon) PNG", width = 90, height = 32)
    Label(ctrl[3, 1], text = "Histogram", halign = :left, tellwidth = false, fontsize = 14, color = ui_text_muted)
    hist_mode_menu = Menu(ctrl[3, 2]; options = ["bars", "kde"], prompt = String(hist_mode_obs[]), width = 96)
    hist_bins_box = Textbox(ctrl[3, 3]; placeholder = "bins", width = 82, height = 32)
    hist_xmin_box = Textbox(ctrl[3, 4]; placeholder = "x min", width = 100, height = 32)
    hist_xmax_box = Textbox(ctrl[3, 5]; placeholder = "x max", width = 100, height = 32)
    hist_apply_btn = Button(ctrl[3, 6]; label = "Apply x", width = 82, height = 32)
    hist_auto_btn = Button(ctrl[3, 7]; label = "Auto x", width = 82, height = 32)
    hist_ymin_box = Textbox(ctrl[4, 4]; placeholder = "y min", width = 100, height = 32)
    hist_ymax_box = Textbox(ctrl[4, 5]; placeholder = "y max", width = 100, height = 32)
    hist_y_apply_btn = Button(ctrl[4, 6]; label = "Apply y", width = 82, height = 32)
    hist_y_auto_btn = Button(ctrl[4, 7]; label = "Auto y", width = 82, height = 32)
    ui_status = Observable(" ")
    grid[3, 1] = Label(grid[3, 1]; text = ui_status, halign = :left, tellwidth = false)

    foreach(w -> manta_style_button_primary!(w, ui_theme), (apply_btn, save_btn, hist_apply_btn, hist_y_apply_btn))
    foreach(w -> manta_style_button!(w, ui_theme),         (p1_btn, p5_btn))
    foreach(w -> manta_style_button_ghost!(w, ui_theme),   (auto_btn, hist_auto_btn, hist_y_auto_btn, help_btn))
    foreach(w -> manta_style_menu!(w, ui_theme), (scale_menu, cmap_menu, hist_mode_menu))
    foreach(w -> manta_style_textbox!(w, ui_theme), (clim_min_box, clim_max_box, hist_bins_box, hist_xmin_box, hist_xmax_box, hist_ymin_box, hist_ymax_box))
    manta_style_checkbox!(invert_chk, ui_theme)
    invert_chk.checked[] = invert
    cmap_menu.selection[] = String(cmap_name[])
    set_box_text_local!(tb, s::AbstractString) = begin
        str = String(s)
        tb.displayed_string[] = str
        tb.stored_string[] = str
        nothing
    end
    set_box_text_local!(hist_bins_box, string(hist_bins_obs[]))
    if hist_xlimits_manual[]
        lo, hi = hist_xlimits_manual_value[]
        set_box_text_local!(hist_xmin_box, string(lo))
        set_box_text_local!(hist_xmax_box, string(hi))
    end
    if hist_ylimits_manual[]
        lo, hi = hist_ylimits_manual_value[]
        set_box_text_local!(hist_ymin_box, string(lo))
        set_box_text_local!(hist_ymax_box, string(hi))
    end

    set_status!(msg::AbstractString) = (ui_status[] = String(msg); nothing)

    # ── Live cursor readout (DS9 / CARTA style) ──────────────────────────
    # Pixel value + WCS world coordinates under the pointer, pushed into the
    # shared status line on every move. `heatmap!(ax, matrix)` samples the
    # image at integer coordinates 1:N, so the rounded data-space position is
    # the FITS pixel index along each axis (dim 1 = rows, dim 2 = columns).
    # Exact celestial coordinates (CD matrix + projection) when the loader
    # supplied a full WCSTransform; otherwise cursor_world_strings degrades to
    # the per-axis linear world_coord.
    nx2d, ny2d = size(data2d)
    function _readout_at(px::Real, py::Real)
        ix = clamp(round(Int, px), 1, nx2d)
        iy = clamp(round(Int, py), 1, ny2d)
        val = @inbounds data2d[ix, iy]
        world = cursor_world_strings(wcs, wcs_transform, [1, 2], [ix, iy])
        return format_cursor_readout(["x" => ix, "y" => iy], val, unit_label, world)
    end
    on(events(ax).mouseposition) do _
        Makie.is_mouseinside(ax.scene) || return
        p = mouseposition(ax)
        (isfinite(p[1]) && isfinite(p[2])) || return
        set_status!(_readout_at(p[1], p[2]))
    end

    set_box_text!(tb, s::AbstractString) = begin
        str = String(s)
        tb.displayed_string[] = str
        tb.stored_string[] = str
        nothing
    end
    function apply_percentile_clims!(lo::Real, hi::Real)
        parsed = percentile_clims(img_disp[], lo, hi)
        clims_manual[] = parsed
        use_manual[] = true
        set_box_text!(clim_min_box, string(first(parsed)))
        set_box_text!(clim_max_box, string(last(parsed)))
        set_status!("Contrast set to p$(lo)-p$(hi).")
    end

    function refresh_hist_axes!()
        xlo, xhi = hist_limits_obs[]
        if hist_ylimits_manual[]
            ylo, yhi = hist_ylimits_manual_value[]
            limits!(ax_hist, Float32(xlo), Float32(xhi), Float32(ylo), Float32(yhi))
        else
            autolimits!(ax_hist)
            xlims!(ax_hist, Float32(xlo), Float32(xhi))
        end
    end

    on(scale_menu.selection) do sel
        sel === nothing && return
        scale_mode[] = Symbol(sel)
    end
    on(cmap_menu.selection) do sel
        sel === nothing && return
        cmap_name[] = Symbol(sel)
        set_status!("Colormap set to $(String(sel)).")
    end
    on(invert_chk.checked) do v
        invert_cmap[] = v
    end
    on(apply_btn.clicks) do _
        ok, manual, parsed, msg = parse_manual_clims(
            get_box_str(clim_min_box),
            get_box_str(clim_max_box);
            fallback = clims_manual[],
        )
        set_status!(msg)
        ok || return
        if manual
            clims_manual[] = parsed
            use_manual[] = true
            set_box_text!(clim_min_box, string(first(parsed)))
            set_box_text!(clim_max_box, string(last(parsed)))
        else
            use_manual[] = false
        end
    end
    on(auto_btn.clicks) do _
        use_manual[] = false
        set_box_text!(clim_min_box, "")
        set_box_text!(clim_max_box, "")
        set_status!("Automatic contrast enabled.")
    end
    on(p1_btn.clicks) do _; apply_percentile_clims!(1, 99); end
    on(p5_btn.clicks) do _; apply_percentile_clims!(5, 95); end
    on(hist_mode_menu.selection) do sel
        sel === nothing && return
        hist_mode_obs[] = normalize_histogram_mode(sel)
        set_status!("Histogram mode set to $(String(hist_mode_obs[])).")
    end
    on(hist_apply_btn.clicks) do _
        ok_bins, bins, bins_msg = parse_histogram_bins(get_box_str(hist_bins_box); fallback = hist_bins_obs[])
        ok_x, manual_x, xlim, x_msg = parse_histogram_xlimits(
            get_box_str(hist_xmin_box),
            get_box_str(hist_xmax_box);
            fallback = hist_xlimits_manual_value[],
        )
        if !ok_bins
            set_status!(bins_msg)
            return
        end
        if !ok_x
            set_status!(x_msg)
            return
        end
        hist_bins_obs[] = bins
        hist_xlimits_manual_value[] = xlim
        hist_xlimits_manual[] = manual_x
        set_box_text!(hist_bins_box, string(bins))
        if manual_x
            set_box_text!(hist_xmin_box, string(first(xlim)))
            set_box_text!(hist_xmax_box, string(last(xlim)))
        else
            set_box_text!(hist_xmin_box, "")
            set_box_text!(hist_xmax_box, "")
        end
        refresh_hist_axes!()
        set_status!("$(bins_msg) $(x_msg)")
    end
    on(hist_auto_btn.clicks) do _
        hist_xlimits_manual[] = false
        set_box_text!(hist_xmin_box, "")
        set_box_text!(hist_xmax_box, "")
        refresh_hist_axes!()
        set_status!("Automatic histogram x-axis enabled.")
    end
    on(hist_y_auto_btn.clicks) do _
        hist_ylimits_manual[] = false
        set_box_text!(hist_ymin_box, "")
        set_box_text!(hist_ymax_box, "")
        refresh_hist_axes!()
        set_status!("Automatic histogram y-axis enabled.")
    end
    on(hist_y_apply_btn.clicks) do _
        ok_y, manual_y, ylim, y_msg = parse_histogram_ylimits(
            get_box_str(hist_ymin_box),
            get_box_str(hist_ymax_box);
            fallback = hist_ylimits_manual_value[],
        )
        set_status!(y_msg)
        ok_y || return
        hist_ylimits_manual_value[] = ylim
        hist_ylimits_manual[] = manual_y
        if manual_y
            set_box_text!(hist_ymin_box, string(first(ylim)))
            set_box_text!(hist_ymax_box, string(last(ylim)))
        else
            set_box_text!(hist_ymin_box, "")
            set_box_text!(hist_ymax_box, "")
        end
        refresh_hist_axes!()
    end
    on(hist_limits_obs) do _
        refresh_hist_axes!()
    end
    on(hist_y_obs) do _
        refresh_hist_axes!()
    end

    save_root = if save_dir === nothing
        d = joinpath(homedir(), "Desktop")
        isdir(d) ? d : pwd()
    else
        path = String(save_dir)
        isdir(path) || mkpath(path)
        path
    end
    on(save_btn.clicks) do _
        out = joinpath(save_root, "$(title)_image2d.png")
        try
            CairoMakie.save(String(out), fig; backend = CairoMakie)
            set_status!("Saved image to $(out).")
        catch e
            msg = "Failed to save image: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
        end
    end

    # ---------- Keyboard shortcuts (2D image view) ----------
    _trigger_button_2d!(btn) = (btn.clicks[] = btn.clicks[] + 1)
    function _cycle_log_scale_2d!()
        next = cycle_scale_mode(scale_mode[])
        scale_menu.selection[] = String(next)
        set_status!("Image scale: $(String(next)).")
    end
    shortcuts_2d = ShortcutBinding[
        ShortcutBinding(Keyboard.i,  () -> (invert_cmap[] = !invert_cmap[]);
                        description = "invert cmap"),
        ShortcutBinding(Keyboard.a,  () -> _trigger_button_2d!(auto_btn);
                        description = "auto contrast"),
        ShortcutBinding(Keyboard._1, () -> apply_percentile_clims!(1, 99);
                        description = "p1-p99"),
        ShortcutBinding(Keyboard._5, () -> apply_percentile_clims!(5, 95);
                        description = "p5-p95"),
        ShortcutBinding(Keyboard.r,  () -> autolimits!(ax);
                        description = "reset zoom"),
        ShortcutBinding(Keyboard.s,  () -> _trigger_button_2d!(save_btn);
                        description = "save image"),
        ShortcutBinding(Keyboard.l,  () -> _cycle_log_scale_2d!();
                        description = "cycle scale"),
    ]
    # `+`/`-` (and numpad) centered zoom on the image axis.
    append!(shortcuts_2d, zoom_shortcut_bindings(ax; on_change = set_status!))
    if has_grat
        # 'g' flips the checkbox; its handler propagates to `show_graticule`
        # and thus to the overlay, keeping the UI in sync with the shortcut.
        push!(shortcuts_2d,
              ShortcutBinding(Keyboard.g, () -> (grat_chk.checked[] = !grat_chk.checked[]);
                              description = "toggle WCS graticule"))
    end
    # Help window — both the Shift+/ binding and the Help button open a
    # dedicated Makie figure listing every documented shortcut. The status
    # bar still gets a one-line recap so headless / scripted users keep a
    # textual trace.
    function _open_help_2d!()
        try
            open_shortcut_help_window(shortcuts_2d;
                title = "MANTA — 2D image shortcuts", theme = ui_theme)
        catch e
            @warn "Could not open shortcut help window" exception = (e, catch_backtrace())
        end
        set_status!(shortcut_help_message(shortcuts_2d))
    end
    push!(shortcuts_2d,
          ShortcutBinding(Keyboard.slash,
                          _open_help_2d!;
                          description = "this help",
                          modifier = :shift))
    on(help_btn.clicks) do _
        _open_help_2d!()
    end
    register_shortcuts!(fig, shortcuts_2d;
        textboxes = (clim_min_box, clim_max_box,
                     hist_bins_box, hist_xmin_box, hist_xmax_box,
                     hist_ymin_box, hist_ymax_box),
    )

    register_window_close!(fig)  # anchor in GC root — see _KEEP_ALIVE block comment
    enable_file_drop!(fig; activate_gl = activate_gl, display_fig = display_fig)
    refresh_hist_axes!()
    display_fig && display(fig)
    return fig
end

function manta(
    img::AbstractArray;
    title::AbstractString = "RGB image",
    xlabel = nothing,
    ylabel = nothing,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
)
    rgb_img = as_rgb_image(img)
    pick_backend!(activate_gl)
    fig = Figure(size = _pick_fig_size(figsize))
    ax = Axis(
        fig[1, 1];
        title = make_main_title(title),
        xlabel = xlabel === nothing ? L"\text{pixel x}" : xlabel,
        ylabel = ylabel === nothing ? L"\text{pixel y}" : ylabel,
        aspect = DataAspect(),
    )
    rows, cols = size(rgb_img)
    image!(ax, (1, cols), (1, rows), permutedims(rgb_img))
    register_window_close!(fig)  # anchor in GC root — see _KEEP_ALIVE block comment
    enable_file_drop!(fig; activate_gl = activate_gl, display_fig = display_fig)
    display_fig && display(fig)
    return fig
end

"""
    manta_panels(panels...; titles, cmaps, clims, figsize, activate_gl, display_fig)

Side-by-side static viewer for N 2-D arrays or RGB images.

Each element of `panels` becomes one column in a single `Figure`.  Scalar
panels are shown as heatmaps with an individual colorbar; RGB/RGBA panels
are rendered as true-color images.

Kwargs:
- `titles`  : vector of column titles (default: "panel 1", "panel 2", …).
- `cmaps`   : vector of colormaps, one per scalar panel (default `:viridis`).
- `clims`   : vector of `(lo, hi)` tuples; `nothing` auto-scales each panel.
- `figsize` : explicit `(width, height)` in pixels; falls back to
              `_pick_fig_size(nothing)` (1800×1000).
- `activate_gl`, `display_fig` : same semantics as `manta`.

Returns the `Figure`.

Lifecycle:
The figure is pinned in `_KEEP_ALIVE` on construction and released
automatically via the `window_open` event when the window is closed.
See `keepalive!` / `forget!` for details.
"""
function manta_panels(
    panels::Vararg{Any,N};
    titles = nothing,
    cmaps = nothing,
    clims = nothing,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
) where {N}
    N >= 1 || throw(ArgumentError("Provide at least one panel."))
    pick_backend!(activate_gl)
    fig = Figure(size = _pick_fig_size(figsize))
    title_at(i) = titles === nothing ? "panel $(i)" : String(titles[i])
    cmap_at(i) = cmaps === nothing ? :viridis : cmaps[i]
    clim_at(i, vals) = clims === nothing ? clamped_extrema(vals) : clims[i]
    for (i, panel) in enumerate(panels)
        panel_grid = fig[1, i] = GridLayout()
        colgap!(panel_grid, -8)
        ax = Axis(
            panel_grid[1, 1];
            title = make_main_title(title_at(i)),
            aspect = DataAspect(),
        )
        if is_rgb_like(panel)
            img = as_rgb_image(panel)
            rows, cols = size(img)
            image!(ax, (1, cols), (1, rows), permutedims(img))
        else
            vals = Float32.(panel)
            hm = heatmap!(ax, vals; colormap = cmap_at(i), colorrange = clim_at(i, vals))
            Colorbar(
                panel_grid[1, 2],
                hm;
                width = 16,
                height = _axis_render_height(ax),
                tellheight = false,
                valign = :center,
            )
        end
    end
    register_window_close!(fig)  # anchor in GC root — see _KEEP_ALIVE block comment
    enable_file_drop!(fig; activate_gl = activate_gl, display_fig = display_fig)
    display_fig && display(fig)
    return fig
end

# ----------------------------------------------------------------------------
# Dataset-aware dispatch.
#
# These methods are the view half of "load_dataset(x) -> view it". Paths and
# in-memory 3D arrays now enter through `load_dataset`; 2D scalar images and
# RGB arrays keep their direct lightweight viewers above.
# ----------------------------------------------------------------------------

"""
    manta(ds::AbstractMANTADataset; kwargs...)

Dispatch a pre-built MANTA dataset to the matching viewer.
"""
function manta(ds::ImageDataset; kwargs...)
    xform = get(ds.metadata, :wcs_transform, nothing)
    return manta(ds.data;
        title = ds.source_id,
        unit_label = ds.unit_label,
        wcs = ds.wcs,
        wcs_transform = xform isa WCSTransform ? xform : nothing,
        kwargs...)
end

function manta(ds::HealpixMapDataset; kwargs...)
    return _view_healpix_map(ds; kwargs...)
end

function manta(
    ds::HealpixCubeDataset;
    rgb::Bool = false,
    cmap::Symbol = :inferno,
    vmin = nothing,
    vmax = nothing,
    invert::Bool = false,
    scale::Symbol = :lin,
    asinh_softening::Real = ASINH_SOFTENING_DEFAULT,
    nx::Int = 1200,
    ny::Int = 600,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    hist_mode::Symbol = :bars,
    hist_bins::Int = HIST_BINS_DEFAULT,
    hist_xlimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    hist_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    spec_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    moment_threshold::Real = 0.0,
    moment_nsigma::Union{Nothing,Real} = nothing,
    moment_channels::Union{Nothing,AbstractVector{<:Integer}} = nothing,
)
    if rgb
        return manta_healpix(as_rgb_pixels(ds.data);
            title = ds.source_id,
            nx = nx, ny = ny, figsize = figsize,
            activate_gl = activate_gl, display_fig = display_fig)
    end
    return _view_healpix_cube(ds;
        cmap = cmap, vmin = vmin, vmax = vmax, invert = invert,
        scale = scale, asinh_softening = asinh_softening,
        nx = nx, ny = ny, figsize = figsize,
        save_dir = save_dir, activate_gl = activate_gl,
        display_fig = display_fig,
        hist_mode = hist_mode, hist_bins = hist_bins,
        hist_xlimits = hist_xlimits, hist_ylimits = hist_ylimits,
        spec_ylimits = spec_ylimits,
        moment_threshold = moment_threshold,
        moment_nsigma = moment_nsigma,
        moment_channels = moment_channels)
end

function manta(
    ds::CubeDataset;
    rgb::Bool = false,
    cmap::Symbol = :viridis,
    vmin = nothing,
    vmax = nothing,
    invert::Bool = false,
    scale::Symbol = :lin,
    asinh_softening::Real = ASINH_SOFTENING_DEFAULT,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    settings_path::Union{Nothing,AbstractString} = nothing,
    hist_mode::Symbol = :bars,
    hist_bins::Int = HIST_BINS_DEFAULT,
    hist_xlimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    hist_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    spec_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    moment_threshold::Real = 0.0,
    moment_nsigma::Union{Nothing,Real} = nothing,
    moment_channels::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    compare::Union{Nothing,AbstractString} = nothing,
    state = nothing,
)
    if rgb
        return manta(as_rgb_image(ds.data);
            title = ds.source_id,
            figsize = figsize,
            activate_gl = activate_gl,
            display_fig = display_fig)
    end
    return _view_cube(ds;
        cmap = cmap, vmin = vmin, vmax = vmax, invert = invert,
        scale = scale,
        asinh_softening = asinh_softening,
        figsize = figsize, save_dir = save_dir,
        activate_gl = activate_gl, display_fig = display_fig,
        settings_path = settings_path,
        hist_mode = hist_mode, hist_bins = hist_bins,
        hist_xlimits = hist_xlimits, hist_ylimits = hist_ylimits,
        spec_ylimits = spec_ylimits,
        moment_threshold = moment_threshold,
        moment_nsigma = moment_nsigma,
        moment_channels = moment_channels,
        compare = compare,
        state = state)
end

function manta(
    ds::VectorDataset;
    title::Union{Nothing,AbstractString} = nothing,
    xscale::Symbol = :lin,
    yscale::Symbol = :lin,
    xlimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
)
    return _view_vector(ds;
        title = title,
        xscale = xscale, yscale = yscale,
        xlimits = xlimits, ylimits = ylimits,
        figsize = figsize, save_dir = save_dir,
        activate_gl = activate_gl, display_fig = display_fig)
end

function manta(ds::MultiChannelDataset; kwargs...)
    if ds.kind === :image
        panels = [ds.channels[k].data for k in sort!(collect(keys(ds.channels)))]
        titles = [String(k) for k in sort!(collect(keys(ds.channels)))]
        return manta_panels(panels...; titles = titles, kwargs...)
    end
    throw(ErrorException(
        "MANTA: viewing MultiChannelDataset of kind $(ds.kind) is not implemented yet."))
end

# Bridges for inputs that are NOT already paths / numeric arrays. These are
# safe to add: existing `manta(::AbstractMatrix{<:Real})`, `manta(::AbstractArray)`
# (RGB) and `manta(filepath::String)` remain the most-specific matches for
# their respective inputs.
manta(x::NamedTuple; kwargs...) = manta(load_dataset(x); kwargs...)
manta(x::AbstractDict; kwargs...) = manta(load_dataset(x); kwargs...)
manta(x::Healpix.HealpixMap; kwargs...) = manta(load_dataset(x); kwargs...)

# 3D numeric arrays: route through load_dataset → CubeDataset → _view_cube,
# so an in-memory cube gets the full interactive viewer (slice navigation,
# spectra, moments, comparison, power spectrum, exports) without ever
# touching disk.
manta(x::AbstractArray{<:Real,3}; kwargs...) = manta(load_dataset(x); kwargs...)

# 1D numeric arrays: route through load_dataset → VectorDataset → _view_vector.
# A plain `AbstractVector{<:Real}` thus becomes a fully interactive line plot
# (scale toggles, selection-based stats, PNG/PDF/CSV export) with no FITS file
# required. Real numbers only; for complex/RGB data the user should compute
# magnitudes / channels first and pass a regular numeric vector.
manta(x::AbstractVector{<:Real}; kwargs...) = manta(load_dataset(x); kwargs...)

# ---- Precompile workload --------------------------------------------------
#
# Exercises the main public entry points in headless mode so that Julia bakes
# the specialized methods into the package's pkgimage cache. This wipes out
"""
    manta_batch(paths; format=:png, save_dir=nothing, prefix="", kwargs...) -> Vector{String}

Headless batch export: render each FITS file in `paths` and save one image
per file.  Returns the list of output file paths written.

### Arguments
- `paths` — `AbstractVector{<:AbstractString}` of FITS file paths.
- `format` — output format, either `:png` (default) or `:pdf`.
- `save_dir` — directory for the exported images.  Defaults to the same
  folder as each source file when `nothing`.
- `prefix` — optional string prepended to every output filename.
- `kwargs...` — forwarded verbatim to `manta(path; activate_gl=false,
  display_fig=false, kwargs...)`.  Common ones: `cmap`, `vmin`, `vmax`,
  `invert`, `figsize`, `scale`.

### Notes
- Each figure is built headlessly (no GL context required) and written via
  CairoMakie through `with_export_backend`.  Existing output files are
  silently overwritten.
- The keep-alive reference is released via `forget!` after each save so
  that Julia's GC can reclaim figure memory between iterations.
- Errors for individual files are caught and logged as `@warn`; processing
  continues with the remaining paths so that one bad file does not abort a
  large batch.
"""
function manta_batch(
    paths::AbstractVector{<:AbstractString};
    format::Symbol = :png,
    save_dir::Union{Nothing,AbstractString} = nothing,
    prefix::AbstractString = "",
    kwargs...,
) :: Vector{String}
    format in (:png, :pdf) ||
        throw(ArgumentError("manta_batch: format must be :png or :pdf, got :$(format)"))

    out_paths = String[]

    for path in paths
        try
            fig = manta(path; activate_gl = false, display_fig = false, kwargs...)

            stem = splitext(basename(path))[1]
            fname = string(prefix, stem, ".", format)
            dir   = save_dir === nothing ? dirname(abspath(path)) : String(save_dir)
            isdir(dir) || mkpath(dir)
            out = joinpath(dir, fname)

            with_export_backend() do
                Makie.save(out, fig)
            end
            forget!(fig)
            push!(out_paths, out)
            @info "manta_batch: saved $(out)"
        catch e
            @warn "manta_batch: failed for $(path)" exception=(e, catch_backtrace())
        end
    end

    return out_paths
end

# most of the TTFP (Time-To-First-Plot) on every subsequent launch.
#
# Every call uses `activate_gl=false, display_fig=false` — the same headless
# mode that `test/runtests.jl` relies on. No GL context is required during
# precompilation, so this stays compatible with the Docker build and CI.
#
# Inputs are tiny in-memory arrays: nothing is written to disk, and the
# `forget!` calls release the keep-alive references that the viewers attach
# to figures.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    # Small in-memory datasets covering the public viewer paths.
    _pc_vec1d  = rand(Float32, 32)
    _pc_img2d  = rand(Float32, 8, 8)
    _pc_cube3d = rand(Float32, 6, 6, 4)
    _pc_hpix_map_ds  = load_dataset(
        Healpix.HealpixMap{Float64,Healpix.RingOrder,Vector{Float64}}(
            collect(1.0:12.0)))
    _pc_hpix_cube_ds = HealpixCubeDataset(
        reshape(Float32.(1:48), 12, 4);
        nside = 1,
        source_id = "precompile_hpix_cube")

    # On-disk FITS cube for the full "open a cube → … → export" path. Written
    # to a throwaway temp dir so the export workload exercises the real FITS
    # loader (not just the in-memory bridge) and the CairoMakie save pipeline.
    # Removed again after the workload — nothing persists past precompilation.
    _pc_tmpdir = mktempdir()
    _pc_fits   = joinpath(_pc_tmpdir, "precompile_cube.fits")
    FITS(_pc_fits, "w") do f
        write(f, reshape(Float32.(1:(6 * 6 * 4)), 6, 6, 4))
    end

    @compile_workload begin
        # 1D vector → line plot + stats + selection path.
        let fig = manta(_pc_vec1d;
                        activate_gl = false, display_fig = false,
                        figsize = (500, 360))
            forget!(fig)
        end

        # 2D image (Float32 matrix) → simple 2D view.
        let fig = manta(_pc_img2d;
                        activate_gl = false, display_fig = false,
                        figsize = (500, 360))
            forget!(fig)
        end

        # 3D cube → slice + spectrum + histogram + moments path.
        let fig = manta(_pc_cube3d;
                        activate_gl = false, display_fig = false,
                        figsize = (700, 450))
            forget!(fig)
        end

        # HEALPix map → Mollweide projection path.
        let fig = manta(_pc_hpix_map_ds;
                        activate_gl = false, display_fig = false,
                        nx = 60, ny = 30, figsize = (500, 320))
            forget!(fig)
        end

        # HEALPix PPV cube → per-channel map + spectrum path.
        let fig = manta(_pc_hpix_cube_ds;
                        activate_gl = false, display_fig = false,
                        nx = 60, ny = 30, figsize = (600, 380))
            forget!(fig)
        end

        # Full export path: open the FITS cube → build the cube viewer headless
        # → save through CairoMakie (with_export_backend). This is the heaviest
        # TTFP cost on a real launch and was previously uncompiled. PNG and PDF
        # go through distinct Cairo surfaces (raster vs vector), so both are
        # warmed. `manta_batch` traps per-file render errors internally (logs a
        # @warn and continues), so this can never abort precompilation.
        manta_batch([_pc_fits];
                    format = :png, save_dir = _pc_tmpdir,
                    activate_gl = false, display_fig = false,
                    figsize = (400, 300))
        manta_batch([_pc_fits];
                    format = :pdf, save_dir = _pc_tmpdir,
                    activate_gl = false, display_fig = false,
                    figsize = (400, 300))
    end

    # Transient: drop the temp dir built for the export workload above.
    rm(_pc_tmpdir; recursive = true, force = true)
end

end # module
