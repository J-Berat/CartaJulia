# path: src/views/cube/CubeViewState.jl
#
# Centralised observable-state contract for the 3D cube viewer.
#
# `CubeViewObservables` is a plain struct that names and types every
# *source* Observable owned by `_view_cube`.  "Source" means an
# `Observable(...)` call — not a derived `lift(...)` chain.
#
# Purpose
# -------
# 1. **Documentation** — one place to see every piece of mutable state in the
#    viewer without scrolling through 2 000 lines of layout and callbacks.
# 2. **Type-safe handoff** — future helper functions (layout builder, panel
#    builder, …) can accept a single `st::CubeViewObservables` argument instead
#    of a 30-keyword named-tuple.
# 3. **Onboarding** — new contributors can read the struct + its docstring to
#    understand what drives the reactive graph before diving into the view code.
#
# Non-state Observables
# ---------------------
# `_view_cube` still owns a few local `Observable(...)` values that are not
# part of this contract: render buffers (`spec_y_raw`, previews), popout-window
# internals, and widget/page-local state.  Those values are intentionally kept
# beside the code that consumes them.
#
# Derived Observables (lift chains)
# ----------------------------------
# The following are *not* stored here because they require outputs from the
# slice pipeline (`slice_disp`) that is built after state construction:
#   · `cm_obs`              lift(cmap_name, invert_cmap)
#   · `clims_auto`          lift(slice_disp)
#   · `clims_obs`           lift(use_manual, clims_auto, clims_manual)
#   · `clims_safe`          lift(clims_obs)        ← used as a cross-bundle interface
#   · `compare_clims_safe`  lift(compare_slice_disp, compare_mode, clims_safe)
#   · All histogram, contour and spectrum *display* lifts
#
# They remain local to `_view_cube` and are passed as kwargs to bundles.
# A future `CubeViewDerived` struct can consolidate them once the slice
# pipeline itself is extracted.
#
# Migration guide
# ---------------
# `_view_cube` now constructs this struct once and aliases its fields locally
# while the remaining large body is gradually split into bundles.  New bundles
# should prefer accepting `st::CubeViewObservables` directly instead of adding
# another long keyword list of source observables.

# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------

"""
    CubeViewObservables

All *source* `Observable`s owned by the cube viewer (`_view_cube`).

Fields are grouped into logical sections; see inline comments.  Every field
is an `Observable{T}` so callers can `lift` or `on` any of them directly.

Construct with [`build_cube_view_state`](@ref).
"""
struct CubeViewObservables
    # ── Navigation ────────────────────────────────────────────────────────
    # The three independent coordinates that uniquely identify what is
    # displayed: slicing axis, primary-cube slice index, compare-cube index.
    axis        ::Observable{Int}   # active slicing axis — 1 (x), 2 (y), or 3 (z)
    idx         ::Observable{Int}   # primary slice index along `axis`
    compare_idx ::Observable{Int}   # comparison cube slice index along `axis`

    # Voxel indices of the current cursor / selection point.
    # (i, j, k) are the dataset-space indices (always in data order);
    # (u, v) are the projected indices on the current slice plane (row, col).
    i_idx ::Observable{Int}
    j_idx ::Observable{Int}
    k_idx ::Observable{Int}
    u_idx ::Observable{Int}   # projected slice-plane index — row
    v_idx ::Observable{Int}   # projected slice-plane index — col

    # ── Colormap & scale ──────────────────────────────────────────────────
    cmap_name   ::Observable{Symbol}  # Makie colormap name, e.g. :viridis
    invert_cmap ::Observable{Bool}    # reverse the colormap?

    img_scale_mode  ::Observable{Symbol}  # :lin | :log10 | :ln  (image panel)
    spec_scale_mode ::Observable{Symbol}  # :lin | :log10 | :ln  (spectrum panel)

    # ── Contrast ──────────────────────────────────────────────────────────
    # When `use_manual` is false the viewer auto-stretches to the slice min/max.
    # When true, `clims_manual` overrides.  Both feed into `clims_obs` and
    # ultimately `clims_safe` (derived; not stored here).
    use_manual    ::Observable{Bool}
    clims_manual  ::Observable{Tuple{Float32,Float32}}

    # ── Gaussian smoothing ────────────────────────────────────────────────
    gauss_on ::Observable{Bool}     # toggle per-slice Gaussian smoothing
    sigma    ::Observable{Float32}  # smoothing σ in pixels

    # ── Display toggles ───────────────────────────────────────────────────
    show_crosshair ::Observable{Bool}   # crosshair lines at selected voxel
    show_marker    ::Observable{Bool}   # point marker at selected voxel
    show_grid      ::Observable{Bool}   # Makie axis grid on all panels
    show_contours  ::Observable{Bool}   # contour overlay on image panel

    # ── Contours ──────────────────────────────────────────────────────────
    # When `contour_use_manual` is false, levels are auto-computed.
    # Manual levels & colors come from the text-box; see `parse_contour_spec`.
    contour_use_manual     ::Observable{Bool}
    contour_manual_levels  ::Observable{Vector{Float32}}
    contour_manual_colors  ::Observable{Vector{String}}

    # ── Selection / interaction ───────────────────────────────────────────
    selection_mode ::Observable{Symbol}                # :point | :box | :circle
    region_shape   ::Observable{Symbol}                # :box | :circle (drag shape)
    region_uvs     ::Observable{Vector{Tuple{Int,Int}}} # pixel UV coords in region
    region_start   ::Observable{Point2f}               # drag-start pixel position
    region_end     ::Observable{Point2f}               # drag-end   pixel position
    region_drag_active ::Observable{Bool}

    zoom_drag_active ::Observable{Bool}
    zoom_drag_start  ::Observable{Point2f}
    zoom_drag_end    ::Observable{Point2f}

    # ── Comparison cube ───────────────────────────────────────────────────
    # `compare_data` holds the aligned Float32 array once loaded, or `nothing`.
    # `compare_mode` controls how A and B are blended in the image panel.
    compare_data         ::Observable{Union{Nothing,Array{Float32,3}}}  # aligned cube B, or nothing
    compare_visible      ::Observable{Bool}    # is a second cube currently loaded?
    compare_name         ::Observable{String}  # display name (basename) of cube B
    compare_path_current ::Observable{String}  # last attempted/loaded path

    # :A — show cube A only in the comparison panel
    # :B — show cube B only  (default)
    # :diff | :ratio | :residuals — arithmetic product
    compare_mode ::Observable{Symbol}

    # ── View product & layout mode ────────────────────────────────────────
    # `view_product` selects what is shown in the image panel.
    # `layout_mode`  controls which secondary panel is visible in the right column.
    # `control_mode` controls which tab is active in the bottom control strip.
    view_product  ::Observable{Symbol}  # :slice | :moment | :rotproj
    moment_order  ::Observable{Int}     # 0 | 1 | 2
    rotation_axis ::Observable{NTuple{3,Float32}}  # arbitrary axis for :rotproj
    rotation_angle ::Observable{Float32}            # degrees for :rotproj
    rotation_projection_mode ::Observable{Symbol}   # :mean | :sum | :max
    layout_mode   ::Observable{Symbol}  # :base | :power_spectrum
    control_mode  ::Observable{Symbol}  # :navigation | :analysis | :export
    anim_playing  ::Observable{Bool}    # auto-advance animation running?

    # ── Mask ──────────────────────────────────────────────────────────────
    # The mask is stored declaratively as a `MaskSource` + a concretised
    # `BitArray{3}`.  The BitArray is recomputed whenever the source changes.
    # `nothing` means "no mask applied".
    mask_source_obs ::Observable{MaskSource}
    mask_bits_obs   ::Observable{Union{Nothing,BitArray{3}}}
    mask_status_obs ::Observable{String}  # human-readable status string

    # ── Spectrum y-axis limits ────────────────────────────────────────────
    # `spec_ylimits_source` tracks where the current limits came from:
    #   :auto   — recomputed from spec data on every refresh
    #   :contrast — pinned to current image clims
    #   :manual — user-specified via the text-boxes
    spec_ylimits_source ::Observable{Symbol}
    spec_ylimits_value  ::Observable{Tuple{Float32,Float32}}

    # ── Histogram ─────────────────────────────────────────────────────────
    hist_mode_obs ::Observable{Symbol}  # :bars | :kde

    # `hist_bins_obs` is clamped to [HIST_BINS_MIN, HIST_BINS_MAX] on write.
    hist_bins_obs ::Observable{Int}

    # Manual x-limits for the histogram axis.
    # When `hist_xlimits_manual` is false the limits track `clims_safe`.
    hist_xlimits_manual       ::Observable{Bool}
    hist_xlimits_manual_value ::Observable{Tuple{Float32,Float32}}

    # Manual y-limits (counts / density) for the histogram axis.
    hist_ylimits_manual       ::Observable{Bool}
    hist_ylimits_manual_value ::Observable{Tuple{Float32,Float32}}

    # ── Status / power-spectrum ───────────────────────────────────────────
    # Single-line status string shown in the bottom status bar.
    ui_status ::Observable{String}

    # Status string local to the power-spectrum panel header.
    ps_layout_status ::Observable{String}
end

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

"""
    build_cube_view_state(;
        cmap             = :viridis,
        invert           = false,
        vmin             = nothing,
        vmax             = nothing,
        hist_mode        = :bars,
        hist_bins        = 64,
        hist_xlimits     = nothing,
        hist_ylimits     = nothing,
        spec_ylimits     = nothing,
    ) -> CubeViewObservables

Allocate and initialise all source Observables for the cube viewer.

Keyword arguments mirror the public kwargs of `_view_cube`/`view_cube`; see
that function's docstring for semantics.  Only *source* Observables are
created here — derived `lift(...)` chains are built later in `_view_cube`
once `slice_disp` (returned by the slice pipeline bundle) is available.

### Example

```julia
st = build_cube_view_state(; cmap = :inferno, vmin = 0.0, vmax = 100.0)
on(st.idx) do i
    println("slice moved to index ", i)
end
```
"""
function build_cube_view_state(;
    cmap         ::Symbol                              = :viridis,
    invert       ::Bool                               = false,
    vmin                                              = nothing,
    vmax                                              = nothing,
    scale        ::Symbol                             = :lin,
    hist_mode    ::Symbol                             = :bars,
    hist_bins    ::Int                                = 64,
    hist_xlimits ::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    hist_ylimits ::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    spec_ylimits ::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
)
    # ── Contrast: resolve initial clims from kwargs ────────────────────────
    _clims_manual, _use_manual = if vmin !== nothing && vmax !== nothing
        vmin_f, vmax_f = Float32(vmin), Float32(vmax)
        if vmin_f == vmax_f               # avoid degenerate zero-width range
            vmin_f = prevfloat(vmin_f)
            vmax_f = nextfloat(vmax_f)
        end
        (vmin_f, vmax_f), true
    else
        (0f0, 1f0), false
    end

    # ── Histogram: parse x/y limit kwargs ────────────────────────────────
    _hist_xlim_manual = hist_xlimits !== nothing
    _hist_xlim_value  = _hist_xlim_manual ?
        parse_histogram_xlimits(
            string(first(hist_xlimits)), string(last(hist_xlimits)))[3] :
        (0f0, 1f0)

    _hist_ylim_manual = hist_ylimits !== nothing
    _hist_ylim_value  = _hist_ylim_manual ?
        parse_histogram_ylimits(
            string(first(hist_ylimits)), string(last(hist_ylimits)))[3] :
        (0f0, 1f0)

    # ── Spectrum: parse y-limit kwarg ─────────────────────────────────────
    _spec_source, _spec_value = if spec_ylimits !== nothing
        :manual,
        parse_spectrum_ylimits(
            string(first(spec_ylimits)), string(last(spec_ylimits)))[3]
    else
        _use_manual ? (:contrast, _clims_manual) : (:auto, (0f0, 1f0))
    end

    CubeViewObservables(
        # Navigation
        Observable(3),               # axis
        Observable(1),               # idx
        Observable(1),               # compare_idx
        Observable(1),               # i_idx
        Observable(1),               # j_idx
        Observable(1),               # k_idx
        Observable(1),               # u_idx
        Observable(1),               # v_idx
        # Colormap & scale
        Observable(cmap),            # cmap_name
        Observable(invert),          # invert_cmap
        Observable(scale),           # img_scale_mode
        Observable(scale),           # spec_scale_mode
        # Contrast
        Observable(_use_manual),     # use_manual
        Observable(_clims_manual),   # clims_manual
        # Smoothing
        Observable(false),           # gauss_on
        Observable(1.5f0),           # sigma
        # Display toggles
        Observable(true),            # show_crosshair
        Observable(true),            # show_marker
        Observable(false),           # show_grid
        Observable(false),           # show_contours
        # Contours
        Observable(false),           # contour_use_manual
        Observable(Float32[]),       # contour_manual_levels
        Observable(String[]),        # contour_manual_colors
        # Selection / interaction
        Observable(:point),          # selection_mode
        Observable(:box),            # region_shape
        Observable(Tuple{Int,Int}[]),# region_uvs
        Observable(Point2f(NaN32, NaN32)), # region_start
        Observable(Point2f(NaN32, NaN32)), # region_end
        Observable(false),           # region_drag_active
        Observable(false),           # zoom_drag_active
        Observable(Point2f(NaN32, NaN32)), # zoom_drag_start
        Observable(Point2f(NaN32, NaN32)), # zoom_drag_end
        # Comparison cube
        Observable{Union{Nothing,Array{Float32,3}}}(nothing),  # compare_data
        Observable(false),           # compare_visible
        Observable(""),              # compare_name
        Observable(""),              # compare_path_current
        Observable(:B),              # compare_mode
        # View product & layout
        Observable(:slice),          # view_product
        Observable(0),               # moment_order
        Observable((0f0, 0f0, 1f0)), # rotation_axis
        Observable(0f0),             # rotation_angle
        Observable(:mean),           # rotation_projection_mode
        Observable(:base),           # layout_mode
        Observable(:navigation),     # control_mode
        Observable(false),           # anim_playing
        # Mask
        Observable{MaskSource}(NoMaskSource()),                   # mask_source_obs
        Observable{Union{Nothing,BitArray{3}}}(nothing),          # mask_bits_obs
        Observable("No mask applied"),                            # mask_status_obs
        # Spectrum limits
        Observable(_spec_source),    # spec_ylimits_source
        Observable(_spec_value),     # spec_ylimits_value
        # Histogram
        Observable(normalize_histogram_mode(hist_mode)),          # hist_mode_obs
        Observable(clamp(hist_bins, HIST_BINS_MIN, HIST_BINS_MAX)), # hist_bins_obs
        Observable(_hist_xlim_manual),       # hist_xlimits_manual
        Observable(_hist_xlim_value),        # hist_xlimits_manual_value
        Observable(_hist_ylim_manual),       # hist_ylimits_manual
        Observable(_hist_ylim_value),        # hist_ylimits_manual_value
        # Status
        Observable(" "),             # ui_status
        Observable(" "),             # ps_layout_status
    )
end
