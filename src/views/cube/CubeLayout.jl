# path: src/views/cube/CubeLayout.jl
#
# Builds the complete Makie grid skeleton for the 3D cube viewer.
#
# Extracted from src/views/CubeView.jl (no behaviour change).  Contains
# everything that is "pure Makie geometry": Figure creation, GridLayout
# hierarchy, rowsize!/colsize! calls, card-frame helpers, and mode-bar
# button construction.  Nothing here touches Observables, data, or
# reactive callbacks.
#
# Public entry point: `_build_cube_layout(; fig_size, compact_layout, fig_bg, ui_theme)`
#
# ── Boundary ──────────────────────────────────────────────────────────────
# IN  (returned): Figure, every GridLayout / GridPosition, all card-frame
#                 GridLayouts, mode-bar Buttons, hint + status Labels,
#                 sizing constants, `control_card!` and `control_label!`
#                 helpers.
# OUT (stays in _view_cube): all interactive widgets (Axis, heatmap, Button
#                 inside cards, Slider, Menu, Textbox, Checkbox), the spec /
#                 hist Axes, ps_header widgets, info-panel Label, and all
#                 Observable wiring.
#
# ── Migration notes ───────────────────────────────────────────────────────
# _view_cube currently inlines all of this code.  To migrate:
#
#   1. Call `_build_cube_layout(; fig_size, compact_layout, fig_bg, ui_theme)`
#      at the point where `fig = Figure(...)` currently appears.
#   2. Destructure the returned NamedTuple.
#   3. Delete the inlined block.
#
# The return NamedTuple mirrors every local variable name that downstream
# code (widget creation, Axis placement, bundle calls) already uses, so
# destructuring produces a zero-diff migration for those consumers.

# ---------------------------------------------------------------------------
# Sizing constants
# ---------------------------------------------------------------------------

"""
    CubeLayoutMetrics

Pre-computed size constants derived from `compact_layout` and `fig_size`.
Bundled in a struct so callers can pass them as a single argument to widget
builders that need to know e.g. `spec_axis_height`.
"""
struct CubeLayoutMetrics
    compact_layout    ::Bool
    fig_size          ::Tuple{Int,Int}

    # Panel heights (px)
    spec_axis_height  ::Int   # height of the spectrum Axis
    hist_axis_height  ::Int   # height of the histogram Axis
    ps_header_height  ::Int   # height reserved for the power-spectrum header
    ps_axis_size      ::Int   # square side for image/ps Axes in compact mode

    # Controls strip geometry
    controls_row_heights ::Tuple{Int,Int,Int}
    controls_gap         ::Int   # gap between control-row cells
    controls_height      ::Int   # total height of the controls strip
    card_pad             ::Int   # outer padding inside each card GridLayout
    card_gap             ::Int   # gap between rows/cols inside a card

    # Row gaps
    main_row_gap     ::Int

    # Dynamic row heights (used in :power_spectrum layout mode)
    plot_row_height     ::Int   # main-grid row 1 height when compact
    ps_plot_row_height  ::Int   # row 1 height when stacking heatmap + PSD
end

# ---------------------------------------------------------------------------
# Status footer styling
# ---------------------------------------------------------------------------

const _CUBE_STATUS_SUCCESS_PREFIXES = (
    "saved",
    "loaded",
    "copied",
    "applied",
    "mask applied",
)

const _CUBE_STATUS_ERROR_PREFIXES = (
    "failed",
    "invalid",
    "mask not applied",
    "settings file not found",
    "second cube not found",
    "no 1d points",
)

const _CUBE_STATUS_ACTION_PREFIXES = (
    "animation",
    "automatic",
    "base layout",
    "colorbar",
    "colormap",
    "compare slice",
    "contrast",
    "contours",
    "draw",
    "dual product",
    "enter",
    "gaussian",
    "histogram",
    "image scale",
    "interactive animation",
    "mask cleared",
    "mask reset",
    "moment map",
    "power spectrum",
    "preview slice",
    "provide",
    "redo",
    "selection",
    "showing",
    "slice ",
    "undo",
    "zoom",
)

_cube_status_startswith_any(s::AbstractString, prefixes) =
    any(p -> startswith(s, p), prefixes)

"""
    _cube_status_tone(msg) -> Symbol

Classify the bottom footer text into one of the visual status tones used by
the cube viewer: `:success`, `:error`, `:action`, or `:neutral`.
"""
function _cube_status_tone(msg::AbstractString)
    s = lowercase(strip(String(msg)))
    isempty(s) && return :neutral

    if _cube_status_startswith_any(s, _CUBE_STATUS_ERROR_PREFIXES) ||
       occursin(" not found", s) ||
       occursin(" must ", s) ||
       occursin(" unavailable", s) ||
       startswith(s, "nothing to save")
        return :error
    end

    if _cube_status_startswith_any(s, _CUBE_STATUS_SUCCESS_PREFIXES) ||
       occursin(" exported", s)
        return :success
    end

    if _cube_status_startswith_any(s, _CUBE_STATUS_ACTION_PREFIXES) ||
       occursin(" set to ", s) ||
       endswith(s, " enabled.") ||
       endswith(s, " disabled.") ||
       endswith(s, " restored.") ||
       endswith(s, " cleared.") ||
       endswith(s, " applied.") ||
       endswith(s, " paused.") ||
       endswith(s, " playing.")
        return :action
    end

    return :neutral
end

function _cube_status_color(msg::AbstractString, ui_theme::MANTAUITheme)
    tone = _cube_status_tone(msg)
    tone === :success && return ui_theme.success
    tone === :error   && return ui_theme.error
    tone === :action  && return ui_theme.accent
    return ui_theme.text_muted
end

function _style_cube_status_footer!(status_footer_label, msg::AbstractString,
                                    ui_theme::MANTAUITheme)
    status_footer_label.color[] = _cube_status_color(msg, ui_theme)
    return status_footer_label
end

function _set_cube_status_footer!(status_footer_label, msg::AbstractString,
                                  ui_theme::MANTAUITheme)
    status_footer_label.text[] = String(msg)
    _style_cube_status_footer!(status_footer_label, msg, ui_theme)
end

function CubeLayoutMetrics(fig_size::Tuple{Int,Int}, compact_layout::Bool)
    crh   = compact_layout ? (212, 164, 42) : (214, 146, 46)
    cgap  = compact_layout ? 8 : 16
    cht   = sum(crh) + 2 * cgap
    mrg   = compact_layout ? 8 : 14
    CubeLayoutMetrics(
        compact_layout, fig_size,
        compact_layout ? 185 : 320,          # spec_axis_height
        compact_layout ? 60  : 105,          # hist_axis_height
        compact_layout ? 0   : 90,           # ps_header_height
        compact_layout ? 320 : 520,          # ps_axis_size
        crh, cgap, cht,
        compact_layout ? 9   : 12,           # card_pad
        compact_layout ? 7   : 10,           # card_gap
        mrg,
        compact_layout ? max(320, fig_size[2] - cht - 8 * mrg) : 0,
        max(360, fig_size[2] - cht - 2 * mrg - 16),
    )
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

"""
    _build_cube_layout(; fig_size, compact_layout, fig_bg, ui_theme) -> NamedTuple

Build the complete Makie grid skeleton for the cube viewer and return all
grid references together with sizing constants and helper closures.

### Arguments
| Name             | Description                                              |
|------------------|----------------------------------------------------------|
| `fig_size`       | `Tuple{Int,Int}` — figure size from `_pick_fig_size`     |
| `compact_layout` | `Bool` — pre-computed from `fig_size` thresholds         |
| `fig_bg`         | background colour for `Figure` (from `ui_theme`)         |
| `ui_theme`       | `MANTATheme` — colours for card frames and mode buttons  |

### Return fields
The returned NamedTuple contains, in order:

*Metrics:*
`metrics` — `CubeLayoutMetrics` (all sizing constants as a struct)

*Figure:*
`fig`

*Top-level grids:*
`main_grid`, `img_grid`, `img_a_grid`, `img_cmp_grid`,
`spec_grid`, `info_panel`,
`ps_layout`, `ps_header`, `ps_plot_grid`

*Controls strip:*
`controls_grid`, `mode_bar`,
`export_top`, `analysis_bottom`

*Card frames (GridLayout for widget population):*
`view_card`, `slice_card`, `contrast_card`,
`output_card`, `anim_card`,
`region_card`, `contour_card`,
`hist_card`, `moment_card`, `mask_card`

*Export spacers:*
`export_left_spacer`, `export_mid_spacer`, `export_right_spacer`

*Analysis spacers:*
`analysis_spacer_1`, `analysis_spacer_3`, `analysis_spacer_5`, `analysis_spacer_7`

*Mode-bar buttons (always visible, define the tab navigation structure):*
`mode_nav_btn`, `mode_analysis_btn`, `mode_export_btn`,
`help_btn`, `btn_undo`, `btn_redo`

*Status footer elements:*
`hint_label`, `status_footer_label`

*Helper closures (captured from this scope; pass back to widget builders):*
`control_card!`, `control_label!`, `status_color_for`,
`style_status_footer!`, `set_status_footer!`
"""
function _build_cube_layout(;
    fig_size       ::Tuple{Int,Int},
    compact_layout ::Bool,
    fig_bg,
    ui_theme,
)
    # ── Destructure the theme colours used by card helpers ─────────────────
    ui_panel         = ui_theme.panel
    ui_panel_header  = ui_theme.panel_header
    ui_border        = ui_theme.border
    ui_text          = ui_theme.text
    ui_text_muted    = ui_theme.text_muted

    # ── Sizing constants ───────────────────────────────────────────────────
    m = CubeLayoutMetrics(fig_size, compact_layout)

    # Short aliases kept for local readability (card_pad/gap captured by helpers)
    card_pad   = m.card_pad
    card_gap   = m.card_gap
    crh        = m.controls_row_heights
    cgap       = m.controls_gap
    main_row_gap = m.main_row_gap

    # ── Figure ────────────────────────────────────────────────────────────
    fig = Figure(size = fig_size, backgroundcolor = fig_bg)

    # ── Main grid ─────────────────────────────────────────────────────────
    main_grid = fig[1, 1] = GridLayout()
    colgap!(main_grid, 18)
    rowgap!(main_grid, main_row_gap)

    # Image panel (col 1, row 1 of main_grid).
    # valign=:center vertically centres the heatmap+colorbar block in its cell.
    # tellwidth is intentionally NOT set to false: CompareBundle dynamically
    # resizes img_grid col 2 and the propagation must reach main_grid col 1.
    img_grid     = main_grid[1, 1] = GridLayout(; valign = :center)
    colgap!(img_grid, compact_layout ? 8 : 14)
    img_a_grid   = img_grid[1, 1] = GridLayout()
    img_cmp_grid = img_grid[1, 2] = GridLayout()
    colgap!(img_a_grid,   -8)   # pull colorbar tight against the axis
    colgap!(img_cmp_grid, -8)

    # Right column: spectrum + histogram (base mode) or PSD (power-spectrum mode).
    # Both share [1, 2] of main_grid; visibility is toggled by the layout callbacks.
    spec_grid  = main_grid[1, 2] = GridLayout()
    info_panel = spec_grid[1, 1] = GridLayout(; alignmode = Outside())

    # Power-spectrum panel — overlaid in the same main_grid cell as spec_grid;
    # made invisible in base mode (see UICallbacksBundle / reactivity section).
    ps_layout = main_grid[1, 2] = GridLayout(;
        alignmode = Outside(compact_layout ? 4 : 8),
        halign    = :center,
        valign    = :top,
        tellwidth = false,
        tellheight = false,
    )
    ps_header    = ps_layout[1, 1] = GridLayout()
    colgap!(ps_header, 8)
    rowgap!(ps_header, compact_layout ? 6 : 8)

    ps_plot_grid = ps_layout[2, 1] = GridLayout(2, 4; halign = :center, valign = :top)
    colgap!(ps_plot_grid, -8)
    rowgap!(ps_plot_grid, compact_layout ? 6 : 12)
    rowsize!(ps_layout, 1, Fixed(m.ps_header_height))
    rowsize!(ps_layout, 2, Relative(1))
    colsize!(ps_plot_grid, 1, Auto())
    rowsize!(ps_plot_grid, 1, Auto())
    rowsize!(ps_plot_grid, 2, Auto())

    # ── Controls strip ────────────────────────────────────────────────────
    controls_grid = main_grid[2, 1:2] = GridLayout(; alignmode = Outside())
    colgap!(controls_grid, cgap)
    rowgap!(controls_grid, cgap)
    rowsize!(main_grid, 2, Fixed(m.controls_height))
    compact_layout && rowsize!(main_grid, 1, Fixed(m.plot_row_height))

    # ── Card-frame helpers ────────────────────────────────────────────────
    # These closures capture the sizing constants and theme colours above.
    # They are returned so widget builders can create additional cards with
    # a consistent visual style without importing these captured variables.

    """
    Create a titled card frame in `parent[row, col]`.  Returns the card
    GridLayout; call `card[r, c] = Widget(...)` to populate it.

    The frame consists of three overlapping Box elements:
      • full-height body box (background + border)
      • top-row header band (slightly different background)
      • invisible bottom spacer row (ensures consistent bottom padding)
    """
    function control_card!(
        parent,
        row, col,
        title::AbstractString;
        rows::Int      = 4,
        cols::Int      = 4,
        title_color    = ui_text,
    )
        card = parent[row, col] = GridLayout(;
            alignmode = Outside(card_pad),
            tellwidth  = false,
            tellheight = false,
        )
        body_rows = rows + 1   # +1 for the invisible bottom spacer row
        card_is_dark = ui_theme.background.r < 0.5
        card_border = RGBAf(ui_border.r, ui_border.g, ui_border.b, card_is_dark ? 0.58 : 0.82)
        header_divider = RGBAf(ui_border.r, ui_border.g, ui_border.b, card_is_dark ? 0.30 : 0.50)
        # Full-height card background
        Box(card[1:body_rows, 1:cols];
            color        = ui_panel,
            strokecolor  = card_border,
            strokewidth  = card_is_dark ? 0.8 : 0.9,
            cornerradius = 8,
            z            = -6)
        # Header band
        Box(card[1, 1:cols];
            color        = ui_panel_header,
            strokecolor  = header_divider,
            strokewidth  = 0.8,
            cornerradius = 8,
            z            = -5)
        Label(card[1, 1:cols];
            text      = title,
            halign    = :left,
            tellwidth = false,
            fontsize  = 12,
            color     = title_color,
            padding   = (10, 10, 5, 5))
        # Invisible bottom spacer row (ensures consistent padding at the bottom)
        Box(card[body_rows, 1:cols]; color = :transparent, strokewidth = 0, z = -7)
        rowsize!(card, 1, Fixed(compact_layout ? 28 : 32))
        rowsize!(card, body_rows, Fixed(compact_layout ? 10 : 12))
        rowgap!(card, compact_layout ? 7 : 9)
        colgap!(card, card_gap)
        return card
    end

    # Muted label on the left side of a card row.
    control_label!(layout, pos, txt) = Label(
        layout[pos...];
        text      = txt,
        halign    = :left,
        tellwidth = false,
        fontsize  = 13,
        color     = ui_text_muted,
    )

    # ── Mode bar ──────────────────────────────────────────────────────────
    # Row 3 of controls_grid: Navigation | Analysis | Export tabs + Undo/Redo.
    mode_bar = controls_grid[3, 1:3] = GridLayout(; alignmode = Outside(0), halign = :center)
    colgap!(mode_bar, compact_layout ? 8 : 12)

    # These buttons define the tab structure and are always visible; their
    # visual state (active / inactive) is managed by `set_mode_button_active!`
    # in the main view body.
    mode_segment = mode_bar[1, 1] = GridLayout(; alignmode = Outside(0))
    colgap!(mode_segment, 0)
    mode_nav_btn      = Button(mode_segment[1, 1]; label = "$(MANTA_ICONS.nav) Navigation", width = 154, height = 32)
    mode_analysis_btn = Button(mode_segment[1, 2]; label = "$(MANTA_ICONS.analysis) Analysis", width = 140, height = 32)
    mode_export_btn   = Button(mode_segment[1, 3]; label = "$(MANTA_ICONS.export_icon) Export", width = 118, height = 32)
    foreach(c -> colsize!(mode_segment, c, Auto()), 1:3)
    help_btn          = Button(mode_bar[1, 2]; label = MANTA_ICONS.help,                 width = 46,  height = 32)
    btn_undo          = Button(mode_bar[1, 3]; label = MANTA_ICONS.undo,                 width = 46,  height = 32)
    btn_redo          = Button(mode_bar[1, 4]; label = MANTA_ICONS.redo,                 width = 46,  height = 32)
    foreach(c -> colsize!(mode_bar, c, Auto()), 1:4)

    # ── Navigation mode cards (row 1 of controls_grid) ────────────────────
    # Three cards share row 1 with the Analysis-mode cards via the same grid
    # cells; visibility is toggled by `refresh_control_mode!` in _view_cube.
    view_card     = control_card!(controls_grid, 1, 1, "View";     rows = 6, cols = 4)
    slice_card    = control_card!(controls_grid, 1, 2, "Slice";    rows = 5, cols = 5)
    display_card  = control_card!(controls_grid, 1, 3, "Display";  rows = 5, cols = 4)

    # ── Analysis mode cards (row 1 of controls_grid) ──────────────────────
    # Overlaid in the same cells as the Navigation cards above.
    contrast_card = control_card!(controls_grid, 1, 1,
                        "$(MANTA_ICONS.contrast) Contrast";        rows = 4, cols = 5)

    # ── Export mode cards (row 1, full-width sub-grid) ────────────────────
    # Only two cards; centred via a spacer-column layout so they don't span
    # the full width of the three-column controls_grid.
    export_top = controls_grid[1, 1:3] = GridLayout(; alignmode = Outside(0))
    colgap!(export_top, cgap)

    export_left_spacer  = Box(export_top[1, 1]; color = :transparent, strokewidth = 0)
    output_card         = control_card!(export_top, 1, 2, "Output";    rows = 5, cols = 5)
    export_mid_spacer   = Box(export_top[1, 3]; color = :transparent, strokewidth = 0)
    anim_card           = control_card!(export_top, 1, 4, "Animation"; rows = 4, cols = 5)
    export_right_spacer = Box(export_top[1, 5]; color = :transparent, strokewidth = 0)

    colsize!(export_top, 1, Relative(1))
    colsize!(export_top, 2, Fixed(compact_layout ? 660 : 740))
    colsize!(export_top, 3, Fixed(compact_layout ? 14  :  22))
    colsize!(export_top, 4, Fixed(compact_layout ? 430 : 480))
    colsize!(export_top, 5, Relative(1))

    # ── Analysis mode: row 1 (right two cards in controls_grid) ──────────
    region_card  = control_card!(controls_grid, 1, 2,
                       "$(MANTA_ICONS.selection) Selection Spectrum";
                       rows = 4, cols = 4, title_color = ui_theme.selection)
    contour_card = control_card!(controls_grid, 1, 3, "Contours"; rows = 3, cols = 5)

    # ── Analysis mode: row 2 (Products + Histogram, centred) ──────────────
    # Transparent spacer Boxes in odd columns make the centred sub-grid
    # addressable by colsize!.
    analysis_bottom = controls_grid[2, 1:3] = GridLayout(; alignmode = Outside(0))
    colgap!(analysis_bottom, cgap)

    analysis_spacer_1 = Box(analysis_bottom[1, 1]; color = :transparent, strokewidth = 0)
    mask_card         = control_card!(analysis_bottom, 1, 2,
                             "$(MANTA_ICONS.mask) Mask";
                             rows = 4, cols = 5, title_color = ui_theme.mask)
    analysis_spacer_3 = Box(analysis_bottom[1, 3]; color = :transparent, strokewidth = 0)
    moment_card       = control_card!(analysis_bottom, 1, 4, "Products"; rows = 4, cols = 5)
    analysis_spacer_5 = Box(analysis_bottom[1, 5]; color = :transparent, strokewidth = 0)
    hist_card         = control_card!(analysis_bottom, 1, 6,
                             "$(MANTA_ICONS.histogram) Histogram"; rows = 5, cols = 5)
    analysis_spacer_7 = Box(analysis_bottom[1, 7]; color = :transparent, strokewidth = 0)

    colsize!(analysis_bottom, 1, Relative(1))
    colsize!(analysis_bottom, 2, Fixed(compact_layout ? 360 : 420))  # Mask
    colsize!(analysis_bottom, 3, Fixed(compact_layout ?  14 :  22))
    colsize!(analysis_bottom, 4, Fixed(compact_layout ? 460 : 520))  # Products
    colsize!(analysis_bottom, 5, Fixed(compact_layout ?  14 :  22))
    colsize!(analysis_bottom, 6, Fixed(compact_layout ? 430 : 500))  # Histogram
    colsize!(analysis_bottom, 7, Relative(1))

    # ── Final controls_grid sizing ────────────────────────────────────────
    foreach(c -> colsize!(controls_grid, c, Relative(1 / 3)), 1:3)
    rowsize!(controls_grid, 1, Fixed(crh[1]))
    rowsize!(controls_grid, 2, Fixed(crh[2]))
    rowsize!(controls_grid, 3, Fixed(crh[3]))

    # ── Hint line + status footer ─────────────────────────────────────────
    # These Labels occupy rows 3 and 4 of main_grid.  In compact mode they
    # are hidden (display_fig path already hides them via compact_layout check
    # in _view_cube; hiding here would fire before ui_status is wired).
    hint_label = Label(
        main_grid[3, 2];
        text      = "arrows: move crosshair    left-click: pick / draw region" *
                    "    right-drag: zoom    i: invert colormap",
        halign    = :right,
        fontsize  = 13,
        color     = ui_text_muted,
        tellwidth = false,
    )
    # The text Observable is wired to ui_status in _view_cube after construction.
    # We use a placeholder here and the caller re-assigns `text` via the Observable.
    status_footer_label = Label(
        main_grid[4, 1:2];
        text      = " ",   # replaced by ui_status Observable in _view_cube
        color     = ui_text_muted,
        halign    = :left,
        tellwidth = false,
    )

    status_color_for(msg::AbstractString) = _cube_status_color(msg, ui_theme)
    style_status_footer!(msg::AbstractString) =
        _style_cube_status_footer!(status_footer_label, msg, ui_theme)
    set_status_footer!(msg::AbstractString) =
        _set_cube_status_footer!(status_footer_label, msg, ui_theme)

    # ── Return ────────────────────────────────────────────────────────────
    return (;
        # Sizing constants (bundled)
        metrics = m,

        # Figure
        fig,

        # Top-level grids
        main_grid, img_grid, img_a_grid, img_cmp_grid,
        spec_grid, info_panel,
        ps_layout, ps_header, ps_plot_grid,

        # Controls strip
        controls_grid, mode_bar, mode_segment,
        export_top, analysis_bottom,

        # Card frames
        view_card, slice_card, display_card, contrast_card,
        output_card, anim_card,
        region_card, contour_card,
        hist_card, moment_card, mask_card,

        # Spacers
        export_left_spacer, export_mid_spacer, export_right_spacer,
        analysis_spacer_1, analysis_spacer_3, analysis_spacer_5, analysis_spacer_7,

        # Mode-bar buttons
        mode_nav_btn, mode_analysis_btn, mode_export_btn,
        help_btn, btn_undo, btn_redo,

        # Status footer
        hint_label, status_footer_label,

        # Helper closures (pass to widget builders that need consistent card style)
        control_card!, control_label!,
        status_color_for, style_status_footer!, set_status_footer!,
    )
end
