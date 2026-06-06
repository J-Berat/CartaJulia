# Shared Makie UI theme for MANTA viewers.
# Exports: default_ui_theme, dark_ui_theme, current_ui_theme,
#          set_dark_mode!, is_dark_mode, _makie_theme_for

struct MANTAUITheme
    panel::RGBf
    panel_header::RGBf
    accent::RGBf
    accent_dim::RGBf
    accent_strong::RGBf
    track::RGBf
    surface::RGBf
    surface_hover::RGBf
    surface_active::RGBf
    border::RGBf
    border_strong::RGBf
    text::RGBf
    text_muted::RGBf
    background::RGBf
    selection::RGBf
    compare::RGBf
    success::RGBf
    mask::RGBf
    error::RGBf
end

default_ui_theme() = MANTAUITheme(
    RGBf(0.958, 0.960, 0.958), # panel        — soft grey panels
    RGBf(0.902, 0.910, 0.914), # panel_header — clearer grey header
    RGBf(0.36, 0.39, 0.92),    # accent       — indigo-500
    RGBf(0.68, 0.70, 0.95),    # accent_dim   — indigo-300
    RGBf(0.28, 0.31, 0.82),    # accent_strong — indigo-700
    RGBf(0.88, 0.89, 0.91),    # track        — neutral-200
    RGBf(0.985, 0.985, 0.982), # surface      — neutral card
    RGBf(0.94, 0.945, 0.945),  # surface_hover
    RGBf(0.90, 0.91, 0.91),    # surface_active
    RGBf(0.76, 0.78, 0.80),    # border
    RGBf(0.58, 0.61, 0.65),    # border_strong
    RGBf(0.10, 0.12, 0.20),    # text
    RGBf(0.42, 0.46, 0.56),    # text_muted
    RGBf(0.972, 0.974, 0.972), # background
    RGBf(1.00, 0.68, 0.12),    # selection    — amber/orange
    RGBf(0.90, 0.30, 0.16),    # compare      — red-orange residuals
    RGBf(0.10, 0.58, 0.42),    # success      — green saved/loaded
    RGBf(0.04, 0.62, 0.72),    # mask         — cyan mask affordance
    RGBf(0.72, 0.16, 0.18),    # error        — sober red
)

"""
    dark_ui_theme() -> MANTAUITheme

Anthracite dark palette — optimised for astronomical data visualisation:
deep backgrounds let colormaps pop, muted blue-grey text stays readable
without competing with data, and bright amber/orange accents echo the
warm glow of emission maps.
"""
dark_ui_theme() = MANTAUITheme(
    RGBf(0.19, 0.22, 0.28),    # panel        — anthracite, clearly above bg
    RGBf(0.25, 0.29, 0.36),    # panel_header — lighter band, reads as a title bar
    RGBf(0.54, 0.62, 1.00),    # accent       — luminous indigo
    RGBf(0.64, 0.70, 1.00),    # accent_dim   — indigo tint
    RGBf(0.74, 0.80, 1.00),    # accent_strong — light indigo highlight
    RGBf(0.30, 0.34, 0.40),    # track        — dark slate
    RGBf(0.23, 0.26, 0.32),    # surface      — night-grey cards
    RGBf(0.28, 0.32, 0.39),    # surface_hover
    RGBf(0.33, 0.37, 0.45),    # surface_active
    RGBf(0.42, 0.47, 0.56),    # border       — visible against panel + bg
    RGBf(0.58, 0.64, 0.74),    # border_strong — axis spines / card edges
    RGBf(0.90, 0.92, 0.96),    # text         — off-white
    RGBf(0.66, 0.71, 0.80),    # text_muted   — readable blue-grey
    RGBf(0.09, 0.10, 0.13),    # background   — near-black
    RGBf(1.00, 0.72, 0.18),    # selection    — bright amber
    RGBf(1.00, 0.42, 0.24),    # compare      — hot orange
    RGBf(0.22, 0.84, 0.58),    # success      — vivid mint
    RGBf(0.35, 0.86, 0.92),    # mask         — bright cyan
    RGBf(0.92, 0.30, 0.34),    # error        — clear sober red
)

############################
# Global dark-mode toggle
############################

"""Module-level dark-mode flag (false = light, true = dark)."""
const _MANTA_DARK_MODE = Ref(false)

"""    is_dark_mode() -> Bool"""
is_dark_mode() = _MANTA_DARK_MODE[]

"""
    current_ui_theme() -> MANTAUITheme

Return the active MANTA UI theme: dark when `set_dark_mode!(true)` has been
called in this Julia session, light otherwise.
"""
current_ui_theme() = _MANTA_DARK_MODE[] ? dark_ui_theme() : default_ui_theme()

"""
    _theme_rgba(c::RGBf, alpha::Real) -> RGBAf

Return the theme colour `c` with an `alpha` channel applied.

Centralises the `RGBAf(c.r, c.g, c.b, α)` boilerplate that recurs across the
viewers whenever a translucent variant of a `MANTAUITheme` field is needed —
card borders, header dividers, focus-bar backgrounds, axis grid lines, etc.
Routing these through a single helper keeps the derived colour tied to the
active theme (so it follows `set_dark_mode!`) instead of being re-typed inline
at every call site.
"""
_theme_rgba(c::RGBf, alpha::Real) = RGBAf(c.r, c.g, c.b, alpha)

"""
    _makie_theme_for(t::MANTAUITheme) -> Makie.Theme

Build a `Makie.Theme` whose axis/colorbar/figure colours match the MANTA
UI theme `t`. Intended for use with `Makie.with_theme` or
`Makie.update_theme!` inside MANTA viewer constructors so that tick labels,
grid lines, spines and figure backgrounds follow the chosen palette without
needing per-axis overrides scattered across every view file.
"""
function _makie_theme_for(t::MANTAUITheme)
    grid_c  = _theme_rgba(t.border, 0.45)
    mgrid_c = _theme_rgba(t.border, 0.20)
    spine_c = t.border_strong
    text_c  = t.text
    tick_c  = t.text_muted
    Makie.Theme(
        backgroundcolor = t.background,
        textcolor       = text_c,
        Axis = (
            backgroundcolor  = t.panel,
            xgridcolor       = grid_c,
            ygridcolor       = grid_c,
            xminorgridcolor  = mgrid_c,
            yminorgridcolor  = mgrid_c,
            xticklabelcolor  = tick_c,
            yticklabelcolor  = tick_c,
            xlabelcolor      = text_c,
            ylabelcolor      = text_c,
            titlecolor       = text_c,
            topspinecolor    = spine_c,
            bottomspinecolor = spine_c,
            leftspinecolor   = spine_c,
            rightspinecolor  = spine_c,
            xtickcolor       = spine_c,
            ytickcolor       = spine_c,
        ),
        Colorbar = (
            labelcolor      = text_c,
            ticklabelcolor  = tick_c,
            topspinecolor   = spine_c,
            bottomspinecolor = spine_c,
            leftspinecolor  = spine_c,
            rightspinecolor = spine_c,
        ),
        Legend = (
            bgcolor    = t.panel,
            labelcolor = text_c,
            framecolor = spine_c,
        ),
    )
end

"""
    set_dark_mode!(dark::Bool)

Switch MANTA between its light (default) and dark themes globally for this
Julia session. Call **before** opening any viewer:

```julia
MANTA.set_dark_mode!(true)   # anthracite background, bright colormaps
MANTA.manta("mycube.fits")

MANTA.set_dark_mode!(false)  # back to the standard light theme
```

Internally this updates both the MANTA internal flag (used by viewer panel
and widget colours) and the Makie global theme (used by axes, tick labels,
grid lines and figure backgrounds).
"""
function set_dark_mode!(dark::Bool)
    _MANTA_DARK_MODE[] = dark
    t = current_ui_theme()
    Makie.update_theme!(_makie_theme_for(t))
    return nothing
end

function manta_style_checkbox!(chk, theme::MANTAUITheme = current_ui_theme(); compact::Bool = false)
    chk.size[] = compact ? 18 : 22
    chk.checkmarksize[] = compact ? 0.58 : 0.62
    chk.roundness[] = 0.5
    chk.checkboxstrokewidth[] = 1.4
    chk.checkboxcolor_checked[] = theme.accent
    chk.checkboxcolor_unchecked[] = theme.surface
    chk.checkboxstrokecolor_checked[] = theme.accent_strong
    chk.checkboxstrokecolor_unchecked[] = theme.border
    chk.checkmarkcolor_checked[] = :white
    chk.checkmarkcolor_unchecked[] = RGBf(0.65, 0.70, 0.78)
    chk
end

function manta_style_slider!(sl, theme::MANTAUITheme = current_ui_theme(); compact::Bool = false)
    sl.height[] = compact ? 20 : 26
    sl.linewidth[] = compact ? 8 : 10
    sl.color_active[] = theme.accent
    sl.color_active_dimmed[] = theme.accent_dim
    sl.color_inactive[] = theme.track
    sl
end

function manta_style_button!(btn, theme::MANTAUITheme = current_ui_theme(); compact::Bool = false)
    btn.height[] = compact ? 30 : 34
    btn.cornerradius[] = UI_CORNER_RADIUS
    btn.strokewidth[] = 1.0
    btn.strokecolor[] = theme.border
    btn.buttoncolor[] = theme.surface
    btn.buttoncolor_hover[] = theme.surface_hover
    btn.buttoncolor_active[] = theme.surface_active
    btn.labelcolor[] = theme.text
    btn.labelcolor_hover[] = theme.accent_strong
    btn.labelcolor_active[] = theme.accent_strong
    btn.fontsize[] = compact ? UI_FS_CAPTION : UI_FS_BODY
    btn.padding[] = compact ? (9, 9, 5, 5) : (12, 12, 7, 7)
    btn
end

"""
    manta_style_button_primary!(btn, theme; compact)

**Primary** panel button (Apply, Save, Export GIF, Export FITS…).
`accent` background with white text — draws the eye to the confirm action.
"""
function manta_style_button_primary!(btn, theme::MANTAUITheme = current_ui_theme(); compact::Bool = false)
    manta_style_button!(btn, theme; compact)
    btn.buttoncolor[]        = theme.accent
    btn.buttoncolor_hover[]  = theme.accent_strong
    btn.buttoncolor_active[] = theme.accent_strong
    btn.labelcolor[]         = :white
    btn.labelcolor_hover[]   = :white
    btn.labelcolor_active[]  = :white
    btn.strokecolor[]        = theme.accent_strong
    btn
end

"""
    manta_style_button_ghost!(btn, theme; compact)

**Unobtrusive** button for reset or secondary utility actions
(Auto, Reset zoom, Clear, Help…). Near-transparent background, `text_muted`
label, thin border — visible but not distracting.
"""
function manta_style_button_ghost!(btn, theme::MANTAUITheme = current_ui_theme(); compact::Bool = false)
    manta_style_button!(btn, theme; compact)
    btn.buttoncolor[]        = theme.panel
    btn.buttoncolor_hover[]  = theme.surface_hover
    btn.buttoncolor_active[] = theme.surface_active
    btn.labelcolor[]         = theme.text_muted
    btn.labelcolor_hover[]   = theme.text
    btn.labelcolor_active[]  = theme.text
    btn.strokewidth[]        = 0.8
    btn
end

"""
    manta_style_segmented_button!(btn, theme; compact, active)

Segmented mode switch button for Navigation / Analysis / Export. Uses the
calmer panel surface, reserving the accent for the active label and border.
"""
function manta_style_segmented_button!(btn, theme::MANTAUITheme = current_ui_theme();
                                       compact::Bool = false, active::Bool = false)
    manta_style_button!(btn, theme; compact)
    btn.cornerradius[]       = compact ? 6 : 7
    btn.strokewidth[]        = active ? 1.2 : 0.8
    btn.strokecolor[]        = active ? theme.accent : theme.border
    btn.buttoncolor[]        = active ? theme.surface_active : theme.panel
    btn.buttoncolor_hover[]  = active ? theme.surface_active : theme.surface_hover
    btn.buttoncolor_active[] = theme.surface_active
    btn.labelcolor[]         = active ? theme.accent_strong : theme.text_muted
    btn.labelcolor_hover[]   = active ? theme.accent_strong : theme.text
    btn.labelcolor_active[]  = theme.accent_strong
    btn
end

function manta_style_menu!(menu, theme::MANTAUITheme = current_ui_theme(); compact::Bool = false)
    menu.height[] = compact ? 30 : 34
    menu.width[] = max(something(menu.width[], 0), 96)
    menu.textcolor[] = theme.text
    menu.fontsize[] = compact ? UI_FS_CAPTION : UI_FS_BODY
    menu.dropdown_arrow_color[] = theme.accent
    menu.dropdown_arrow_size[] = compact ? 10 : 11
    menu.textpadding[] = compact ? (8, 8, 5, 5) : (10, 10, 7, 7)
    menu.cell_color_inactive_even[] = theme.surface
    menu.cell_color_inactive_odd[] = theme.surface
    menu.selection_cell_color_inactive[] = theme.surface
    menu.cell_color_hover[] = theme.surface_hover
    menu.cell_color_active[] = theme.surface_active
    menu
end

function manta_style_menu!(selector::ColormapSelector, theme::MANTAUITheme = current_ui_theme(); compact::Bool = false)
    h = compact ? 30 : 34
    selector.height[] = h
    selector.fontsize[] = compact ? UI_FS_CAPTION : UI_FS_BODY
    selector.textpadding[] = compact ? (8, 8, 5, 5) : (10, 10, 7, 7)
    selector.dropdown_arrow_size[] = compact ? 10 : 11
    rowsize!(selector.layout, 1, Fixed(h))
    selector
end

function manta_style_textbox!(tb, theme::MANTAUITheme = current_ui_theme(); compact::Bool = false)
    tb.height[] = compact ? 30 : 34
    tb.fontsize[] = compact ? UI_FS_CAPTION : UI_FS_BODY
    tb.textcolor[] = theme.text
    tb.textcolor_placeholder[] = theme.text_muted
    tb.boxcolor[] = theme.surface
    tb.boxcolor_hover[] = theme.surface_hover
    tb.boxcolor_focused[] = theme.surface_active
    tb.bordercolor[] = theme.border
    tb.bordercolor_hover[] = theme.accent_dim
    tb.bordercolor_focused[] = theme.accent
    tb.borderwidth[] = 1.4
    tb.cornerradius[] = UI_CORNER_RADIUS
    tb.textpadding[] = compact ? (8, 8, 5, 5) : (10, 10, 7, 7)
    tb
end

"""
    manta_flag_textbox!(tb, ok::Bool, theme = current_ui_theme())

Paint a textbox border to signal input validity. When `ok` is `false`
(the parser rejected the field — empty-but-required or malformed) all three
border states are painted with the theme `error` red so the offending box
is visually obvious instead of failing silently; when `ok` is `true` the
normal themed border palette (matching `manta_style_textbox!`) is restored.

Only the border colours are touched, so the box keeps the size / padding /
text colours set at construction. Returns `tb`.
"""
function manta_flag_textbox!(tb, ok::Bool, theme::MANTAUITheme = current_ui_theme())
    if ok
        tb.bordercolor[]         = theme.border
        tb.bordercolor_hover[]   = theme.accent_dim
        tb.bordercolor_focused[] = theme.accent
    else
        tb.bordercolor[]         = theme.error
        tb.bordercolor_hover[]   = theme.error
        tb.bordercolor_focused[] = theme.error
    end
    return tb
end
