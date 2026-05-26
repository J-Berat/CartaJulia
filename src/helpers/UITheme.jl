# Shared Makie UI theme for MANTA viewers.

struct MANTAUITheme
    accent::RGBf
    accent_dim::RGBf
    accent_strong::RGBf
    track::RGBf
    surface::RGBf
    surface_hover::RGBf
    surface_active::RGBf
    panel::RGBf
    panel_header::RGBf
    border::RGBf
    border_strong::RGBf
    text::RGBf
    text_muted::RGBf
    background::RGBf
    selection::RGBf
    compare::RGBf
    success::RGBf
end

default_ui_theme() = MANTAUITheme(
    RGBf(0.36, 0.39, 0.92),    # indigo-500
    RGBf(0.68, 0.70, 0.95),    # indigo-300
    RGBf(0.28, 0.31, 0.82),    # indigo-700
    RGBf(0.88, 0.89, 0.91),    # neutral-200
    RGBf(0.985, 0.985, 0.982), # neutral card
    RGBf(0.94, 0.945, 0.945),
    RGBf(0.90, 0.91, 0.91),
    RGBf(0.958, 0.960, 0.958),
    RGBf(0.925, 0.930, 0.930),
    RGBf(0.76, 0.78, 0.80),
    RGBf(0.58, 0.61, 0.65),
    RGBf(0.10, 0.12, 0.20),
    RGBf(0.42, 0.46, 0.56),
    RGBf(0.972, 0.974, 0.972),
    RGBf(1.00, 0.68, 0.12),    # amber/orange selection
    RGBf(0.90, 0.30, 0.16),    # red-orange comparison/residuals
    RGBf(0.10, 0.58, 0.42),    # green saved/loaded states
)

function manta_style_checkbox!(chk, theme::MANTAUITheme = default_ui_theme(); compact::Bool = false)
    chk.size[] = compact ? 18 : 22
    chk.checkmarksize[] = compact ? 0.58 : 0.62
    chk.roundness[] = 0.5
    chk.checkboxstrokewidth[] = 1.4
    chk.checkboxcolor_checked[] = theme.accent
    chk.checkboxcolor_unchecked[] = RGBf(0.96, 0.965, 0.985)
    chk.checkboxstrokecolor_checked[] = theme.accent_strong
    chk.checkboxstrokecolor_unchecked[] = theme.border
    chk.checkmarkcolor_checked[] = :white
    chk.checkmarkcolor_unchecked[] = RGBf(0.65, 0.70, 0.78)
    chk
end

function manta_style_slider!(sl, theme::MANTAUITheme = default_ui_theme(); compact::Bool = false)
    sl.height[] = compact ? 20 : 26
    sl.linewidth[] = compact ? 8 : 10
    sl.color_active[] = theme.accent
    sl.color_active_dimmed[] = theme.accent_dim
    sl.color_inactive[] = theme.track
    sl
end

function manta_style_button!(btn, theme::MANTAUITheme = default_ui_theme(); compact::Bool = false)
    btn.height[] = compact ? 30 : 34
    btn.cornerradius[] = 8
    btn.strokewidth[] = 1.0
    btn.strokecolor[] = theme.border
    btn.buttoncolor[] = theme.surface
    btn.buttoncolor_hover[] = theme.surface_hover
    btn.buttoncolor_active[] = theme.surface_active
    btn.labelcolor[] = theme.text
    btn.labelcolor_hover[] = theme.accent_strong
    btn.labelcolor_active[] = theme.accent_strong
    btn.fontsize[] = compact ? 13 : 14
    btn.padding[] = compact ? (9, 9, 5, 5) : (12, 12, 7, 7)
    btn
end

function manta_style_menu!(menu, theme::MANTAUITheme = default_ui_theme(); compact::Bool = false)
    menu.height[] = compact ? 30 : 34
    menu.width[] = max(something(menu.width[], 0), 96)
    menu.textcolor[] = theme.text
    menu.fontsize[] = compact ? 13 : 14
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

function manta_style_textbox!(tb, theme::MANTAUITheme = default_ui_theme(); compact::Bool = false)
    tb.height[] = compact ? 30 : 34
    tb.fontsize[] = compact ? 13 : 14
    tb.textcolor[] = theme.text
    tb.textcolor_placeholder[] = theme.text_muted
    tb.boxcolor[] = theme.surface
    tb.boxcolor_hover[] = theme.surface_hover
    tb.boxcolor_focused[] = RGBf(1.0, 1.0, 1.0)
    tb.bordercolor[] = theme.border
    tb.bordercolor_hover[] = theme.accent_dim
    tb.bordercolor_focused[] = theme.accent
    tb.borderwidth[] = 1.4
    tb.cornerradius[] = 8
    tb.textpadding[] = compact ? (8, 8, 5, 5) : (10, 10, 7, 7)
    tb
end
