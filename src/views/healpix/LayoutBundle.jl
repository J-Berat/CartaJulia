# path: src/views/healpix/LayoutBundle.jl
#
# Shared HEALPix viewer layout helpers.
#
# These functions are included into the parent MANTA module, like the cube
# bundles. They keep common Makie layout wiring out of the map and PPV cube
# viewers while preserving the existing callback flow in those viewers.

function _healpix_set_block_visible!(block, visible::Bool)
    try; block.visible[] = visible;            catch; end
    try; block.scene.visible[] = visible;      catch; end
    try; block.blockscene.visible[] = visible; catch; end
    nothing
end

function _healpix_set_layout_contents_visible!(layout, visible::Bool)
    for block in try; contents(layout); catch; Any[]; end
        _healpix_set_block_visible!(block, visible)
        block isa GridLayout && _healpix_set_layout_contents_visible!(block, visible)
    end
    nothing
end

function _healpix_control_card!(
    parent,
    row::Integer,
    col,
    title::AbstractString;
    rows::Int = 4,
    cols::Int = 4,
    card_pad,
    card_gap,
    compact_layout::Bool,
    ui_theme,
)
    card = parent[row, col] = GridLayout(;
        alignmode = Outside(card_pad), tellwidth = false, tellheight = false)
    body_rows = rows + 1
    card_is_dark = ui_theme.background.r < 0.5
    card_border = _theme_rgba(ui_theme.border, card_is_dark ? 0.58 : 0.82)
    header_divider = _theme_rgba(ui_theme.border, card_is_dark ? 0.30 : 0.50)
    Box(card[1:body_rows, 1:cols]; color = ui_theme.panel, strokecolor = card_border,
        strokewidth = card_is_dark ? 0.8 : 0.9, cornerradius = 8, z = -6)
    Box(card[1, 1:cols]; color = ui_theme.panel_header, strokecolor = header_divider,
        strokewidth = 0.8, cornerradius = 8, z = -5)
    Label(card[1, 1:cols]; text = title, halign = :left, tellwidth = false,
        fontsize = 12, color = ui_theme.text, padding = (10, 10, 5, 5))
    Box(card[body_rows, 1:cols]; color = :transparent, strokewidth = 0, z = -7)
    rowsize!(card, 1, Fixed(compact_layout ? 28 : 32))
    rowsize!(card, body_rows, Fixed(compact_layout ? 10 : 12))
    rowgap!(card, card_gap)
    colgap!(card, card_gap)
    return card
end

function _healpix_ctrl_label!(layout, pos, txt; font_sz::Real, ui_theme)
    Label(layout[pos...]; text = txt, halign = :left,
        tellwidth = false, fontsize = font_sz, color = ui_theme.text_muted)
end

function _healpix_focus_bar!(position; tight_layout::Bool, compact_layout::Bool, ui_theme)
    focus_bar = GridLayout(
        position;
        alignmode = Outside(0),
        halign = :center,
        valign = :center,
        tellwidth = false,
        tellheight = false,
    )
    colgap!(focus_bar, tight_layout ? 4 : 6)
    Box(focus_bar[1, 1:5];
        color = _theme_rgba(ui_theme.panel, 0.94),
        strokecolor = _theme_rgba(ui_theme.border, 0.75),
        strokewidth = 0.9,
        cornerradius = 8,
        z = -5)
    Label(focus_bar[1, 1];
        text = "Focus",
        halign = :left,
        tellwidth = false,
        fontsize = tight_layout ? 12 : 13,
        color = ui_theme.text,
        padding = (10, 4, 5, 5))
    focus_exit_btn = Button(focus_bar[1, 2]; label = "Exit", width = 62, height = tight_layout ? 28 : 30)
    focus_fit_btn = Button(focus_bar[1, 3]; label = MANTA_ICONS.fit, width = 42, height = tight_layout ? 28 : 30)
    focus_auto_btn = Button(focus_bar[1, 4]; label = MANTA_ICONS.contrast, width = 42, height = tight_layout ? 28 : 30)
    focus_help_btn = Button(focus_bar[1, 5]; label = MANTA_ICONS.help, width = 42, height = tight_layout ? 28 : 30)
    foreach(c -> colsize!(focus_bar, c, Auto()), 1:5)
    foreach(w -> manta_style_button_ghost!(w, ui_theme; compact = compact_layout),
            (focus_exit_btn, focus_fit_btn, focus_auto_btn, focus_help_btn))
    _healpix_set_layout_contents_visible!(focus_bar, false)
    return (;
        grid = focus_bar,
        exit_btn = focus_exit_btn,
        fit_btn = focus_fit_btn,
        auto_btn = focus_auto_btn,
        help_btn = focus_help_btn,
    )
end

function _healpix_power_spectrum_panel!(
    position;
    tight_layout::Bool,
    compact_layout::Bool,
    font_sz::Real,
    ui_theme,
)
    ps_cl_x = Observable(Float64[1.0])
    ps_cl_y = Observable(Float64[eps(Float64)])
    ps_dl_x = Observable(Float64[1.0])
    ps_dl_y = Observable(Float64[eps(Float64)])
    ps_title_cl = Observable(latexstring("\\text{HEALPix}\\;C_\\ell"))
    ps_title_dl = Observable(latexstring("\\text{HEALPix}\\;D_\\ell"))
    ps_status = Observable(latexstring("\\text{Press }p\\text{ to compute }C_\\ell\\text{ from spherical harmonics.}"))

    ps_grid = GridLayout(position; alignmode = Outside(8), tellwidth = false, tellheight = false)
    ps_header = ps_grid[1, 1:2] = GridLayout(; alignmode = Outside(0))
    ps_back_btn = Button(ps_header[1, 1]; label = "Map", width = 74, height = tight_layout ? 28 : 30)
    manta_style_button_ghost!(ps_back_btn, ui_theme; compact = compact_layout)
    Label(ps_header[1, 2];
        text = L"\text{HEALPix spherical-harmonic power spectrum}",
        halign = :left, tellwidth = false, fontsize = tight_layout ? 16 : 18,
        color = ui_theme.text)
    colsize!(ps_header, 1, Fixed(82))
    colsize!(ps_header, 2, Relative(1))

    ax_ps_cl = Axis(ps_grid[2, 1];
        title = ps_title_cl,
        xlabel = L"\ell",
        ylabel = L"C_\ell",
        xscale = log10,
        yscale = log10,
        backgroundcolor = ui_theme.panel,
        xgridcolor = (ui_theme.border, 0.35),
        ygridcolor = (ui_theme.border, 0.35),
        xtickformat = _latex_tick_formatter,
        ytickformat = _latex_tick_formatter,
        titlecolor = ui_theme.text,
        xlabelcolor = ui_theme.text,
        ylabelcolor = ui_theme.text,
        xticklabelcolor = ui_theme.text_muted,
        yticklabelcolor = ui_theme.text_muted)
    ax_ps_dl = Axis(ps_grid[2, 2];
        title = ps_title_dl,
        xlabel = L"\ell",
        ylabel = L"D_\ell = \ell(\ell+1)C_\ell/(2\pi)",
        xscale = log10,
        yscale = log10,
        backgroundcolor = ui_theme.panel,
        xgridcolor = (ui_theme.border, 0.35),
        ygridcolor = (ui_theme.border, 0.35),
        xtickformat = _latex_tick_formatter,
        ytickformat = _latex_tick_formatter,
        titlecolor = ui_theme.text,
        xlabelcolor = ui_theme.text,
        ylabelcolor = ui_theme.text,
        xticklabelcolor = ui_theme.text_muted,
        yticklabelcolor = ui_theme.text_muted)
    lines!(ax_ps_cl, ps_cl_x, ps_cl_y; color = ui_theme.accent, linewidth = 2.2)
    scatter!(ax_ps_cl, ps_cl_x, ps_cl_y; color = ui_theme.accent, markersize = 4)
    lines!(ax_ps_dl, ps_dl_x, ps_dl_y; color = ui_theme.selection, linewidth = 2.2)
    scatter!(ax_ps_dl, ps_dl_x, ps_dl_y; color = ui_theme.selection, markersize = 4)
    guard_log_zoom!(ax_ps_cl)
    guard_log_zoom!(ax_ps_dl)
    Label(ps_grid[3, 1:2], ps_status;
        halign = :left, tellwidth = false, fontsize = font_sz,
        color = ui_theme.text_muted)
    rowsize!(ps_grid, 1, Fixed(tight_layout ? 34 : 40))
    rowsize!(ps_grid, 2, Relative(1))
    rowsize!(ps_grid, 3, Fixed(tight_layout ? 24 : 30))
    colsize!(ps_grid, 1, Relative(1 / 2))
    colsize!(ps_grid, 2, Relative(1 / 2))
    colgap!(ps_grid, tight_layout ? 8 : 14)
    rowgap!(ps_grid, tight_layout ? 6 : 10)
    _healpix_set_layout_contents_visible!(ps_grid, false)

    return (;
        grid = ps_grid,
        back_btn = ps_back_btn,
        ax_cl = ax_ps_cl,
        ax_dl = ax_ps_dl,
        cl_x = ps_cl_x,
        cl_y = ps_cl_y,
        dl_x = ps_dl_x,
        dl_y = ps_dl_y,
        title_cl = ps_title_cl,
        title_dl = ps_title_dl,
        status = ps_status,
    )
end

function _healpix_ps_positive_or_floor(ell, y)
    x, yy = _healpix_ps_positive_points(ell, y)
    isempty(x) || return x, yy
    return Float64[1.0], Float64[eps(Float64)]
end

function _healpix_render_power_spectrum_panel!(
    panel;
    pixels,
    fname::AbstractString,
    source_label::AbstractString = "",
    set_status!,
)
    try
        ps = healpix_power_spectrum(pixels)
        xcl, ycl = _healpix_ps_positive_or_floor(ps.ell, ps.cl)
        xdl, ydl = _healpix_ps_positive_or_floor(ps.ell, ps.dl)
        panel.cl_x[] = xcl; panel.cl_y[] = ycl
        panel.dl_x[] = xdl; panel.dl_y[] = ydl
        source_tex = isempty(source_label) ? "" :
            "}\\;\\text{" * latex_safe(source_label)
        panel.title_cl[] = latexstring("\\text{", latex_safe(fname), source_tex, "}\\;C_\\ell")
        panel.title_dl[] = latexstring("\\text{", latex_safe(fname), source_tex, "}\\;D_\\ell")
        panel.status[] = latexstring(
            "\\mathrm{nside}=", ps.nside,
            "\\quad \\ell_{\\max}=", ps.lmax,
            "\\quad f_{\\mathrm{sky}}=", string(round(ps.f_sky; digits = 4)),
            "\\quad \\text{method}=\\mathrm{anafast}"
        )
        autolimits!(panel.ax_cl)
        autolimits!(panel.ax_dl)
        set_status!("HEALPix power spectrum rendered in the main window.")
    catch e
        panel.status[] = latexstring("\\text{Failed to compute }C_\\ell\\text{: }", latex_safe(sprint(showerror, e)))
        set_status!("Failed to compute HEALPix power spectrum: $(sprint(showerror, e))")
    end
    nothing
end
