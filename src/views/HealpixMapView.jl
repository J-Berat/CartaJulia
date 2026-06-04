# path: src/views/HealpixMapView.jl
#
# HEALPix 1D map viewer (Mollweide projection). Body extracted verbatim from
# `manta_healpix(filepath; …)` in MANTAHealpix.jl; only the prologue changed
# so the function takes a `HealpixMapDataset` rather than reading FITS itself.

function _view_healpix_map(
    ds::HealpixMapDataset;
    cmap::Symbol = :inferno,
    vmin = nothing,
    vmax = nothing,
    invert::Bool = false,
    scale::Symbol = :lin,
    asinh_softening::Real = ASINH_SOFTENING_DEFAULT,
    nx::Int = 1400,
    ny::Int = 700,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    hist_mode::Symbol = :bars,
    hist_bins::Int = HIST_BINS_DEFAULT,
    hist_xlimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    hist_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
)
    m          = ds.map
    column     = ds.column
    unit_label = ds.unit_label
    fname      = ds.source_id
    @info "HEALPix map" source=fname nside=m.resolution.nside npix=length(m)
    unit_label_tex = latexstring("\\text{", latex_safe(unit_label), "}")

    # ---------- Reprojection (once, kept in memory) ----------
    # why: project once, then gather into both the displayed image and the
    # pixel-index grid used for region selection. Avoids running the
    # Mollweide trig + ang2pix once; the map-aware grid preserves RING/NESTED.
    ipix_grid = _cached_mollweide_pixel_index(m, nx, ny)
    img_raw   = mollweide_apply_index(ipix_grid, m)

    # ---------- State ----------
    cmap_name   = Observable(cmap)
    invert_cmap = Observable(invert)
    cm_obs = lift(cmap_name, invert_cmap) do name, inv
        base = to_cmap(name); inv ? reverse(base) : base
    end

    scale_mode = Observable(scale)
    gauss_on = Observable(false)
    sigma = Observable(1.5f0)
    img_proc = lift(gauss_on, sigma) do on, σ
        on ? nan_gaussian_filter(img_raw, σ) : img_raw
    end
    img_disp = lift(img_proc, scale_mode) do im, m_
        out = apply_scale(im, m_; asinh_softening = asinh_softening)
        # protect: NaN/Inf already turned to NaN by apply_scale in log modes
        out2 = similar(out, Float32)
        @inbounds for k in eachindex(out)
            x = out[k]
            out2[k] = isfinite(x) ? Float32(x) : Float32(NaN32)
        end
        out2
    end

    use_manual = Observable(false)
    clims_manual = Observable((0f0, 1f0))
    clims_auto = lift(img_disp) do im
        fin = filter(isfinite, im)
        isempty(fin) && return (0f0, 1f0)
        qlo = scale_mode[] === :lin ? 0.02 : 0.05
        (Float32(quantile(fin, qlo)), Float32(quantile(fin, 0.98)))
    end
    if vmin !== nothing && vmax !== nothing
        a, b = Float32(vmin), Float32(vmax)
        a == b && (a = prevfloat(a); b = nextfloat(b))
        clims_manual[] = (a, b); use_manual[] = true
    end
    clims_obs = lift(use_manual, clims_auto, clims_manual) do um, ca, cm
        um ? cm : ca
    end
    clims_safe = lift(clims_obs) do (lo, hi)
        if !(isfinite(lo) && isfinite(hi)) || lo == hi
            (0f0, 1f0)
        else
            (lo, hi)
        end
    end

    contour_auto_levels = lift(img_disp) do im
        automatic_contour_levels(im; n = CONTOUR_N_LEVELS_DEFAULT)
    end
    contour_use_manual = Observable(false)
    contour_manual_levels = Observable(Float32[])
    contour_manual_colors = Observable(String[])
    contour_levels_obs = lift(contour_use_manual, contour_manual_levels, contour_auto_levels) do use_man, manual, auto
        use_man && !isempty(manual) ? manual : auto
    end
    # Auto-contrasted contour colour: white on dark images, black on bright ones.
    contour_default_color_obs = lift(img_disp) do img
        fv = filter(isfinite, vec(Float32.(img)))
        isempty(fv) && return RGBAf(0f0, 0f0, 0f0, CONTOUR_AUTO_DARK_ALPHA)
        lo, hi = percentile_clims(fv, 5, 95)
        rng = hi - lo
        rng < 1f-9 && return RGBAf(0f0, 0f0, 0f0, CONTOUR_AUTO_DARK_ALPHA)
        t = clamp((median(fv) - lo) / rng, 0f0, 1f0)
        t > CONTOUR_AUTO_BRIGHTNESS_THRESHOLD ?
            RGBAf(0f0, 0f0, 0f0, CONTOUR_AUTO_DARK_ALPHA) :
            RGBAf(1f0, 1f0, 1f0, CONTOUR_AUTO_LIGHT_ALPHA)
    end
    contour_colors_obs = lift(contour_levels_obs, contour_use_manual, contour_manual_colors,
                              contour_default_color_obs) do levels, use_man, colors, def_color
        contour_color_values(use_man ? colors : String[], length(levels), def_color)
    end
    show_contours = Observable(false)

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

    zoom_drag_active = Observable(false)
    zoom_drag_start  = Observable(Point2f(NaN32, NaN32))
    zoom_drag_end    = Observable(Point2f(NaN32, NaN32))
    show_graticule   = Observable(true)
    focus_image      = Observable(false)
    selection_mode = Observable(:point)
    region_shape = Observable(:box)
    region_drag_active = Observable(false)
    region_start = Observable(Point2f(NaN32, NaN32))
    region_end = Observable(Point2f(NaN32, NaN32))
    region_ipix = Observable(Int[])

    ui_theme = current_ui_theme()
    ui_accent = ui_theme.accent
    ui_selection = ui_theme.selection
    ui_text_muted = ui_theme.text_muted

    # ---------- Figure ----------
    pick_backend!(activate_gl)
    fig_size = _pick_fig_size(figsize)
    if figsize === nothing
        fig_size = (min(fig_size[1], 1500), 900)
    end
    # Responsive layout: same thresholds as CubeView for consistency when
    # switching between views. Without this, Fixed heights push the info bar
    # / histogram / controls off screen on small window sizes (≤ 1500×950).
    compact_layout  = fig_size[1] <= COMPACT_LAYOUT_W || fig_size[2] <= COMPACT_LAYOUT_H
    tight_layout    = fig_size[1] <= 1280 || fig_size[2] <= 760
    cbar_h_px       = tight_layout ? 30 : compact_layout ? 38  : 52
    info_h_px       = tight_layout ? 18 : compact_layout ? 22  : 30
    hist_h_px       = tight_layout ? 45  : compact_layout ? 70  : 95
    map_h_px        = tight_layout ? 160 : compact_layout ? 220 : 330
    map_w_px        = tight_layout ? 320 : compact_layout ? 440 : 660
    font_sz         = tight_layout ? 12 : compact_layout ? 13  : 15
    fig = Figure(size = fig_size, backgroundcolor = ui_theme.background)

    main_grid = fig[1, 1] = GridLayout()
    colsize!(fig.layout, 1, Relative(1))
    colsize!(main_grid, 1, Relative(1))
    colgap!(main_grid, -8)
    rowgap!(main_grid, -8)

    ax_img = Axis(
        main_grid[1, 1];
        title  = make_main_title(fname),
        aspect = DataAspect(),
        xticksvisible = false, yticksvisible = false,
        xticklabelsvisible = false, yticklabelsvisible = false,
        bottomspinevisible = false, topspinevisible = false,
        leftspinevisible   = false, rightspinevisible = false,
        width = map_w_px,
        height = map_h_px,
        tellwidth = false,
        tellheight = false,
        halign = :center,
    )
    xs = LinRange(-2f0, 2f0, nx)
    ys = LinRange(-1f0, 1f0, ny)
    img_for_plot = lift(img_disp) do im
        permutedims(im)  # (nx, ny) layout pour heatmap(xs, ys, A)
    end
    hm = heatmap!(ax_img, xs, ys, img_for_plot;
                  colormap=cm_obs, colorrange=clims_safe, nan_color=:white)
    contour!(ax_img, xs, ys, img_for_plot;
             levels=contour_levels_obs, color=contour_colors_obs, linewidth=CONTOUR_LW_HP,
             visible=show_contours)
    full_map_bounds = MOLLWEIDE_BOUNDS
    set_mollweide_view!(ax_img, full_map_bounds...)
    graticule = draw_mollweide_graticule!(ax_img)
    refresh_graticule_labels!(graticule, ax_img; bounds=full_map_bounds)

    # Mollweide ellipse border (purely cosmetic)
    ell_x = [2cos(t) for t in LinRange(0, 2π, 200)]
    ell_y = [sin(t)  for t in LinRange(0, 2π, 200)]
    lines!(ax_img, ell_x, ell_y; color=:black, linewidth=0.8)

    # zoom rectangle
    zoom_box_segments = lift(zoom_drag_active, zoom_drag_start, zoom_drag_end) do active, p0, p1
        active || return Point2f[]
        if !(isfinite(p0[1]) && isfinite(p0[2]) && isfinite(p1[1]) && isfinite(p1[2]))
            return Point2f[]
        end
        x0, y0 = p0; x1, y1 = p1
        Point2f[
            Point2f(x0,y0), Point2f(x1,y0),
            Point2f(x1,y0), Point2f(x1,y1),
            Point2f(x1,y1), Point2f(x0,y1),
            Point2f(x0,y1), Point2f(x0,y0),
        ]
    end
    # Corner accents for the zoom rectangle.
    zoom_corner_segments = lift(zoom_drag_active, zoom_drag_start, zoom_drag_end) do active, p0, p1
        active || return Point2f[]
        if !(isfinite(p0[1]) && isfinite(p0[2]) && isfinite(p1[1]) && isfinite(p1[2]))
            return Point2f[]
        end
        x0, y0 = Float32(p0[1]), Float32(p0[2])
        x1, y1 = Float32(p1[1]), Float32(p1[2])
        cx = sign(x1 - x0) * abs(x1 - x0) * ZOOM_BEZIER_FACTOR
        cy = sign(y1 - y0) * abs(y1 - y0) * ZOOM_BEZIER_FACTOR
        Point2f[
            Point2f(x0, y0), Point2f(x0 + cx, y0),
            Point2f(x0, y0), Point2f(x0, y0 + cy),
            Point2f(x1, y0), Point2f(x1 - cx, y0),
            Point2f(x1, y0), Point2f(x1, y0 + cy),
            Point2f(x1, y1), Point2f(x1 - cx, y1),
            Point2f(x1, y1), Point2f(x1, y1 - cy),
            Point2f(x0, y1), Point2f(x0 + cx, y1),
            Point2f(x0, y1), Point2f(x0, y1 - cy),
        ]
    end
    linesegments!(ax_img, zoom_box_segments;    color=(ui_selection, ZOOM_BOX_ALPHA),    linewidth=ZOOM_BOX_LW,    linestyle=:dash)
    linesegments!(ax_img, zoom_corner_segments; color=(ui_selection, ZOOM_CORNER_ALPHA), linewidth=ZOOM_CORNER_LW, linestyle=:solid)
    region_segments = lift(region_start, region_end, region_shape, region_ipix, region_drag_active) do p0, p1, shape, ipixs, dragging
        (dragging || !isempty(ipixs)) ? projected_region_segments(p0, p1, shape) : Point2f[]
    end
    lines!(ax_img, region_segments; color=(ui_selection, REGION_ALPHA), linewidth=REGION_LW_HP)

    map_colorbar = Colorbar(
        main_grid[2, 1],
        hm;
        label = lift(m_ -> scale_label_tex(m_, unit_label), scale_mode),
        vertical = false,
        height = 18,
        width = map_w_px,
        tellwidth = false,
        tellheight = false,
        halign = :center,
    )
    rowsize!(main_grid, 1, Fixed(map_h_px))
    rowsize!(main_grid, 2, Fixed(cbar_h_px))

    # Bandeau info
    info_obs = Observable(latexstring("\\text{move cursor over the map}"))
    Label(main_grid[3, 1], info_obs; halign=:left, fontsize=font_sz, tellheight=false)
    rowsize!(main_grid, 3, Fixed(info_h_px))

    # Controls
    ax_hist = Axis(
        main_grid[4, 1];
        title = L"\text{Visible map histogram}",
        xlabel = unit_label_tex,
        ylabel = hist_ylabel_obs,
        # height is governed by `rowsize!(main_grid, 4, ...)` below — no
        # hard-coded value here (cf. CLAUDE.md / anti-patterns).
        xtickformat = _latex_tick_formatter,
        ytickformat = _latex_tick_formatter,
        width = map_w_px,
        tellwidth = false,
        tellheight = false,
        halign = :center,
    )
    barplot!(ax_hist, hist_x_obs, hist_y_obs; width=hist_width_obs, color=(ui_accent, HIST_BAR_ALPHA), strokecolor=ui_accent, strokewidth=HIST_BAR_STROKE_LW, visible=hist_bars_visible)
    lines!(ax_hist, hist_x_obs, hist_y_obs; color=ui_accent, linewidth=HIST_KDE_LW, visible=hist_kde_visible)
    vlines!(ax_hist, lift(lim -> [first(lim), last(lim)], clims_safe);
            color=(ui_text_muted, HIST_LIMITS_ALPHA), linewidth=HIST_LIMITS_LW_HP, linestyle=:dash)
    rowsize!(main_grid, 4, Fixed(hist_h_px))

    # ---------- Controls (card-based modal layout, mirrors CubeView) ----------
    ctrl_row_h   = tight_layout ? (138, 98, 30) : compact_layout ? (168, 130, 36) : (188, 150, 42)
    ctrl_gap     = tight_layout ? 5 : compact_layout ? 6 : 10
    card_pad     = tight_layout ? 7 : compact_layout ? 9 : 12
    card_gap     = tight_layout ? 5 : compact_layout ? 7 : 10
    ctrl_total_h = sum(ctrl_row_h) + 2 * ctrl_gap

    ctrl_grid = main_grid[5, 1] = GridLayout(; alignmode = Outside())
    rowsize!(main_grid, 5, Fixed(ctrl_total_h))
    colgap!(ctrl_grid, ctrl_gap)
    rowgap!(ctrl_grid, ctrl_gap)

    # -- block/layout visibility helpers (verbatim from CubeView) --
    set_block_visible!(block, visible::Bool) = begin
        try; block.visible[] = visible;            catch; end
        try; block.scene.visible[] = visible;      catch; end
        try; block.blockscene.visible[] = visible; catch; end
        nothing
    end
    function set_layout_contents_visible!(layout, visible::Bool)
        for block in try; contents(layout); catch; Any[]; end
            set_block_visible!(block, visible)
            block isa GridLayout && set_layout_contents_visible!(block, visible)
        end
        nothing
    end

    focus_bar = GridLayout(
        main_grid[2, 1];
        alignmode = Outside(0),
        halign = :center,
        valign = :center,
        tellwidth = false,
        tellheight = false,
    )
    colgap!(focus_bar, tight_layout ? 4 : 6)
    Box(focus_bar[1, 1:5];
        color = RGBAf(ui_theme.panel.r, ui_theme.panel.g, ui_theme.panel.b, 0.94),
        strokecolor = RGBAf(ui_theme.border.r, ui_theme.border.g, ui_theme.border.b, 0.75),
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

    # -- card factory (mirrors CubeView card styling) --
    function control_card!(parent, row, col, title::AbstractString; rows::Int = 4, cols::Int = 4)
        card = parent[row, col] = GridLayout(;
            alignmode = Outside(card_pad), tellwidth = false, tellheight = false)
        body_rows = rows + 1
        card_is_dark = ui_theme.background.r < 0.5
        card_border = RGBAf(ui_theme.border.r, ui_theme.border.g, ui_theme.border.b,
                            card_is_dark ? 0.58 : 0.82)
        header_divider = RGBAf(ui_theme.border.r, ui_theme.border.g, ui_theme.border.b,
                               card_is_dark ? 0.30 : 0.50)
        Box(card[1:body_rows, 1:cols]; color = ui_theme.panel, strokecolor = card_border,
            strokewidth = card_is_dark ? 0.8 : 0.9, cornerradius = 8, z = -6)
        Box(card[1, 1:cols]; color = ui_theme.panel_header, strokecolor = header_divider,
            strokewidth = 0.8, cornerradius = 8, z = -5)
        Label(card[1, 1:cols]; text = title, halign = :left, tellwidth = false,
            fontsize = 12, color = ui_theme.text,
            padding = (10, 10, 5, 5))
        Box(card[body_rows, 1:cols]; color = :transparent, strokewidth = 0, z = -7)
        rowsize!(card, 1, Fixed(compact_layout ? 28 : 32))
        rowsize!(card, body_rows, Fixed(compact_layout ? 10 : 12))
        rowgap!(card, tight_layout ? 5 : compact_layout ? 7 : 9)
        colgap!(card, card_gap)
        return card
    end
    ctrl_lbl!(layout, pos, txt) = Label(layout[pos...]; text = txt, halign = :left,
        tellwidth = false, fontsize = 13, color = ui_theme.text_muted)

    # ── Mode bar (full width) ─────────────────────────────────────────────
    mode_bar = ctrl_grid[3, 1:3] = GridLayout(; alignmode = Outside(0), halign = :center)
    colgap!(mode_bar, compact_layout ? 8 : 12)
    tab_h = tight_layout ? 28 : 32
    mode_segment = mode_bar[1, 1] = GridLayout(; alignmode = Outside(0))
    colgap!(mode_segment, 0)
    mode_nav_btn      = Button(mode_segment[1, 1]; label = "$(MANTA_ICONS.nav) Navigation", width = tight_layout ? 136 : 148, height = tab_h)
    mode_analysis_btn = Button(mode_segment[1, 2]; label = "$(MANTA_ICONS.analysis) Analysis", width = tight_layout ? 122 : 132, height = tab_h)
    mode_export_btn   = Button(mode_segment[1, 3]; label = "$(MANTA_ICONS.export_icon) Export", width = tight_layout ? 106 : 114, height = tab_h)
    foreach(c -> colsize!(mode_segment, c, Auto()), 1:3)
    help_btn          = Button(mode_bar[1, 2]; label = MANTA_ICONS.help, width = tight_layout ? 42 : 46, height = tab_h)
    focus_btn         = Button(mode_bar[1, 3]; label = "Focus", width = tight_layout ? 68 : 76, height = tab_h)
    btn_undo          = Button(mode_bar[1, 4]; label = MANTA_ICONS.undo, width = tight_layout ? 42 : 46, height = tab_h)
    btn_redo          = Button(mode_bar[1, 5]; label = MANTA_ICONS.redo, width = tight_layout ? 42 : 46, height = tab_h)
    foreach(c -> colsize!(mode_bar, c, Auto()), 1:5)
    control_mode = Observable(:navigation)

    # ── NAVIGATION: Display card (row 1 col 1) ────────────────────────────
    display_card = control_card!(ctrl_grid, 1, 1, "Display"; rows = 4, cols = 5)
    ctrl_lbl!(display_card, (2, 1), "Scale")
    scale_menu = Menu(display_card[2, 2:3]; options = scale_menu_options(),
                      prompt = String(scale), width = compact_layout ? 88 : 102)
    invert_chk = Checkbox(display_card[2, 4])
    Label(display_card[2, 5]; text = "Invert", halign = :left, tellwidth = false,
          fontsize = 13, color = ui_theme.text)
    invert_chk.checked[] = invert_cmap[]
    ctrl_lbl!(display_card, (3, 1), "Colormap")
    cmap_menu = colormap_selector!(display_card[3, 2:5];
                     cmap = cmap, width = compact_layout ? 144 : 164,
                     compact = compact_layout, theme = ui_theme)
    gauss_chk = Checkbox(display_card[4, 1])
    Label(display_card[4, 2]; text = "Smoothing", halign = :left, tellwidth = false,
          fontsize = 13, color = ui_theme.text)
    sigma_label = Label(display_card[4, 3];
        text = latexstring("\\sigma = 1.5\\,\\text{px}"),
        fontsize = 13, halign = :left, tellwidth = false, color = ui_theme.text)
    sigma_slider = Slider(display_card[4, 4:5];
        range = LinRange(0, 10, 101), startvalue = 1.5,
        width = compact_layout ? 128 : 158, height = 24)
    foreach(c -> colsize!(display_card, c, Auto()), 1:5)

    # ── NAVIGATION: View card (row 1 col 2) ───────────────────────────────
    nav_view_card = control_card!(ctrl_grid, 1, 2, "View"; rows = 3, cols = 4)
    graticule_chk = Checkbox(nav_view_card[2, 1])
    Label(nav_view_card[2, 2]; text = "Graticule", halign = :left, tellwidth = false,
          fontsize = 13, color = ui_theme.text)
    graticule_chk.checked[] = show_graticule[]
    reset_zoom_btn = Button(nav_view_card[3, 1:2]; label = "$(MANTA_ICONS.fit) Fit", width = 88, height = 32)
    foreach(c -> colsize!(nav_view_card, c, Auto()), 1:4)

    # ── ANALYSIS: Contrast card (row 1 col 1) ────────────────────────────
    contrast_card = control_card!(ctrl_grid, 1, 1, "$(MANTA_ICONS.contrast) Contrast"; rows = 4, cols = 5)
    clim_min_box  = Textbox(contrast_card[2, 1]; placeholder = "min", width = 110, height = 32)
    clim_max_box  = Textbox(contrast_card[2, 2]; placeholder = "max", width = 110, height = 32)
    apply_btn     = Button(contrast_card[2, 3]; label = "Apply",  width = 78, height = 32)
    auto_btn      = Button(contrast_card[2, 4]; label = "Auto",   width = 72, height = 32)
    p1_btn        = Button(contrast_card[3, 1]; label = "p1-p99", width = 86, height = 32)
    p5_btn        = Button(contrast_card[3, 2]; label = "p5-p95", width = 86, height = 32)
    foreach(c -> colsize!(contrast_card, c, Auto()), 1:5)

    # ── ANALYSIS: Selection card (row 1 col 2) ───────────────────────────
    selection_card = control_card!(ctrl_grid, 1, 2, "$(MANTA_ICONS.selection) Selection"; rows = 3, cols = 4)
    region_mode_menu = Menu(selection_card[2, 1]; options = ["point", "box", "circle"],
                            prompt = "point", width = compact_layout ? 100 : 116)
    region_clear_btn = Button(selection_card[2, 2]; label = "Clear",
                              width = compact_layout ? 84 : 96, height = 32)
    region_count_label = Label(selection_card[2, 3:4]; text = "0 pix", halign = :left,
                               tellwidth = false, fontsize = 13, color = ui_theme.text_muted)
    foreach(c -> colsize!(selection_card, c, Auto()), 1:4)

    # ── ANALYSIS: Contours card (row 1 col 3) ────────────────────────────
    contour_card = control_card!(ctrl_grid, 1, 3, "Contours"; rows = 3, cols = 5)
    contour_chk = Checkbox(contour_card[2, 1])
    Label(contour_card[2, 2]; text = "Show", halign = :left, tellwidth = false,
          fontsize = 13, color = ui_theme.text)
    contour_levels_box = Textbox(contour_card[2, 3:4];
        placeholder = "auto or 1:red, 2:#00ffaa",
        width = compact_layout ? 174 : 202, height = 32)
    contour_apply_btn = Button(contour_card[2, 5]; label = "Apply", width = 74, height = 32)
    contour_chk.checked[] = show_contours[]
    foreach(c -> colsize!(contour_card, c, Auto()), 1:5)

    # ── ANALYSIS bottom: Histogram card (row 2, centred) ─────────────────
    analysis_bottom = ctrl_grid[2, 1:3] = GridLayout(; alignmode = Outside(0))
    colgap!(analysis_bottom, ctrl_gap)
    hist_card = control_card!(analysis_bottom, 1, 2, "Histogram"; rows = 3, cols = 6)
    hist_mode_menu   = Menu(hist_card[2, 1]; options = ["bars", "kde"],
                            prompt = String(hist_mode_obs[]), width = compact_layout ? 88 : 98)
    hist_bins_box    = Textbox(hist_card[2, 2]; placeholder = "bins",  width = 68, height = 32)
    hist_xmin_box    = Textbox(hist_card[2, 3]; placeholder = "x min", width = 84, height = 32)
    hist_xmax_box    = Textbox(hist_card[2, 4]; placeholder = "x max", width = 84, height = 32)
    hist_apply_btn   = Button(hist_card[2, 5]; label = "Apply x", width = 74, height = 32)
    hist_auto_btn    = Button(hist_card[2, 6]; label = "Auto x",  width = 74, height = 32)
    hist_ymin_box    = Textbox(hist_card[3, 3]; placeholder = "y min", width = 84, height = 32)
    hist_ymax_box    = Textbox(hist_card[3, 4]; placeholder = "y max", width = 84, height = 32)
    hist_y_apply_btn = Button(hist_card[3, 5]; label = "Apply y", width = 74, height = 32)
    hist_y_auto_btn  = Button(hist_card[3, 6]; label = "Auto y",  width = 74, height = 32)
    foreach(c -> colsize!(hist_card, c, Auto()), 1:6)
    Box(analysis_bottom[1, 1]; color = :transparent, strokewidth = 0)
    Box(analysis_bottom[1, 3]; color = :transparent, strokewidth = 0)
    colsize!(analysis_bottom, 1, Relative(1))
    colsize!(analysis_bottom, 2, Fixed(compact_layout ? 480 : 560))
    colsize!(analysis_bottom, 3, Relative(1))

    # ── EXPORT: Output card (row 1 col 1) ────────────────────────────────
    output_card = control_card!(ctrl_grid, 1, 1, "Output"; rows = 3, cols = 3)
    save_btn = Button(output_card[2, 1]; label = "$(MANTA_ICONS.export_icon) PNG", width = 90, height = 32)
    foreach(c -> colsize!(output_card, c, Auto()), 1:3)

    # ── Grid sizing ───────────────────────────────────────────────────────
    foreach(c -> colsize!(ctrl_grid, c, Relative(1 / 3)), 1:3)
    rowsize!(ctrl_grid, 1, Fixed(ctrl_row_h[1]))
    rowsize!(ctrl_grid, 2, Fixed(0))
    rowsize!(ctrl_grid, 3, Fixed(ctrl_row_h[3]))

    # ── Style ─────────────────────────────────────────────────────────────
    foreach(w -> manta_style_menu!(w, ui_theme),
            (scale_menu, cmap_menu, region_mode_menu, hist_mode_menu))
    # Mode navigation segmented control (state managed by set_mode_button_active!)
    foreach(w -> manta_style_segmented_button!(w, ui_theme),
            (mode_nav_btn, mode_analysis_btn, mode_export_btn))
    foreach(w -> manta_style_button!(w, ui_theme), (p1_btn, p5_btn))
    # Primary actions (accent)
    foreach(w -> manta_style_button_primary!(w, ui_theme),
            (apply_btn, save_btn, contour_apply_btn,
             hist_apply_btn, hist_y_apply_btn))
    # Resets, Clear, and utilities (unobtrusive)
    foreach(w -> manta_style_button_ghost!(w, ui_theme),
            (help_btn, focus_btn, btn_undo, btn_redo, auto_btn, reset_zoom_btn,
             region_clear_btn, hist_auto_btn, hist_y_auto_btn,
             focus_exit_btn, focus_fit_btn, focus_auto_btn, focus_help_btn))
    foreach(w -> manta_style_checkbox!(w, ui_theme),
            (invert_chk, graticule_chk, gauss_chk, contour_chk))
    foreach(w -> manta_style_textbox!(w, ui_theme),
            (clim_min_box, clim_max_box, contour_levels_box,
             hist_bins_box, hist_xmin_box, hist_xmax_box,
             hist_ymin_box, hist_ymax_box))
    manta_style_slider!(sigma_slider, ui_theme)

    # ── Mode switching ────────────────────────────────────────────────────
    nav_cards_hp      = (display_card, nav_view_card)
    analysis_cards_hp = (contrast_card, selection_card, contour_card, hist_card)
    export_cards_hp   = (output_card,)
    set_layout_contents_visible!(focus_bar, false)

    function set_mode_button_active!(btn, active::Bool)
        manta_style_segmented_button!(btn, ui_theme; active = active)
        nothing
    end
    function refresh_control_mode!()
        if focus_image[]
            set_layout_contents_visible!(ctrl_grid, false)
            set_layout_contents_visible!(focus_bar, true)
            return nothing
        end
        mode = control_mode[]
        set_layout_contents_visible!(ctrl_grid, true)
        set_layout_contents_visible!(focus_bar, false)
        for card in nav_cards_hp;      set_layout_contents_visible!(card, mode === :navigation); end
        for card in analysis_cards_hp; set_layout_contents_visible!(card, mode === :analysis);   end
        for card in export_cards_hp;   set_layout_contents_visible!(card, mode === :export);     end
        if mode === :navigation
            rowsize!(ctrl_grid, 1, Fixed(ctrl_row_h[1]))
            rowsize!(ctrl_grid, 2, Fixed(0))
        elseif mode === :analysis
            rowsize!(ctrl_grid, 1, Fixed(ctrl_row_h[1]))
            rowsize!(ctrl_grid, 2, Fixed(ctrl_row_h[2]))
        else
            rowsize!(ctrl_grid, 1, Fixed(tight_layout ? 90 : 105))
            rowsize!(ctrl_grid, 2, Fixed(0))
        end
        rowsize!(ctrl_grid, 3, Fixed(ctrl_row_h[3]))
        set_mode_button_active!(mode_nav_btn,      mode === :navigation)
        set_mode_button_active!(mode_analysis_btn, mode === :analysis)
        set_mode_button_active!(mode_export_btn,   mode === :export)
        nothing
    end
    refresh_control_mode!()

    on(mode_nav_btn.clicks)      do _; control_mode[] = :navigation; refresh_control_mode!(); end
    on(mode_analysis_btn.clicks) do _; control_mode[] = :analysis;   refresh_control_mode!(); end
    on(mode_export_btn.clicks)   do _; control_mode[] = :export;     refresh_control_mode!(); end

    if use_manual[]
        a, b = clims_manual[]
        s_a = string(a); s_b = string(b)
        clim_min_box.displayed_string[] = s_a; clim_min_box.stored_string[] = s_a
        clim_max_box.displayed_string[] = s_b; clim_max_box.stored_string[] = s_b
    end

    set_box_text!(tb, s::AbstractString) = begin
        str = String(s)
        tb.displayed_string[] = str
        tb.stored_string[] = str
        nothing
    end
    set_box_text!(hist_bins_box, string(hist_bins_obs[]))
    if hist_xlimits_manual[]
        lo, hi = hist_xlimits_manual_value[]
        set_box_text!(hist_xmin_box, string(lo))
        set_box_text!(hist_xmax_box, string(hi))
    end
    if hist_ylimits_manual[]
        lo, hi = hist_ylimits_manual_value[]
        set_box_text!(hist_ymin_box, string(lo))
        set_box_text!(hist_ymax_box, string(hi))
    end
    _hp_status!(msg::AbstractString) =
        (info_obs[] = latexstring("\\text{", latex_safe(msg), "}"); nothing)
    _hp_snapshot() = (;
        scale_mode = scale_mode[],
        cmap_name = cmap_name[],
        invert_cmap = invert_cmap[],
        gauss_on = gauss_on[],
        sigma = sigma[],
        show_graticule = show_graticule[],
        show_contours = show_contours[],
        use_manual = use_manual[],
        clims_manual = clims_manual[],
        hist_mode = hist_mode_obs[],
        hist_bins = hist_bins_obs[],
        hist_xmanual = hist_xlimits_manual[],
        hist_xlimits = hist_xlimits_manual_value[],
        hist_ymanual = hist_ylimits_manual[],
        hist_ylimits = hist_ylimits_manual_value[],
    )
    _hp_undo_stack = UndoRedoStack(_hp_snapshot(); capacity = UNDO_STACK_CAPACITY)
    for _obs in (
        scale_mode, cmap_name, invert_cmap, gauss_on, sigma, show_graticule,
        show_contours, use_manual, clims_manual, hist_mode_obs, hist_bins_obs,
        hist_xlimits_manual, hist_xlimits_manual_value,
        hist_ylimits_manual, hist_ylimits_manual_value,
    )
        on(_obs) do _
            _hp_undo_stack.suppress && return
            register_state!(_hp_undo_stack, _hp_snapshot())
        end
    end
    on(_hp_undo_stack.can_undo; update = true) do can
        btn_undo.labelcolor[] = can ? ui_theme.text : ui_theme.text_muted
        btn_undo.labelcolor_hover[] = can ? ui_theme.accent_strong : ui_theme.text_muted
    end
    on(_hp_undo_stack.can_redo; update = true) do can
        btn_redo.labelcolor[] = can ? ui_theme.text : ui_theme.text_muted
        btn_redo.labelcolor_hover[] = can ? ui_theme.accent_strong : ui_theme.text_muted
    end
    function _apply_hp_snap!(snap)
        snap === nothing && return
        with_suppression(_hp_undo_stack) do
            scale_menu.selection[] = String(snap.scale_mode)
            if String(snap.cmap_name) in MANTA_COLORMAP_OPTIONS
                cmap_menu.selection[] = String(snap.cmap_name)
            else
                cmap_name[] = snap.cmap_name
            end
            invert_chk.checked[] = snap.invert_cmap
            gauss_chk.checked[] = snap.gauss_on
            sigma_slider.value[] = Float32(snap.sigma)
            graticule_chk.checked[] = snap.show_graticule
            contour_chk.checked[] = snap.show_contours
            hist_mode_menu.selection[] = String(snap.hist_mode)
            hist_bins_obs[] = snap.hist_bins
            hist_xlimits_manual_value[] = snap.hist_xlimits
            hist_xlimits_manual[] = snap.hist_xmanual
            hist_ylimits_manual_value[] = snap.hist_ylimits
            hist_ylimits_manual[] = snap.hist_ymanual
            set_box_text!(hist_bins_box, string(snap.hist_bins))
            set_box_text!(hist_xmin_box, snap.hist_xmanual ? string(first(snap.hist_xlimits)) : "")
            set_box_text!(hist_xmax_box, snap.hist_xmanual ? string(last(snap.hist_xlimits)) : "")
            set_box_text!(hist_ymin_box, snap.hist_ymanual ? string(first(snap.hist_ylimits)) : "")
            set_box_text!(hist_ymax_box, snap.hist_ymanual ? string(last(snap.hist_ylimits)) : "")
            if snap.use_manual
                clims_manual[] = snap.clims_manual
                use_manual[] = true
                set_box_text!(clim_min_box, string(first(snap.clims_manual)))
                set_box_text!(clim_max_box, string(last(snap.clims_manual)))
            else
                use_manual[] = false
                set_box_text!(clim_min_box, "")
                set_box_text!(clim_max_box, "")
            end
            refresh_hist_axes!()
        end
        nothing
    end
    on(btn_undo.clicks) do _
        snap = undo!(_hp_undo_stack)
        snap === nothing && (_hp_status!("Nothing to undo."); return)
        _apply_hp_snap!(snap)
        _hp_status!("Undo.")
    end
    on(btn_redo.clicks) do _
        snap = redo!(_hp_undo_stack)
        snap === nothing && (_hp_status!("Nothing to redo."); return)
        _apply_hp_snap!(snap)
        _hp_status!("Redo.")
    end
    function clear_region!()
        region_ipix[] = Int[]
        region_start[] = Point2f(NaN32, NaN32)
        region_end[] = Point2f(NaN32, NaN32)
        region_drag_active[] = false
        region_count_label.text[] = "0 pix"
        nothing
    end
    function apply_region!(p0::Point2f, p1::Point2f)
        ips = projected_region_ipix(ipix_grid, p0[1], p0[2], p1[1], p1[2], region_shape[])
        region_ipix[] = ips
        region_count_label.text[] = "$(length(ips)) pix"
        mean_val = healpix_region_mean(m.pixels, ips)
        valstr = isfinite(mean_val) ? string(round(mean_val; digits=4)) : "NaN"
        shape = region_shape[] === :circle ? "circle" : "box"
        info_obs[] = latexstring(
            "\\mathrm{region}\\,\\text{", shape, "}\\;N=\\mathbf{", length(ips), "}",
            "\\;\\mathrm{mean}=\\mathbf{", valstr, "}",
            "\\;\\mathrm{", latex_safe(unit_label), "}"
        )
        nothing
    end
    function apply_percentile_clims!(lo::Real, hi::Real)
        clims = percentile_clims(img_disp[], lo, hi)
        clims_manual[] = clims
        use_manual[] = true
        set_box_text!(clim_min_box, string(first(clims)))
        set_box_text!(clim_max_box, string(last(clims)))
        nothing
    end

    focus_map_w = max(map_w_px, min(fig_size[1] - 48, 2 * (fig_size[2] - 72)))
    focus_map_h = max(map_h_px, focus_map_w ÷ 2)
    function apply_focus_image_mode!()
        if focus_image[]
            set_layout_contents_visible!(ctrl_grid, false)
            set_layout_contents_visible!(focus_bar, true)
            rowsize!(main_grid, 1, Relative(1))
            for r in 2:5
                rowsize!(main_grid, r, Fixed(0))
            end
            rowsize!(main_grid, 2, Fixed(tight_layout ? 34 : 38))
            rowgap!(main_grid, 0)
            ax_img.width[] = focus_map_w
            ax_img.height[] = focus_map_h
            set_block_visible!(map_colorbar, false)
            set_block_visible!(ax_hist, false)
            info_obs[] = latexstring("\\text{Focus image: controls hidden. Press f or Exit to return.}")
        else
            set_layout_contents_visible!(focus_bar, false)
            rowgap!(main_grid, -8)
            rowsize!(main_grid, 1, Fixed(map_h_px))
            rowsize!(main_grid, 2, Fixed(cbar_h_px))
            rowsize!(main_grid, 3, Fixed(info_h_px))
            rowsize!(main_grid, 4, Fixed(hist_h_px))
            rowsize!(main_grid, 5, Fixed(ctrl_total_h))
            ax_img.width[] = map_w_px
            ax_img.height[] = map_h_px
            set_block_visible!(map_colorbar, true)
            set_block_visible!(ax_hist, true)
            refresh_control_mode!()
            info_obs[] = latexstring("\\text{Focus image closed.}")
        end
        nothing
    end

    on(focus_image) do _
        apply_focus_image_mode!()
    end

    on(focus_btn.clicks) do _
        focus_image[] = true
    end

    on(focus_exit_btn.clicks) do _
        focus_image[] = false
    end

    toggle_focus_image!() = begin
        focus_image[] = !focus_image[]
        nothing
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

    # ---------- Reactivity ----------
    on(scale_menu.selection) do sel
        sel === nothing && return
        scale_mode[] = Symbol(sel)
    end
    on(cmap_menu.selection) do sel
        sel === nothing && return
        cmap_name[] = Symbol(sel)
    end
    on(invert_chk.checked) do v; invert_cmap[] = v; end
    on(gauss_chk.checked) do v
        gauss_on[] = v
    end
    on(sigma_slider.value) do v
        sigma[] = Float32(v)
        sigma_label.text[] = latexstring("\\sigma = $(round(v; digits=2))\\,\\text{px}")
    end
    on(graticule_chk.checked) do v
        show_graticule[] = v
        set_graticule_visible!(graticule, v)
    end
    function reset_hp_zoom!()
        set_mollweide_view!(ax_img, full_map_bounds...)
        refresh_graticule_labels!(graticule, ax_img; bounds=full_map_bounds)
        nothing
    end
    on(reset_zoom_btn.clicks) do _
        reset_hp_zoom!()
    end
    on(focus_fit_btn.clicks) do _
        focus_image[] || return
        reset_hp_zoom!()
        info_obs[] = latexstring("\\text{Focus image: view fitted.}")
    end
    on(focus_auto_btn.clicks) do _
        focus_image[] || return
        use_manual[] = false
        set_box_text!(clim_min_box, "")
        set_box_text!(clim_max_box, "")
    end
    on(focus_help_btn.clicks) do _
        help_btn.clicks[] = help_btn.clicks[] + 1
    end
    on(apply_btn.clicks) do _
        ok, manual, clims, _msg = parse_manual_clims(
            get_box_str(clim_min_box), get_box_str(clim_max_box);
            fallback = clims_manual[])
        ok || return
        if manual
            clims_manual[] = clims; use_manual[] = true
        else
            use_manual[] = false
        end
    end
    on(auto_btn.clicks) do _
        use_manual[] = false
        set_box_text!(clim_min_box, "")
        set_box_text!(clim_max_box, "")
    end
    on(p1_btn.clicks) do _; apply_percentile_clims!(1, 99); end
    on(p5_btn.clicks) do _; apply_percentile_clims!(5, 95); end
    on(hist_mode_menu.selection) do sel
        sel === nothing && return
        hist_mode_obs[] = normalize_histogram_mode(sel)
    end
    on(hist_apply_btn.clicks) do _
        ok_bins, bins, _bins_msg = parse_histogram_bins(get_box_str(hist_bins_box); fallback = hist_bins_obs[])
        ok_x, manual_x, xlim, _x_msg = parse_histogram_xlimits(
            get_box_str(hist_xmin_box),
            get_box_str(hist_xmax_box);
            fallback = hist_xlimits_manual_value[],
        )
        ok_bins && ok_x || return
        hist_bins_obs[] = bins
        hist_xlimits_manual_value[] = xlim
        hist_xlimits_manual[] = manual_x
        set_box_text!(hist_bins_box, string(bins))
        set_box_text!(hist_xmin_box, manual_x ? string(first(xlim)) : "")
        set_box_text!(hist_xmax_box, manual_x ? string(last(xlim)) : "")
        refresh_hist_axes!()
    end
    on(hist_auto_btn.clicks) do _
        hist_xlimits_manual[] = false
        set_box_text!(hist_xmin_box, "")
        set_box_text!(hist_xmax_box, "")
        refresh_hist_axes!()
    end
    on(hist_y_auto_btn.clicks) do _
        hist_ylimits_manual[] = false
        set_box_text!(hist_ymin_box, "")
        set_box_text!(hist_ymax_box, "")
        refresh_hist_axes!()
    end
    on(hist_y_apply_btn.clicks) do _
        ok_y, manual_y, ylim, _y_msg = parse_histogram_ylimits(
            get_box_str(hist_ymin_box),
            get_box_str(hist_ymax_box);
            fallback = hist_ylimits_manual_value[],
        )
        ok_y || return
        hist_ylimits_manual_value[] = ylim
        hist_ylimits_manual[] = manual_y
        set_box_text!(hist_ymin_box, manual_y ? string(first(ylim)) : "")
        set_box_text!(hist_ymax_box, manual_y ? string(last(ylim)) : "")
        refresh_hist_axes!()
    end
    on(hist_limits_obs) do _
        refresh_hist_axes!()
    end
    on(hist_y_obs) do _
        refresh_hist_axes!()
    end
    on(region_mode_menu.selection) do sel
        sel === nothing && return
        mode = Symbol(String(sel))
        mode in (:point, :box, :circle) || return
        selection_mode[] = mode
        region_shape[] = mode === :circle ? :circle : :box
        mode === :point && clear_region!()
    end
    on(region_clear_btn.clicks) do _
        clear_region!()
        info_obs[] = latexstring("\\text{region cleared}")
    end
    on(contour_chk.checked) do v
        show_contours[] = v
    end
    on(contour_apply_btn.clicks) do _
        ok, use_man, levels, colors, _msg = parse_contour_specs(
            get_box_str(contour_levels_box);
            fallback_levels=contour_manual_levels[],
            fallback_colors=contour_manual_colors[],
        )
        ok || return
        contour_use_manual[] = use_man
        contour_manual_levels[] = levels
        contour_manual_colors[] = colors
        set_box_text!(contour_levels_box, use_man ? format_contour_specs(levels, colors) : "")
        show_contours[] = true
        contour_chk.checked[] = true
    end

    # zoom right-drag, identical to `manta`
    on(events(ax_img).mousebutton) do ev
        if ev.button == Mouse.right && ev.action == Mouse.press
            p = mouseposition(ax_img); any(isnan, p) && return
            zoom_drag_start[] = Point2f(p[1], p[2])
            zoom_drag_end[]   = Point2f(p[1], p[2])
            zoom_drag_active[] = true
        elseif ev.button == Mouse.right && ev.action == Mouse.release
            zoom_drag_active[] || return
            p = mouseposition(ax_img)
            !any(isnan, p) && (zoom_drag_end[] = Point2f(p[1], p[2]))
            p0 = zoom_drag_start[]; p1 = zoom_drag_end[]
            zoom_drag_active[] = false
            zoom_drag_start[] = Point2f(NaN32, NaN32)
            zoom_drag_end[]   = Point2f(NaN32, NaN32)
            (isfinite(p0[1]) && isfinite(p1[1])) || return
            xmin, xmax = minmax(p0[1], p1[1])
            ymin, ymax = minmax(p0[2], p1[2])
            (abs(xmax-xmin) < 1e-3 || abs(ymax-ymin) < 1e-3) && return
            zoom_bounds = (Float64(xmin), Float64(xmax), Float64(ymin), Float64(ymax))
            set_mollweide_view!(ax_img, zoom_bounds...)
            refresh_graticule_labels!(graticule, ax_img; bounds=zoom_bounds)
        elseif ev.button == Mouse.left && ev.action == Mouse.press && selection_mode[] != :point
            p = mouseposition(ax_img); any(isnan, p) && return
            mollweide_xy_to_lonlat(p[1], p[2]) === nothing && return
            region_start[] = Point2f(p[1], p[2])
            region_end[] = Point2f(p[1], p[2])
            region_drag_active[] = true
            region_ipix[] = Int[]
        elseif ev.button == Mouse.left && ev.action == Mouse.release && region_drag_active[]
            p = mouseposition(ax_img)
            !any(isnan, p) && (region_end[] = Point2f(p[1], p[2]))
            p0 = region_start[]; p1 = region_end[]
            region_drag_active[] = false
            if isfinite(p0[1]) && isfinite(p1[1])
                apply_region!(p0, p1)
            else
                clear_region!()
            end
        end
    end
    on(events(ax_img).mouseposition) do p
        if zoom_drag_active[] && !any(isnan, p)
            zoom_drag_end[] = Point2f(p[1], p[2])
        elseif region_drag_active[] && !any(isnan, p)
            region_end[] = Point2f(p[1], p[2])
        end
        region_drag_active[] && return
        !isempty(region_ipix[]) && return
        # info hover (l, b)
        ll = mollweide_xy_to_lonlat(p[1], p[2])
        if ll === nothing
            info_obs[] = latexstring("\\text{outside Mollweide ellipse}")
        else
            l_deg, b_deg = ll
            # on remappe l ∈ (-180, 180] → l ∈ [0, 360) pour conv. astro
            l_disp = mod(l_deg, 360)
            θhp = deg2rad(90 - b_deg)
            φhp = deg2rad(mod(l_deg, 360))
            ipix = Healpix.ang2pix(m, θhp, φhp)
            val  = m.pixels[ipix]
            valstr = (isfinite(val) && val != Healpix.UNSEEN) ?
                     string(round(Float32(val); digits=4)) : "NaN"
            info_obs[] = latexstring(
                "\\mathrm{(l,b)}=\\mathbf{(",
                string(round(l_disp; digits=2)), "^\\circ,",
                string(round(b_deg; digits=2)), "^\\circ)}\\;",
                "\\mathrm{pix}=\\mathbf{", string(ipix), "}\\;",
                "\\mathrm{", latex_safe(unit_label), "}=\\mathbf{", valstr, "}"
            )
        end
    end

    # save image
    save_root = save_dir === nothing ? begin
        d = joinpath(homedir(), "Desktop"); isdir(d) ? d : pwd()
    end : (isdir(save_dir) ? String(save_dir) : (mkpath(save_dir); String(save_dir)))

    on(save_btn.clicks) do _
        out = joinpath(save_root, "$(fname)_mollweide.png")
        try
            CairoMakie.save(String(out), fig; backend=CairoMakie)
            @info "Saved" out
        catch e
            @error "Failed to save" exception=(e, catch_backtrace())
        end
    end

    # ---------- Keyboard shortcuts (HEALPix Mollweide map) ----------
    _set_status_hp!(msg::AbstractString) =
        (info_obs[] = latexstring("\\text{", latex_safe(msg), "}"); nothing)
    _trigger_btn_hp!(btn) = (btn.clicks[] = btn.clicks[] + 1)
    function _toggle_contours_hp!()
        new_val = !show_contours[]
        contour_chk.checked[] = new_val
        _set_status_hp!(new_val ? "Contours enabled." : "Contours hidden.")
    end
    function _cycle_log_scale_hp!()
        next = cycle_scale_mode(scale_mode[])
        scale_menu.selection[] = String(next)
        _set_status_hp!("Image scale: $(String(next)).")
    end
    shortcuts_hp = ShortcutBinding[
        ShortcutBinding(Keyboard.i,  () -> (invert_cmap[] = !invert_cmap[]);
                        description = "invert cmap"),
        ShortcutBinding(Keyboard.a,  () -> _trigger_btn_hp!(auto_btn);
                        description = "auto contrast"),
        ShortcutBinding(Keyboard._1, () -> apply_percentile_clims!(1, 99);
                        description = "p1-p99"),
        ShortcutBinding(Keyboard._5, () -> apply_percentile_clims!(5, 95);
                        description = "p5-p95"),
        ShortcutBinding(Keyboard.r,  () -> _trigger_btn_hp!(reset_zoom_btn);
                        description = "reset zoom"),
        ShortcutBinding(Keyboard.s,  () -> _trigger_btn_hp!(save_btn);
                        description = "save image"),
        ShortcutBinding(Keyboard.c,  () -> _toggle_contours_hp!();
                        description = "contours"),
        ShortcutBinding(Keyboard.l,  () -> _cycle_log_scale_hp!();
                        description = "cycle scale"),
        ShortcutBinding(Keyboard.f,  () -> toggle_focus_image!();
                        description = "focus image"),
    ]
    # `g` toggles the graticule checkbox; its handler propagates to the
    # overlay, keeping the UI control and the shortcut in sync.
    push!(shortcuts_hp,
          ShortcutBinding(Keyboard.g,
                          () -> (graticule_chk.checked[] = !graticule_chk.checked[]);
                          description = "toggle graticule"))
    # `+`/`-` (and numpad) centered zoom on the Mollweide image axis.
    append!(shortcuts_hp, zoom_shortcut_bindings(ax_img; on_change = _set_status_hp!))
    # Help: Shift+/ and the Help button open a dedicated Makie figure
    # with the full binding list; status bar keeps the one-line recap.
    function _open_help_hp!()
        try
            open_shortcut_help_window(shortcuts_hp;
                title = "MANTA — HEALPix map shortcuts", theme = ui_theme)
        catch e
            @warn "Could not open shortcut help window" exception = (e, catch_backtrace())
        end
        _set_status_hp!(shortcut_help_message(shortcuts_hp))
    end
    push!(shortcuts_hp,
          ShortcutBinding(Keyboard.slash,
                          _open_help_hp!;
                          description = "this help",
                          modifier = :shift))
    on(help_btn.clicks) do _
        _open_help_hp!()
    end
    register_shortcuts!(fig, shortcuts_hp;
        textboxes = (clim_min_box, clim_max_box, contour_levels_box,
                     hist_bins_box, hist_xmin_box, hist_xmax_box,
                     hist_ymin_box, hist_ymax_box),
        is_blocked = () -> zoom_drag_active[] || region_drag_active[],
    )

    refresh_hist_axes!()
    register_window_close!(fig)  # anchor in GC root — see MANTA._KEEP_ALIVE block comment
    enable_file_drop!(fig; activate_gl = activate_gl, display_fig = display_fig)
    display_fig && display(fig)
    return fig
end
