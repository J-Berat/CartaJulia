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
    nx::Int = 1400,
    ny::Int = 700,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    hist_mode::Symbol = :bars,
    hist_bins::Int = 64,
    hist_xlimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    hist_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
)
    m          = ds.map
    column     = ds.column
    unit_label = ds.unit_label
    fname      = ds.source_id
    @info "HEALPix map" source=fname nside=m.resolution.nside npix=length(m)
    unit_label_tex = latexstring("\\text{", latex_safe(unit_label), "}")

    # ---------- Reprojection (une seule fois, conservée en mémoire) ----------
    # why: project once, then gather into both the displayed image and the
    # pixel-index grid used for region selection. Avoids running the
    # Mollweide trig + ang2pixRing twice.
    ipix_grid = mollweide_pixel_index(m.resolution, nx, ny)
    img_raw   = mollweide_apply_index(ipix_grid, m)

    # ---------- État ----------
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
        out = apply_scale(im, m_)
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
        automatic_contour_levels(im; n = 7)
    end
    contour_use_manual = Observable(false)
    contour_manual_levels = Observable(Float32[])
    contour_manual_colors = Observable(String[])
    contour_levels_obs = lift(contour_use_manual, contour_manual_levels, contour_auto_levels) do use_man, manual, auto
        use_man && !isempty(manual) ? manual : auto
    end
    contour_default_color = RGBAf(0, 0, 0, 0.62)
    contour_colors_obs = lift(contour_levels_obs, contour_use_manual, contour_manual_colors) do levels, use_man, colors
        contour_color_values(use_man ? colors : String[], length(levels), contour_default_color)
    end
    show_contours = Observable(false)

    hist_mode_obs = Observable(normalize_histogram_mode(hist_mode))
    hist_bins_obs = Observable(clamp(hist_bins, 4, 512))
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
    selection_mode = Observable(:point)
    region_shape = Observable(:box)
    region_drag_active = Observable(false)
    region_start = Observable(Point2f(NaN32, NaN32))
    region_end = Observable(Point2f(NaN32, NaN32))
    region_ipix = Observable(Int[])

    ui_theme = default_ui_theme()
    ui_accent = ui_theme.accent
    ui_selection = ui_theme.selection
    ui_text_muted = ui_theme.text_muted

    # ---------- Figure ----------
    activate_gl ? GLMakie.activate!() : CairoMakie.activate!()
    fig = Figure(size = _pick_fig_size(figsize), backgroundcolor = ui_theme.background)

    main_grid = fig[1, 1] = GridLayout()
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
    )
    xs = LinRange(-2f0, 2f0, nx)
    ys = LinRange(-1f0, 1f0, ny)
    img_for_plot = lift(img_disp) do im
        permutedims(im)  # (nx, ny) layout pour heatmap(xs, ys, A)
    end
    hm = heatmap!(ax_img, xs, ys, img_for_plot;
                  colormap=cm_obs, colorrange=clims_safe, nan_color=:white)
    contour!(ax_img, xs, ys, img_for_plot;
             levels=contour_levels_obs, color=contour_colors_obs, linewidth=1.1,
             visible=show_contours)
    full_map_bounds = (-2.0, 2.0, -1.0, 1.0)
    set_mollweide_view!(ax_img, full_map_bounds...)
    graticule = draw_mollweide_graticule!(ax_img)
    refresh_graticule_labels!(graticule, ax_img; bounds=full_map_bounds)

    # cadre ellipse Mollweide (purement esthétique)
    ell_x = [2cos(t) for t in LinRange(0, 2π, 200)]
    ell_y = [sin(t)  for t in LinRange(0, 2π, 200)]
    lines!(ax_img, ell_x, ell_y; color=:black, linewidth=0.8)

    # rectangle de zoom
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
    linesegments!(ax_img, zoom_box_segments; color=(ui_selection, 0.95),
                  linewidth=2.0, linestyle=:dash)
    region_segments = lift(region_start, region_end, region_shape, region_ipix, region_drag_active) do p0, p1, shape, ipixs, dragging
        (dragging || !isempty(ipixs)) ? projected_region_segments(p0, p1, shape) : Point2f[]
    end
    lines!(ax_img, region_segments; color=(ui_selection, 0.98), linewidth=2.3)

    Colorbar(
        main_grid[2, 1],
        hm;
        label = unit_label_tex,
        vertical = false,
        height = 18,
        tellwidth = false,
        halign = :center,
    )
    rowsize!(main_grid, 1, Relative(1))
    rowsize!(main_grid, 2, Fixed(52))

    # Bandeau info
    info_obs = Observable(latexstring("\\text{move cursor over the map}"))
    Label(main_grid[3, 1], info_obs; halign=:left, fontsize=15)
    rowsize!(main_grid, 3, Fixed(30))

    # Contrôles
    ax_hist = Axis(
        main_grid[4, 1];
        title = L"\text{Visible map histogram}",
        xlabel = unit_label_tex,
        ylabel = hist_ylabel_obs,
        # height is governed by `rowsize!(main_grid, 4, ...)` below — no
        # hard-coded value here (cf. CLAUDE.md / anti-patterns).
        xtickformat = _latex_tick_formatter,
        ytickformat = _latex_tick_formatter,
    )
    barplot!(ax_hist, hist_x_obs, hist_y_obs; width=hist_width_obs, color=(ui_accent, 0.44), strokecolor=ui_accent, strokewidth=0.3, visible=hist_bars_visible)
    lines!(ax_hist, hist_x_obs, hist_y_obs; color=ui_accent, linewidth=1.8, visible=hist_kde_visible)
    vlines!(ax_hist, lift(lim -> [first(lim), last(lim)], clims_safe);
            color=(ui_text_muted, 0.65), linewidth=1.0, linestyle=:dash)
    rowsize!(main_grid, 4, Fixed(105))

    ctrl = main_grid[5, 1] = GridLayout(; alignmode=Outside())
    rowsize!(main_grid, 5, Fixed(225))
    Label(ctrl[1,1], text=L"\text{Scale}", halign=:left, tellwidth=false, fontsize=15)
    scale_menu = Menu(ctrl[1,2]; options=["lin","log10","ln"],
                     prompt = String(scale), width=92)
    Label(ctrl[1,3], text=L"\text{Colormap}", halign=:left, tellwidth=false, fontsize=15)
    cmap_menu = Menu(ctrl[1,4]; options=ui_colormap_options(), prompt=String(cmap), width=112)
    invert_chk = Checkbox(ctrl[1,5])
    Label(ctrl[1,6], text="Invert", halign=:left, tellwidth=false, fontsize=15)
    invert_chk.checked[] = invert_cmap[]

    Label(ctrl[2,1], text=L"\text{Contrast}", halign=:left, tellwidth=false, fontsize=15)
    clim_min_box = Textbox(ctrl[2,2]; placeholder="min", width=110, height=30)
    clim_max_box = Textbox(ctrl[2,3]; placeholder="max", width=110, height=30)
    apply_btn    = Button(ctrl[2,4]; label="Apply", width=80, height=30)
    auto_btn     = Button(ctrl[2,5]; label="Auto", width=76, height=30)
    p1_btn       = Button(ctrl[2,6]; label="p1-p99", width=88, height=30)
    p5_btn       = Button(ctrl[2,7]; label="p5-p95", width=88, height=30)

    graticule_chk = Checkbox(ctrl[3,1])
    Label(ctrl[3,2], text="Graticule", halign=:left, tellwidth=false, fontsize=15)
    graticule_chk.checked[] = show_graticule[]
    reset_zoom_btn = Button(ctrl[3,3]; label="Reset zoom", width=120, height=30)
    save_btn       = Button(ctrl[3,4]; label="Save PNG", width=120, height=30)

    gauss_chk = Checkbox(ctrl[3,5])
    Label(ctrl[3,6], text="Smoothing", halign=:left, tellwidth=false, fontsize=15)
    sigma_label = Label(ctrl[3,7], text=latexstring("\\sigma = 1.5\\,\\text{px}"), fontsize=15, halign=:left, tellwidth=false)
    sigma_slider = Slider(ctrl[3,8:10]; range=LinRange(0, 10, 101), startvalue=1.5, width=210, height=14)

    Label(ctrl[4,1], text=L"\text{Selection}", halign=:left, tellwidth=false, fontsize=15)
    region_mode_menu = Menu(ctrl[4,2]; options=["point", "box", "circle"], prompt="point", width=108)
    region_clear_btn = Button(ctrl[4,3]; label="Clear selection", width=138, height=30)
    region_count_label = Label(ctrl[4,4]; text="0 pix", halign=:left, tellwidth=false, fontsize=15)
    Label(ctrl[5,1], text=L"\text{Contours}", halign=:left, tellwidth=false, fontsize=15)
    contour_chk = Checkbox(ctrl[5,2])
    Label(ctrl[5,3], text="Show", halign=:left, tellwidth=false, fontsize=15)
    contour_levels_box = Textbox(ctrl[5,4:6]; placeholder="auto or 1:red, 2:#00ffaa", width=250, height=30)
    contour_apply_btn = Button(ctrl[5,7]; label="Apply", width=80, height=30)
    contour_chk.checked[] = show_contours[]
    Label(ctrl[6,1], text=L"\text{Histogram}", halign=:left, tellwidth=false, fontsize=15)
    hist_mode_menu = Menu(ctrl[6,2]; options=["bars", "kde"], prompt=String(hist_mode_obs[]), width=92)
    hist_bins_box = Textbox(ctrl[6,3]; placeholder="bins", width=80, height=30)
    hist_xmin_box = Textbox(ctrl[6,4]; placeholder="x min", width=100, height=30)
    hist_xmax_box = Textbox(ctrl[6,5]; placeholder="x max", width=100, height=30)
    hist_apply_btn = Button(ctrl[6,6]; label="Apply x", width=88, height=30)
    hist_auto_btn = Button(ctrl[6,7]; label="Auto x", width=82, height=30)
    hist_ymin_box = Textbox(ctrl[6,8]; placeholder="y min", width=100, height=30)
    hist_ymax_box = Textbox(ctrl[6,9]; placeholder="y max", width=100, height=30)
    hist_y_apply_btn = Button(ctrl[6,10]; label="Apply y", width=88, height=30)
    hist_y_auto_btn = Button(ctrl[6,11]; label="Auto y", width=82, height=30)

    foreach(w -> manta_style_menu!(w, ui_theme), (scale_menu, cmap_menu, region_mode_menu, hist_mode_menu))
    foreach(w -> manta_style_button!(w, ui_theme), (apply_btn, auto_btn, p1_btn, p5_btn, reset_zoom_btn, save_btn, region_clear_btn, contour_apply_btn, hist_apply_btn, hist_auto_btn, hist_y_apply_btn, hist_y_auto_btn))
    foreach(w -> manta_style_checkbox!(w, ui_theme), (invert_chk, graticule_chk, gauss_chk, contour_chk))
    foreach(w -> manta_style_textbox!(w, ui_theme), (clim_min_box, clim_max_box, contour_levels_box, hist_bins_box, hist_xmin_box, hist_xmax_box, hist_ymin_box, hist_ymax_box))
    manta_style_slider!(sigma_slider, ui_theme)

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
            "\\text{region ", shape, "}\\;N=", length(ips),
            "\\;\\text{mean}=", valstr,
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
    on(reset_zoom_btn.clicks) do _
        set_mollweide_view!(ax_img, full_map_bounds...)
        refresh_graticule_labels!(graticule, ax_img; bounds=full_map_bounds)
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

    # zoom right-drag, identique à `manta`
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
            ipix = Healpix.ang2pixRing(m.resolution, θhp, φhp)
            val  = m.pixels[ipix]
            valstr = (isfinite(val) && val != Healpix.UNSEEN) ?
                     string(round(Float32(val); digits=4)) : "NaN"
            info_obs[] = latexstring(
                "(l, b) = (",
                string(round(l_disp; digits=2)), "^\\circ, ",
                string(round(b_deg; digits=2)), "^\\circ),\\;",
                "\\text{", latex_safe(unit_label), "} = ", valstr
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

    refresh_hist_axes!()
    keepalive!(fig)
    on(fig.scene.events.window_open) do is_open
        is_open || forget!(fig)
    end
    display_fig && display(fig)
    return fig
end
