# path: src/views/CubeView.jl
#
# 3D cube interactive viewer (slice + per-voxel/region spectrum + comparison
# + moments + power spectrum + FITS products + GIF export + WCS-aware ticks).
#
# This is the full cube viewer body, extracted verbatim from the inline
# definition that used to live in `MANTA.jl::manta(::String)`. The only
# changes versus the original are in the prologue: instead of receiving a
# filepath and reading the FITS itself, the function now takes a
# `CubeDataset` and recovers `filepath` / `header` from `ds.metadata` when
# available (loaders set `:fits_path` and may set `:fits_header`).
#
# Public entry point: `view_cube(ds; kwargs...)`. The internal name
# `_view_cube` is preserved for backwards compatibility with earlier drafts.
#
# Pure helpers used by this file live in sibling files included by
# MANTA.jl before this one:
#   - src/views/cube/PowerSpectrumBundle.jl  -> `_cube_ps_bundle`

"""
    view_cube(ds::CubeDataset; kwargs...) -> Figure

Open the full interactive cube viewer for a `CubeDataset`. Supported kwargs:
`cmap`, `vmin`, `vmax`, `invert`, `figsize`, `save_dir`, `activate_gl`,
`display_fig`, `settings_path`.

The viewer offers slice navigation, per-voxel/region spectra, an optional
3-D cube view, comparison overlay, moment maps (0/1/2), 2-D and 1-D power
spectra, FITS product export, PNG/PDF/CSV exports and GIF recording. When
the dataset was loaded from a FITS file, the viewer reuses that file's
directory for resolving comparison datasets and for export defaults.
"""
function _view_cube(
    ds::CubeDataset;
    cmap::Symbol = :viridis,
    vmin = nothing,
    vmax = nothing,
    invert::Bool = false,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    settings_path::Union{Nothing,AbstractString} = nothing,
    hist_mode::Symbol = :bars,
    hist_bins::Int = 64,
    hist_xlimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    hist_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    spec_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    moment_threshold::Real = 0.0,
    moment_nsigma::Union{Nothing,Real} = nothing,
    moment_channels::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    compare::Union{Nothing,AbstractString} = nothing,
)
    data = as_float32(ds.data)
    siz  = size(data)
    wcs  = ds.wcs
    unit_label = ds.unit_label
    unit_label_tex = latexstring("\\text{", latex_safe(unit_label), "}")

    slice_dims(axis::Integer) = if axis == 1
        (siz[2], siz[3])  # (y, z)
    elseif axis == 2
        (siz[1], siz[3])  # (x, z)
    else
        (siz[1], siz[2])  # (x, y)
    end

    slice_axis_dims(axis::Integer) = if axis == 1
        (2, 3)  # u=y, v=z
    elseif axis == 2
        (1, 3)  # u=x, v=z
    else
        (1, 2)  # u=x, v=y
    end

    pixel_axis_name(dim::Integer) = dim == 1 ? "pixel x" : dim == 2 ? "pixel y" : "pixel z"

    slice_axis_labels(axis::Integer) = begin
        u_dim, v_dim = slice_axis_dims(axis)
        (
            wcs_axis_label(wcs, v_dim; fallback = pixel_axis_name(v_dim)),
            wcs_axis_label(wcs, u_dim; fallback = pixel_axis_name(u_dim)),
        )
    end

    pixel_world_tick_formatter(dim::Integer) = vals -> [
        has_wcs(wcs, dim) ? latex_tick(world_coord(wcs, dim, v)) : latex_tick(v)
        for v in vals
    ]

    spectral_coords(dim::Integer) = Float32[
        has_wcs(wcs, dim) ? Float32(world_coord(wcs, dim, chan)) : Float32(chan)
        for chan in 1:siz[dim]
    ]

    # Source identification. `filepath` is "" for in-memory cubes; callers
    # that loaded the cube from disk get the original FITS path back through
    # `ds.metadata[:fits_path]`. The same is true for `:fits_header`.
    fits_path  = get(ds.metadata, :fits_path, nothing)
    filepath   = fits_path isa AbstractString ? String(fits_path) : ""
    fname_full = filepath != "" ? basename(filepath) : String(ds.source_id)
    fname      = String(replace(fname_full, r"\.fits(\.gz)?$"i => ""))
    header     = get(ds.metadata, :fits_header, nothing)

    @info "Cube ready" source=ds.source_id size=siz

    # ---------- State ----------
    axis   = Observable(3)          # 1/2/3
    idx    = Observable(1)          # slice index

    i_idx  = Observable(1)          # voxel indices
    j_idx  = Observable(1)
    k_idx  = Observable(1)

    u_idx  = Observable(1)          # row
    v_idx  = Observable(1)          # col

    cmap_name   = Observable(cmap)
    invert_cmap = Observable(invert)
    cm_obs = lift(cmap_name, invert_cmap) do name, inv
        base = to_cmap(name); inv ? reverse(base) : base
    end

    img_scale_mode  = Observable(:lin)
    spec_scale_mode = Observable(:lin)
    compare_data = Observable{Any}(nothing)
    compare_visible = Observable(false)
    compare_name = Observable("")
    compare_mode = Observable(:B)
    view_product = Observable(:slice)
    moment_order = Observable(0)
    layout_mode = Observable(:base)
    anim_playing = Observable(false)

    # why: read-only display path — view avoids per-frame allocation of the slice
    # buffer. Mutating consumers downstream must clone; current ones build their
    # own output (`apply_scale`, `imfilter(Float32.(s), ...)`, `dual_view_product`).
    slice_raw = lift(axis, idx) do a, id
        get_slice_view(data, a, clamp(id, 1, siz[a]))
    end

    compare_slice_raw = lift(compare_data, axis, idx) do cmp, a, id
        cmp === nothing && return fill(NaN32, slice_dims(a))
        get_slice_view(cmp, a, clamp(id, 1, siz[a]))
    end

    gauss_on = Observable(false)
    sigma    = Observable(1.5f0)
    # why: Kernel.gaussian((σ, σ)) is rebuilt on every slice / compare update.
    # The viewer is dominated by frames at constant σ, so cache by σ value.
    # Scope is local to this `_view_cube` invocation — released when the
    # figure is GC'd. Float32 keys are well-behaved (σ is always > 0 here).
    _gauss_kernel_cache = Dict{Float32, Any}()
    _get_gauss_kernel(σ::Real) = get!(_gauss_kernel_cache, Float32(σ)) do
        ImageFiltering.Kernel.gaussian((Float32(σ), Float32(σ)))
    end
    show_crosshair = Observable(true)
    show_marker    = Observable(true)
    show_grid      = Observable(false)
    show_contours  = Observable(false)
    contour_use_manual = Observable(false)
    contour_manual_levels = Observable(Float32[])
    contour_manual_colors = Observable(String[])
    selection_mode = Observable(:point)
    region_shape = Observable(:box)
    region_uvs = Observable(Tuple{Int,Int}[])
    region_start = Observable(Point2f(NaN32, NaN32))
    region_end = Observable(Point2f(NaN32, NaN32))
    region_drag_active = Observable(false)
    zoom_drag_active = Observable(false)
    zoom_drag_start  = Observable(Point2f(NaN32, NaN32))
    zoom_drag_end    = Observable(Point2f(NaN32, NaN32))

    ui_theme = default_ui_theme()
    ui_accent = ui_theme.accent
    ui_accent_strong = ui_theme.accent_strong
    ui_surface = ui_theme.surface
    ui_panel = ui_theme.panel
    ui_panel_header = ui_theme.panel_header
    ui_border = ui_theme.border
    ui_text = ui_theme.text
    ui_text_muted = ui_theme.text_muted
    ui_selection = ui_theme.selection
    ui_compare = ui_theme.compare
    ui_success = ui_theme.success
    fig_bg = ui_theme.background

    style_checkbox!(chk) = manta_style_checkbox!(chk, ui_theme; compact = compact_layout)
    style_slider!(sl) = manta_style_slider!(sl, ui_theme; compact = compact_layout)
    style_button!(btn) = manta_style_button!(btn, ui_theme; compact = compact_layout)
    style_menu!(menu) = manta_style_menu!(menu, ui_theme; compact = compact_layout)
    style_textbox!(tb) = manta_style_textbox!(tb, ui_theme; compact = compact_layout)

    # `latex_tick` and `latex_tick_formatter` are now module-level helpers in
    # `src/helpers/UIBits.jl` — they used to live inline here.

    base_slice_proc = lift(slice_raw, gauss_on, sigma) do s, on, σ
        if on && σ > 0
            imfilter(Float32.(s), _get_gauss_kernel(σ))
        else
            s
        end
    end

    # Δx along the slicing axis comes from the WCS step when available; for
    # a pure-channel axis this falls back to dx=1 so M0 stays numerically
    # in "K·channel" rather than silently relying on the old summation.
    moment_dx_for(a::Integer) = has_wcs(wcs, a) ? abs(Float64(wcs[a].cdelt)) : 1.0

    # why: data is immutable during a viewer session and threshold / nsigma /
    # channels are passed once at construction. Cache by (axis, order) so
    # toggling axis or moment order doesn't recompute the whole map. Up to
    # 3 axes × 3 orders = 9 entries.
    _moment_cache = Dict{Tuple{Int,Int}, Matrix{Float32}}()
    moment_raw = lift(axis, moment_order) do a, ord
        get!(_moment_cache, (Int(a), Int(ord))) do
            moment_map(data, a, ord; coords = spectral_coords(a),
                       threshold = moment_threshold,
                       nsigma = moment_nsigma,
                       channels = moment_channels === nothing ? (1:siz[a]) : moment_channels,
                       dx = moment_dx_for(a))
        end
    end

    view_raw = lift(slice_raw, moment_raw, view_product) do s, m, product
        product === :moment ? m : s
    end

    slice_proc = lift(view_raw, gauss_on, sigma, view_product) do s, on, σ, product
        if product === :slice && on && σ > 0
            imfilter(Float32.(s), _get_gauss_kernel(σ))
        else
            s
        end
    end

    compare_slice_proc = lift(compare_slice_raw, gauss_on, sigma) do s, on, σ
        if on && σ > 0
            imfilter(Float32.(s), _get_gauss_kernel(σ))
        else
            s
        end
    end

    compare_product_proc = lift(base_slice_proc, compare_slice_proc, compare_mode, compare_data) do a, b, mode, cmp
        cmp === nothing && return fill(NaN32, size(a))
        dual_view_product(a, b, mode)
    end

    # ---- display array: protect against NaN/Inf after log/ln
    # why: apply_scale_display fuses the scale step and the NaN→0 sanitization
    # into a single pass, saving one allocation per slice update.
    slice_disp = lift(slice_proc, img_scale_mode) do s, m
        apply_scale_display(s, m)
    end

    compare_slice_disp = lift(compare_product_proc, img_scale_mode) do s, m
        apply_scale_display(s, m)
    end

    clims_auto = lift(slice_disp) do s
        clamped_extrema(s)
    end

    contour_auto_levels = lift(slice_disp) do s
        automatic_contour_levels(s; n = 7)
    end

    contour_levels_obs = lift(contour_use_manual, contour_manual_levels, contour_auto_levels) do use_man, manual, auto
        use_man && !isempty(manual) ? manual : auto
    end
    contour_default_color = RGBAf(1, 1, 1, 0.72)
    contour_colors_obs = lift(contour_levels_obs, contour_use_manual, contour_manual_colors) do levels, use_man, colors
        contour_color_values(use_man ? colors : String[], length(levels), contour_default_color)
    end

    clims_manual = Observable((0f0, 1f0))
    use_manual   = Observable(false)

    if vmin !== nothing && vmax !== nothing
        vmin_f, vmax_f = Float32(vmin), Float32(vmax)
        if vmin_f == vmax_f
            vmin_f = prevfloat(vmin_f); vmax_f = nextfloat(vmax_f)  # avoid zero-width
        end
        clims_manual[] = (vmin_f, vmax_f)
        use_manual[]   = true
    end

    clims_obs = lift(use_manual, clims_auto, clims_manual) do um, ca, cm
        um ? cm : ca
    end

    # safe clims for plotting/layout
    clims_safe = lift(clims_obs) do (cmin, cmax)
        if !(isfinite(cmin) && isfinite(cmax)) || isnan(cmin) || isnan(cmax) || cmin == cmax
            (0f0, 1f0)
        else
            (cmin, cmax)
        end
    end

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

    hist_pair_obs = lift(slice_disp, hist_limits_obs, hist_bins_obs, hist_mode_obs) do s, lim, bins, mode
        histogram_profile(s; bins = bins, limits = lim, mode = mode)
    end
    hist_x_obs = lift(p -> p.x, hist_pair_obs)
    hist_y_obs = lift(p -> p.y, hist_pair_obs)
    hist_width_obs = lift(p -> p.width, hist_pair_obs)
    hist_bars_visible = lift(m -> m === :bars, hist_mode_obs)
    hist_kde_visible = lift(m -> m === :kde, hist_mode_obs)
    hist_ylabel_obs = lift(histogram_ylabel, hist_mode_obs)

    compare_hist_pair_obs = lift(compare_slice_proc, img_scale_mode, compare_visible, hist_limits_obs, hist_bins_obs, hist_mode_obs) do s, scale_mode, visible, lim, bins, hist_mode_
        visible || return (Float32[], Float32[])
        A = apply_scale(s, scale_mode)
        out = similar(A, Float32)
        @inbounds for i in eachindex(A)
            x = A[i]
            out[i] = isfinite(x) ? Float32(x) : 0f0
        end
        profile = histogram_profile(out; bins = bins, limits = lim, mode = hist_mode_)
        return (profile.x, profile.y)
    end
    compare_hist_x_obs = lift(p -> p[1], compare_hist_pair_obs)
    compare_hist_y_obs = lift(p -> p[2], compare_hist_pair_obs)

    compare_clims_safe = lift(compare_slice_disp, compare_mode, clims_safe) do s, mode, lim
        if mode in (:A, :B)
            lim
        else
            clamped_extrema(s)
        end
    end

    spec_x_axes = (collect(0:(siz[1] - 1)), collect(0:(siz[2] - 1)), collect(0:(siz[3] - 1)))
    spec_y_buf  = Vector{Float32}(undef, siz[3])
    @views copyto!(spec_y_buf, data[1, 1, :])
    spec_x_raw  = Observable(spec_x_axes[3])
    spec_y_raw  = Observable(spec_y_buf)
    spec_y_disp = lift(spec_y_raw, spec_scale_mode) do y, m
        apply_scale(y, m)
    end
    # Second-cube spectrum buffer / Observables — same x grid as cube A
    # (same region selection), filled by `refresh_spectrum!` only when a
    # comparison cube is loaded. NaN-init so the (hidden) line shows
    # nothing before the first refresh.
    spec_y_compare_buf = Vector{Float32}(undef, siz[3])
    fill!(spec_y_compare_buf, NaN32)
    spec_y_compare_raw = Observable(spec_y_compare_buf)
    spec_y_compare_disp = lift(spec_y_compare_raw, spec_scale_mode) do y, m
        apply_scale(y, m)
    end
    spec_ylimits_value = Observable(spec_ylimits === nothing ?
        (use_manual[] ? clims_manual[] : (0f0, 1f0)) :
        parse_spectrum_ylimits(string(first(spec_ylimits)), string(last(spec_ylimits)))[3])
    spec_ylimits_source = Observable(spec_ylimits === nothing ? (use_manual[] ? :contrast : :auto) : :manual)

    # ---------- Figure & layout ----------
    if activate_gl
        GLMakie.activate!()
    else
        CairoMakie.activate!()
    end
    fig_size = _pick_fig_size(figsize)
    compact_layout = fig_size[1] <= 1500 || fig_size[2] <= 950
    spec_axis_height = compact_layout ? 185 : 320
    hist_axis_height = compact_layout ? 60 : 105
    ps_header_height = compact_layout ? 0 : 90
    ps_axis_size = compact_layout ? 320 : 520
    controls_row_heights = compact_layout ? (42, 212, 164) : (46, 214, 146)
    controls_gap = compact_layout ? 8 : 16
    controls_height = sum(controls_row_heights) + 2 * controls_gap
    card_pad = compact_layout ? 9 : 12
    card_gap = compact_layout ? 7 : 10
    main_row_gap = compact_layout ? 8 : 14
    plot_row_height = compact_layout ? max(320, fig_size[2] - controls_height - 8 * main_row_gap) : 0
    # Hauteur réservée à la rangée 1 quand on stack heatmap + PSD en mode :power_spectrum.
    ps_plot_row_height = max(360, fig_size[2] - controls_height - 2 * main_row_gap - 16)

    fig = Figure(size = fig_size, backgroundcolor = fig_bg)

    main_grid = fig[1, 1] = GridLayout()
    colgap!(main_grid, 18)
    rowgap!(main_grid, main_row_gap)
    # Image + contrast scale
    img_grid  = main_grid[1, 1] = GridLayout()
    colgap!(img_grid, -8)

    xlab0, ylab0 = slice_axis_labels(axis[])
    main_title_obs = lift(view_product, moment_order) do product, order
        product === :moment ?
            latexstring("\\text{", latex_safe(fname), " ", latex_safe(order == 0 ? "moment 0" : order == 1 ? "moment 1" : "moment 2"), "}") :
            make_main_title(fname)
    end
    # Moment captions follow the spectral quantity along the integrated axis:
    # FREQ → "mean frequency / frequency dispersion", WAVE → "wavelength",
    # VRAD/VOPT → "velocity". Falls back to "value" when no WCS classifies
    # the axis (channel-only cubes).
    spectral_word_for(a::Integer) =
        spectral_quantity_word(spectral_quantity(wcs, a))
    moment_unit_for(a::Integer) = begin
        has_wcs(wcs, a) && !isempty(wcs[a].cunit) ? " [" * wcs[a].cunit * "]" : ""
    end
    integrated_intensity_label(a::Integer) = begin
        u_ax = moment_unit_for(a)
        u = isempty(u_ax) ? unit_label :
            (unit_label == "value" ? strip(u_ax, [' ', '[', ']']) :
             unit_label * "·" * strip(u_ax, [' ', '[', ']']))
        "integrated intensity [" * String(u) * "]"
    end
    display_unit_label = lift(view_product, moment_order, axis) do product, order, a
        if product !== :moment
            return unit_label_tex
        end
        word = spectral_word_for(a)
        text = order == 0 ? integrated_intensity_label(a) :
               order == 1 ? "mean " * word * moment_unit_for(a) :
                            word * " dispersion" * moment_unit_for(a)
        latexstring("\\text{", latex_safe(text), "}")
    end
    ax_img = Axis(
        img_grid[1, 1];
        title     = main_title_obs,
        xlabel    = xlab0,
        ylabel    = ylab0,
        aspect    = DataAspect(),
        xtickformat = pixel_world_tick_formatter(slice_axis_dims(axis[])[2]),
        ytickformat = pixel_world_tick_formatter(slice_axis_dims(axis[])[1]),
    )
    compare_mode_label(mode::Symbol) = mode === :A ? "A" :
        mode === :B ? "B" :
        mode === :diff ? "A - B" :
        mode === :ratio ? "A / B" :
        "normalized residuals"

    compare_title_obs = lift(compare_name, compare_mode) do name, mode
        label = compare_mode_label(mode)
        isempty(name) ? latexstring("\\text{", latex_safe(label), "}") :
            latexstring("\\text{", latex_safe(label), ": ", latex_safe(name), "}")
    end
    ax_cmp = Axis(
        img_grid[1, 2];
        title     = compare_title_obs,
        xlabel    = xlab0,
        ylabel    = ylab0,
        aspect    = DataAspect(),
        xtickformat = pixel_world_tick_formatter(slice_axis_dims(axis[])[2]),
        ytickformat = pixel_world_tick_formatter(slice_axis_dims(axis[])[1]),
    )
    if compact_layout
        ax_img.width[] = ps_axis_size
        ax_img.height[] = ps_axis_size
        ax_cmp.width[] = ps_axis_size
        ax_cmp.height[] = ps_axis_size
    end
    colsize!(img_grid, 2, Fixed(0))

    uv_point = Observable(Point2f(1, 1))
    hm = heatmap!(ax_img, slice_disp; colormap = cm_obs, colorrange = clims_safe)
    heatmap!(ax_cmp, compare_slice_disp; colormap = cm_obs, colorrange = compare_clims_safe, visible = compare_visible)
    contour!(ax_img, slice_disp; levels = contour_levels_obs, color = contour_colors_obs, linewidth = 1.2, visible = show_contours)
    compare_contours_visible = lift(show_contours, compare_visible) do contours, visible
        contours && visible
    end
    contour!(ax_cmp, compare_slice_disp; levels = contour_levels_obs, color = contour_colors_obs, linewidth = 1.2, visible = compare_contours_visible)
    crosshair_segments = lift(axis, u_idx, v_idx, show_crosshair) do a, u, v, enabled
        enabled || return Point2f[]
        u_max, v_max = slice_dims(a)
        Point2f[
            Point2f(1, u), Point2f(v_max, u),
            Point2f(v, 1), Point2f(v, u_max),
        ]
    end
    zoom_box_segments = lift(zoom_drag_active, zoom_drag_start, zoom_drag_end) do active, p0, p1
        active || return Point2f[]
        if !(isfinite(p0[1]) && isfinite(p0[2]) && isfinite(p1[1]) && isfinite(p1[2]))
            return Point2f[]
        end
        x0, y0 = p0
        x1, y1 = p1
        Point2f[
            Point2f(x0, y0), Point2f(x1, y0),
            Point2f(x1, y0), Point2f(x1, y1),
            Point2f(x1, y1), Point2f(x0, y1),
            Point2f(x0, y1), Point2f(x0, y0),
        ]
    end
    region_segments_from_points(p0, p1, shape::Symbol) = begin
        if !(isfinite(p0[1]) && isfinite(p0[2]) && isfinite(p1[1]) && isfinite(p1[2]))
            return Point2f[]
        end
        x0, y0 = p0
        x1, y1 = p1
        if shape === :circle
            r = hypot(x1 - x0, y1 - y0)
            r < 0.5 && return Point2f[]
            pts = Point2f[]
            for t in LinRange(0, 2π, 97)
                push!(pts, Point2f(x0 + r * cos(t), y0 + r * sin(t)))
            end
            return pts
        else
            return Point2f[
                Point2f(x0, y0), Point2f(x1, y0),
                Point2f(x1, y1), Point2f(x0, y1),
                Point2f(x0, y0),
            ]
        end
    end
    region_segments = lift(region_start, region_end, region_shape, region_uvs, region_drag_active) do p0, p1, shape, uv, dragging
        (dragging || !isempty(uv)) ? region_segments_from_points(p0, p1, shape) : Point2f[]
    end
    linesegments!(ax_img, crosshair_segments; color = (:white, 0.9), linewidth = 1.6, linestyle = :dot)
    linesegments!(ax_cmp, crosshair_segments; color = (:white, 0.9), linewidth = 1.6, linestyle = :dot, visible = compare_visible)
    linesegments!(ax_img, zoom_box_segments; color = (ui_selection, 0.95), linewidth = 2.0, linestyle = :dash)
    linesegments!(ax_cmp, zoom_box_segments; color = (ui_selection, 0.95), linewidth = 2.0, linestyle = :dash, visible = compare_visible)
    lines!(ax_img, region_segments; color = (ui_selection, 0.98), linewidth = 2.4)
    lines!(ax_cmp, region_segments; color = (ui_selection, 0.98), linewidth = 2.4, visible = compare_visible)
    marker_points = lift(uv_point, show_marker) do p, enabled
        enabled ? Point2f[p] : Point2f[]
    end
    scatter!(ax_img, marker_points; markersize = 10)
    scatter!(ax_cmp, marker_points; markersize = 10, visible = compare_visible)

    # Colorbar linked to plot; tellheight=false avoids layout feedback loops
    img_colorbar = Colorbar(
        img_grid[1, 3],
        hm;
        label = display_unit_label,
        width = 20,
        height = _axis_render_height(ax_img),
        tellheight = false,
        valign = :center,
    )
    # Dedicated colorbar for cube B, slot 4 — hidden until a comparison cube
    # is loaded (mirrors how `ax_cmp` is hidden via `colsize!(img_grid, 2, ...)`).
    img_colorbar_cmp = Colorbar(
        img_grid[1, 4];
        colormap = cm_obs,
        colorrange = compare_clims_safe,
        label = display_unit_label,
        width = 20,
        height = _axis_render_height(ax_cmp),
        tellheight = false,
        valign = :center,
    )
    colsize!(img_grid, 4, Fixed(0))

    # Info + spectrum
    spec_grid = main_grid[1, 2] = GridLayout()
    info_panel = spec_grid[1, 1] = GridLayout(; alignmode = Outside())
    info_box = Box(
        info_panel[1, 1];
        color = ui_surface,
        strokecolor = ui_border,
        strokewidth = 1.0,
        cornerradius = 12,
        z = -5,
    )
    lab_info = Label(
        info_panel[1, 1];
        text      = make_info_tex(1, 1, 1, 1, 1, 0f0),
        halign    = :left,
        valign    = :center,
        fontsize  = 16,
        color     = ui_text,
        padding   = (16, 16, 12, 12),
        lineheight = 1.2,
        tellwidth = false,
    )

    ax_spec = Axis(
        spec_grid[2, 1];
        title  = L"\text{Spectrum at selected pixel}",
        xlabel = L"\text{index along slice axis}",
        ylabel = unit_label_tex,
        width  = 600,
        height = spec_axis_height,
        xtickformat = latex_tick_formatter,
        ytickformat = latex_tick_formatter,
    )
    lines!(ax_spec, spec_x_raw, spec_y_disp; label = "A")
    lines!(ax_spec, spec_x_raw, spec_y_compare_disp;
           color = ui_compare, linewidth = 1.6,
           visible = compare_visible, label = "B")
    # Legend gated by compare_visible — hidden when only cube A is loaded so
    # the legend chip does not waste space.
    spec_legend = axislegend(ax_spec; position = :rt,
                             framevisible = true,
                             backgroundcolor = (ui_panel, 0.85),
                             labelcolor = ui_text,
                             framecolor = ui_border,
                             padding = (8, 8, 4, 4))
    # Toggle whole Legend block across Makie versions: older Makie exposes
    # `.visible`, newer Makie only exposes `.scene.visible` / `.blockscene.visible`.
    _set_legend_visible! = (v::Bool) -> begin
        try; spec_legend.visible[] = v; catch; end
        try; spec_legend.scene.visible[] = v; catch; end
        try; spec_legend.blockscene.visible[] = v; catch; end
        nothing
    end
    _set_legend_visible!(compare_visible[])
    on(compare_visible) do v
        _set_legend_visible!(v)
    end
    ax_img.xgridvisible[] = show_grid[]
    ax_img.ygridvisible[] = show_grid[]
    ax_cmp.xgridvisible[] = show_grid[]
    ax_cmp.ygridvisible[] = show_grid[]
    ax_spec.xgridvisible[] = show_grid[]
    ax_spec.ygridvisible[] = show_grid[]

    ax_hist = Axis(
        spec_grid[3, 1];
        title = L"\text{Visible slice histogram}",
        xlabel = unit_label_tex,
        ylabel = hist_ylabel_obs,
        height = hist_axis_height,
        xtickformat = latex_tick_formatter,
        ytickformat = latex_tick_formatter,
    )
    barplot!(ax_hist, hist_x_obs, hist_y_obs; width = hist_width_obs, color = (ui_accent, 0.44), strokecolor = ui_accent, strokewidth = 0.3, visible = hist_bars_visible)
    lines!(ax_hist, hist_x_obs, hist_y_obs; color = ui_accent, linewidth = 1.8, visible = hist_kde_visible)
    lines!(ax_hist, compare_hist_x_obs, compare_hist_y_obs; color = ui_compare, linewidth = 1.6, visible = compare_visible)
    vlines!(ax_hist, lift(lim -> [first(lim), last(lim)], clims_safe); color = (ui_text_muted, 0.65), linewidth = 1.1, linestyle = :dash)

    ps_layout = main_grid[1, 2] = GridLayout(;
        alignmode = Outside(compact_layout ? 4 : 8),
        halign = :center,
        valign = :top,
        tellwidth = false,
        tellheight = false,
    )
    ps_header = ps_layout[1, 1] = GridLayout()
    colgap!(ps_header, 8)
    rowgap!(ps_header, compact_layout ? 6 : 8)
    ps_ui_blocks = Any[]
    track_ps!(block) = (push!(ps_ui_blocks, block); block)
    track_ps!(Label(ps_header[1, 1]; text = "Source", halign = :right, fontsize = 12, color = ui_text_muted))
    ps_src_menu = track_ps!(Menu(ps_header[1, 2]; options = ["zoom", "full"], prompt = "zoom", width = 76))
    track_ps!(Label(ps_header[1, 3]; text = "Window", halign = :right, fontsize = 12, color = ui_text_muted))
    ps_win_menu = track_ps!(Menu(ps_header[1, 4]; options = ["Hann", "Hamming", "None"], prompt = "Hann", width = 82))
    track_ps!(Label(ps_header[1, 5]; text = "Units", halign = :right, fontsize = 12, color = ui_text_muted))
    ps_unit_menu = track_ps!(Menu(ps_header[1, 6]; options = ["pixel", "physical"], prompt = "pixel", width = 76))
    ps_refresh_btn = track_ps!(Button(ps_header[1, 7]; label = "Refresh", width = 76, height = 28))

    ps_pad_chk = track_ps!(Checkbox(ps_header[2, 1]))
    track_ps!(Label(ps_header[2, 2]; text = "Pad", halign = :left, fontsize = 12, color = ui_text))
    ps_nanapo_chk = track_ps!(Checkbox(ps_header[2, 3]))
    track_ps!(Label(ps_header[2, 4]; text = "NaN", halign = :left, fontsize = 12, color = ui_text))
    ps_fit_chk = track_ps!(Checkbox(ps_header[2, 5]))
    track_ps!(Label(ps_header[2, 6]; text = "Fit", halign = :left, fontsize = 12, color = ui_text))
    ps_kmin_box = track_ps!(Textbox(ps_header[2, 7]; placeholder = "k_min", width = 70, height = 28))
    ps_kmax_box = track_ps!(Textbox(ps_header[2, 8]; placeholder = "k_max", width = 70, height = 28))
    ps_popout_btn = track_ps!(Button(ps_header[2, 9]; label = "Window", width = 76, height = 28))

    ps_layout_status = Observable(" ")

    ps_plot_grid = ps_layout[2, 1] = GridLayout(2, 2; halign = :center, valign = :top)
    colgap!(ps_plot_grid, -8)
    rowgap!(ps_plot_grid, compact_layout ? 6 : 12)
    rowsize!(ps_layout, 1, Fixed(ps_header_height))
    rowsize!(ps_layout, 2, Relative(1))
    colsize!(ps_plot_grid, 1, Auto())
    rowsize!(ps_plot_grid, 1, Auto())
    rowsize!(ps_plot_grid, 2, Auto())

    # Controls
    controls_grid = main_grid[2, 1:2] = GridLayout(; alignmode = Outside())
    colgap!(controls_grid, controls_gap)
    rowgap!(controls_grid, controls_gap)
    rowsize!(main_grid, 2, Fixed(controls_height))
    compact_layout && rowsize!(main_grid, 1, Fixed(plot_row_height))

    function control_card!(parent, row, col, title::AbstractString; rows::Int = 4, cols::Int = 4)
        card = parent[row, col] = GridLayout(;
            alignmode = Outside(card_pad),
            tellwidth = false,
            tellheight = false,
        )
        body_rows = rows + 1
        # Card body
        Box(card[1:body_rows, 1:cols];
            color = ui_panel, strokecolor = ui_border,
            strokewidth = 1.0, cornerradius = 8, z = -6)
        # Header band (visually distinct title row)
        Box(card[1, 1:cols];
            color = ui_panel_header, strokecolor = (:transparent, 0.0),
            strokewidth = 0.0, cornerradius = 8, z = -5)
        Label(card[1, 1:cols];
            text = uppercase(title),
            halign = :left, tellwidth = false,
            fontsize = 13, font = :bold,
            color = ui_accent_strong,
            padding = (10, 10, 6, 6))
        Box(card[body_rows, 1:cols]; color = :transparent, strokewidth = 0, z = -7)
        rowsize!(card, body_rows, Fixed(compact_layout ? 10 : 12))
        rowgap!(card, card_gap)
        colgap!(card, card_gap)
        return card
    end
    control_label!(layout, pos, txt) = Label(layout[pos...]; text = txt, halign = :left, tellwidth = false, fontsize = 13, color = ui_text_muted)

    mode_bar = controls_grid[1, 1:3] = GridLayout(; alignmode = Outside(0))
    colgap!(mode_bar, compact_layout ? 6 : 10)
    mode_nav_btn = Button(mode_bar[1, 1]; label = "Navigation", width = 140, height = 32)
    mode_analysis_btn = Button(mode_bar[1, 2]; label = "Analysis", width = 126, height = 32)
    mode_export_btn = Button(mode_bar[1, 3]; label = "Export", width = 104, height = 32)
    foreach(c -> colsize!(mode_bar, c, Auto()), 1:3)
    control_mode = Observable(:navigation)

    view_card = control_card!(controls_grid, 2, 1, "View"; rows = 5, cols = 4)
    control_label!(view_card, (2, 1), "Image")
    img_scale_menu = Menu(view_card[2, 2]; options = ["lin", "log10", "ln"], prompt = "lin", width = 96)
    control_label!(view_card, (3, 1), "Spectrum")
    spec_scale_menu = Menu(view_card[3, 2]; options = ["lin", "log10", "ln"], prompt = "lin", width = 96)
    reset_zoom_btn = Button(view_card[2, 3:4]; label = "Reset zoom", width = 132, height = 32)
    ps_btn = Button(view_card[4, 1:4]; label = "Power spectrum layout", width = 240, height = 32)
    base_layout_btn = Button(view_card[5, 1:4]; label = "Base layout", width = 240, height = 32)
    foreach(c -> colsize!(view_card, c, Auto()), 1:4)

    slice_card = control_card!(controls_grid, 2, 2, "Slice"; rows = 4, cols = 5)
    axes_labels = ["dim1 (x)", "dim2 (y)", "dim3 (z)"]
    control_label!(slice_card, (2, 1), "Axis")
    axis_menu = Menu(slice_card[2, 2]; options = axes_labels, prompt = "dim3 (z)", width = 128)
    status_label = Label(slice_card[2, 3:5]; text = latexstring("\\text{axis } 3,\\, \\text{index } 1"), fontsize = 14, halign = :left, tellwidth = false, color = ui_text)
    control_label!(slice_card, (3, 1), "Index")
    slice_slider = Slider(
        slice_card[3, 2:4];
        range = 1:siz[3],
        startvalue = 1,
        width = compact_layout ? 220 : 260,
        height = 26,
        halign = :left,
    )
    control_label!(slice_card, (4, 1), "Smoothing")
    sigma_label = Label(slice_card[4, 2]; text = latexstring("\\sigma = 1.5\\,\\text{px}"), fontsize = 14, halign = :left, tellwidth = false, color = ui_text)
    sigma_slider = Slider(
        slice_card[4, 3:4];
        range = LinRange(0, 10, 101),
        startvalue = 1.5,
        width = compact_layout ? 150 : 190,
        height = 26,
        halign = :left,
    )
    foreach(c -> colsize!(slice_card, c, Auto()), 1:5)

    contrast_card = control_card!(controls_grid, 2, 1, "Contrast"; rows = 4, cols = 5)
    clim_min_box   = Textbox(contrast_card[2, 1]; placeholder = "min", width = 120, height = 32)
    clim_max_box   = Textbox(contrast_card[2, 2]; placeholder = "max", width = 120, height = 32)
    clim_apply_btn = Button(contrast_card[2, 3]; label = "Apply", width = 86, height = 32)
    clim_auto_btn  = Button(contrast_card[2, 4]; label = "Auto", width = 78, height = 32)
    clim_p1_btn    = Button(contrast_card[3, 1]; label = "p1-p99", width = 92, height = 32)
    clim_p5_btn    = Button(contrast_card[3, 2]; label = "p5-p95", width = 92, height = 32)
    reset_zoom_analysis_btn = Button(contrast_card[3, 3:4]; label = "Reset zoom", width = 132, height = 32)
    foreach(c -> colsize!(contrast_card, c, Auto()), 1:5)

    output_card = control_card!(controls_grid, 2, 1, "Output"; rows = 5, cols = 5)
    fmt_menu  = Menu(output_card[2, 1]; options = ["png", "pdf"], prompt = "png", width = 90)
    fname_box = Textbox(output_card[2, 2:4]; placeholder = "filename base", width = 220, height = 32)
    reset_zoom_export_btn = Button(output_card[2, 5]; label = "Reset zoom", width = 132, height = 32)
    btn_save_img  = Button(output_card[3, 1]; label = "Save image", width = 116, height = 32)
    btn_save_spec = Button(output_card[3, 2]; label = "Save spectrum", width = 138, height = 32)
    btn_save_state = Button(output_card[3, 3]; label = "Save state", width = 112, height = 32)
    btn_load_state = Button(output_card[3, 4]; label = "Load state", width = 112, height = 32)
    btn_show_compare = Button(output_card[4, 1]; label = "Compare cube...", width = 138, height = 32)
    compare_path_box = Textbox(output_card[4, 2:4]; placeholder = "", width = 0, height = 32)
    btn_load_compare = Button(output_card[4, 5]; label = "", width = 0, height = 32)
    compare_mode_menu = Menu(output_card[4, 2:3]; options = ["A", "B", "A - B", "A / B", "resid z"], prompt = "B", width = 0)
    compare_state_label = Label(output_card[5, 1:5]; text = "Comparison: no cube loaded", halign = :left, tellwidth = false, fontsize = 13, color = ui_text_muted)
    foreach(c -> colsize!(output_card, c, Auto()), 1:5)

    region_card = control_card!(controls_grid, 2, 2, "Selection Spectrum"; rows = 4, cols = 4)
    region_mode_menu = Menu(region_card[2, 1]; options = ["point", "box", "circle"], prompt = "point", width = 112)
    region_clear_btn = Button(region_card[2, 2]; label = "Clear", width = 92, height = 32)
    region_count_label = Label(region_card[2, 3:4]; text = "0 px", halign = :left, tellwidth = false, fontsize = 14, color = ui_text_muted)
    spec_ymin_box = Textbox(region_card[3, 1]; placeholder = "y min", width = 92, height = 32)
    spec_ymax_box = Textbox(region_card[3, 2]; placeholder = "y max", width = 92, height = 32)
    spec_y_apply_btn = Button(region_card[3, 3]; label = "Apply y", width = 82, height = 32)
    spec_y_auto_btn = Button(region_card[3, 4]; label = "Auto y", width = 82, height = 32)
    foreach(c -> colsize!(region_card, c, Auto()), 1:4)

    contour_card = control_card!(controls_grid, 2, 3, "Contours"; rows = 3, cols = 5)
    contour_chk = Checkbox(contour_card[2, 1])
    Label(contour_card[2, 2]; text = "Show", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    contour_levels_box = Textbox(contour_card[2, 3:4]; placeholder = "auto or 1:red, 2:#00ffaa", width = compact_layout ? 170 : 190, height = 32)
    contour_apply_btn = Button(contour_card[2, 5]; label = "Apply", width = 82, height = 32)
    foreach(c -> colsize!(contour_card, c, Auto()), 1:5)

    # Bottom row of Analysis mode: Products + Histogram, centered in a sub-grid.
    # The middle spacer prevents the two cards from touching when fixed-width
    # controls inside either card reach the edge of their cell.
    analysis_bottom = controls_grid[3, 1:3] = GridLayout(; alignmode = Outside(0))
    colgap!(analysis_bottom, controls_gap)

    hist_card = control_card!(analysis_bottom, 1, 4, "Histogram"; rows = 5, cols = 5)
    hist_mode_menu = Menu(hist_card[2, 1]; options = ["bars", "kde"], prompt = String(hist_mode_obs[]), width = 96)
    hist_bins_box = Textbox(hist_card[2, 2]; placeholder = "bins", width = 76, height = 32)
    hist_apply_btn = Button(hist_card[3, 3]; label = "Apply x", width = 82, height = 32)
    hist_auto_btn = Button(hist_card[3, 4]; label = "Auto x", width = 82, height = 32)
    hist_xmin_box = Textbox(hist_card[3, 1]; placeholder = "x min", width = 92, height = 32)
    hist_xmax_box = Textbox(hist_card[3, 2]; placeholder = "x max", width = 92, height = 32)
    hist_ymin_box = Textbox(hist_card[4, 1]; placeholder = "y min", width = 92, height = 32)
    hist_ymax_box = Textbox(hist_card[4, 2]; placeholder = "y max", width = 92, height = 32)
    hist_y_apply_btn = Button(hist_card[4, 3]; label = "Apply y", width = 82, height = 32)
    hist_y_auto_btn = Button(hist_card[4, 4]; label = "Auto y", width = 82, height = 32)
    foreach(c -> colsize!(hist_card, c, Auto()), 1:5)

    anim_card = control_card!(controls_grid, 2, 3, "Animation"; rows = 4, cols = 5)
    start_box = Textbox(anim_card[2, 1]; placeholder = "start", width = 72, height = 32)
    stop_box  = Textbox(anim_card[2, 2]; placeholder = "stop",  width = 72, height = 32)
    step_box  = Textbox(anim_card[2, 3]; placeholder = "step",  width = 72, height = 32)
    fps_box   = Textbox(anim_card[2, 4]; placeholder = "fps",   width = 72, height = 32)
    play_btn = Button(anim_card[3, 1]; label = "Play", width = 78, height = 32)
    anim_btn = Button(anim_card[3, 2:3]; label = "Export GIF", width = 132, height = 32)
    loop_chk = Checkbox(anim_card[3, 4]); Label(anim_card[3, 5], text = "Loop", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    foreach(c -> colsize!(anim_card, c, Auto()), 1:5)

    display_card = control_card!(controls_grid, 2, 3, "Display"; rows = 5, cols = 4)
    Label(display_card[2, 1], text = "Colormap", halign = :left, tellwidth = false, fontsize = 14, color = ui_text_muted)
    cmap_menu = Menu(display_card[2, 2:4]; options = ui_colormap_options(), prompt = String(cmap), width = 156)
    invert_chk = Checkbox(display_card[3, 1]); Label(display_card[3, 2], text = "Invert", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    gauss_chk = Checkbox(display_card[3, 3]); Label(display_card[3, 4], text = "Smoothing", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    crosshair_chk = Checkbox(display_card[4, 1]); Label(display_card[4, 2], text = "Crosshair", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    marker_chk = Checkbox(display_card[4, 3]); Label(display_card[4, 4], text = "Selection", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    grid_chk = Checkbox(display_card[5, 1]); Label(display_card[5, 2], text = "Grid", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    pingpong_chk = Checkbox(display_card[5, 3]); Label(display_card[5, 4], text = "Ping-pong", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    foreach(c -> colsize!(display_card, c, Auto()), 1:4)

    moment_card = control_card!(analysis_bottom, 1, 2, "Products"; rows = 4, cols = 5)
    moment_menu = Menu(moment_card[2, 1]; options = ["M0 integrated", "M1 mean", "M2 dispersion"], prompt = "M0 integrated", width = compact_layout ? 124 : 138)
    btn_show_moment = Button(moment_card[2, 2]; label = "Show", width = compact_layout ? 72 : 82, height = 32)
    btn_show_slice = Button(moment_card[2, 3]; label = "Slice", width = compact_layout ? 72 : 82, height = 32)
    btn_moment_png = Button(moment_card[2, 4]; label = "PNG", width = compact_layout ? 64 : 74, height = 32)
    btn_moment_fits = Button(moment_card[2, 5]; label = "FITS", width = compact_layout ? 64 : 74, height = 32)
    fits_product_menu = Menu(moment_card[3, 1:2]; options = ["slice", "region", "moment", "filtered cube"], prompt = "slice", width = compact_layout ? 136 : 150)
    btn_save_fits = Button(moment_card[3, 3]; label = "Export FITS", width = compact_layout ? 104 : 118, height = 32)
    foreach(c -> colsize!(moment_card, c, Auto()), 1:5)

    # Finalise analysis_bottom: transparent Boxes force spacer columns to exist
    # so colsize! can address them. Cards live in cols 2 and 4.
    Box(analysis_bottom[1, 1]; color = :transparent, strokewidth = 0)
    Box(analysis_bottom[1, 3]; color = :transparent, strokewidth = 0)
    Box(analysis_bottom[1, 5]; color = :transparent, strokewidth = 0)
    colsize!(analysis_bottom, 1, Relative(1))
    colsize!(analysis_bottom, 2, Fixed(compact_layout ? 460 : 520))
    colsize!(analysis_bottom, 3, Fixed(compact_layout ? 28 : 36))
    colsize!(analysis_bottom, 4, Fixed(compact_layout ? 430 : 500))
    colsize!(analysis_bottom, 5, Relative(1))

    foreach(c -> colsize!(controls_grid, c, Relative(1 / 3)), 1:3)
    rowsize!(controls_grid, 1, Fixed(controls_row_heights[1]))
    rowsize!(controls_grid, 2, Fixed(controls_row_heights[2]))
    rowsize!(controls_grid, 3, Fixed(controls_row_heights[3]))

    style_checkbox!(pingpong_chk)
    style_checkbox!(loop_chk)
    style_checkbox!(invert_chk)
    style_checkbox!(gauss_chk)
    style_checkbox!(crosshair_chk)
    style_checkbox!(marker_chk)
    style_checkbox!(grid_chk)
    style_menu!(img_scale_menu)
    style_menu!(spec_scale_menu)
    style_menu!(cmap_menu)
    style_menu!(ps_src_menu)
    style_menu!(ps_win_menu)
    style_menu!(ps_unit_menu)
    style_menu!(fmt_menu)
    style_menu!(compare_mode_menu)
    style_menu!(axis_menu)
    style_menu!(moment_menu)
    style_menu!(fits_product_menu)
    style_textbox!(fname_box)
    style_textbox!(compare_path_box)
    style_textbox!(start_box)
    style_textbox!(stop_box)
    style_textbox!(step_box)
    style_textbox!(fps_box)
    style_textbox!(clim_min_box)
    style_textbox!(clim_max_box)
    style_textbox!(ps_kmin_box)
    style_textbox!(ps_kmax_box)
    style_button!(mode_nav_btn)
    style_button!(mode_analysis_btn)
    style_button!(mode_export_btn)
    style_button!(reset_zoom_btn)
    style_button!(reset_zoom_analysis_btn)
    style_button!(reset_zoom_export_btn)
    style_button!(ps_btn)
    style_button!(base_layout_btn)
    style_button!(ps_refresh_btn)
    style_button!(ps_popout_btn)
    style_checkbox!(ps_pad_chk)
    style_checkbox!(ps_nanapo_chk)
    style_checkbox!(ps_fit_chk)
    style_button!(btn_save_img)
    style_button!(btn_save_spec)
    style_button!(btn_save_state)
    style_button!(btn_load_state)
    style_button!(btn_show_compare)
    style_button!(btn_load_compare)
    style_button!(play_btn)
    style_button!(anim_btn)
    style_button!(clim_apply_btn)
    style_button!(clim_auto_btn)
    style_button!(clim_p1_btn)
    style_button!(clim_p5_btn)
    style_menu!(region_mode_menu)
    style_menu!(hist_mode_menu)
    style_button!(region_clear_btn)
    style_checkbox!(contour_chk)
    style_textbox!(contour_levels_box)
    style_button!(contour_apply_btn)
    style_textbox!(spec_ymin_box)
    style_textbox!(spec_ymax_box)
    style_button!(spec_y_apply_btn)
    style_button!(spec_y_auto_btn)
    style_textbox!(hist_bins_box)
    style_textbox!(hist_xmin_box)
    style_textbox!(hist_xmax_box)
    style_textbox!(hist_ymin_box)
    style_textbox!(hist_ymax_box)
    style_button!(hist_apply_btn)
    style_button!(hist_auto_btn)
    style_button!(hist_y_apply_btn)
    style_button!(hist_y_auto_btn)
    style_button!(btn_show_moment)
    style_button!(btn_show_slice)
    style_button!(btn_moment_png)
    style_button!(btn_moment_fits)
    style_button!(btn_save_fits)
    style_slider!(slice_slider)
    style_slider!(sigma_slider)

    if compact_layout
        for btn in (reset_zoom_btn, reset_zoom_analysis_btn, reset_zoom_export_btn,
                    ps_btn, base_layout_btn, ps_refresh_btn, ps_popout_btn,
                    btn_save_img, btn_save_spec, btn_save_state, btn_load_state,
                    btn_show_compare, btn_load_compare, play_btn, anim_btn, clim_apply_btn,
                    clim_auto_btn, clim_p1_btn, clim_p5_btn, region_clear_btn, contour_apply_btn,
                    spec_y_apply_btn, spec_y_auto_btn, hist_apply_btn, hist_auto_btn, hist_y_apply_btn, hist_y_auto_btn,
                    btn_show_moment, btn_show_slice, btn_moment_png, btn_moment_fits, btn_save_fits)
            btn.height[] = 30
            btn.fontsize[] = 13
            btn.padding[] = (9, 9, 5, 5)
        end
        for menu in (img_scale_menu, spec_scale_menu, cmap_menu, ps_src_menu, ps_win_menu,
                     ps_unit_menu, fmt_menu, compare_mode_menu, axis_menu, region_mode_menu, hist_mode_menu,
                     moment_menu, fits_product_menu)
            menu.height[] = 30
            menu.fontsize[] = 13
            menu.textpadding[] = (8, 8, 5, 5)
            menu.dropdown_arrow_size[] = 10
        end
        for tb in (ps_kmin_box, ps_kmax_box, clim_min_box, clim_max_box, fname_box,
                   compare_path_box, contour_levels_box, spec_ymin_box, spec_ymax_box,
                   hist_bins_box, hist_xmin_box, hist_xmax_box, hist_ymin_box, hist_ymax_box,
                   start_box, stop_box, step_box, fps_box)
            tb.height[] = 30
            tb.fontsize[] = 13
            tb.textpadding[] = (8, 8, 5, 5)
        end
        for chk in (ps_pad_chk, ps_nanapo_chk, ps_fit_chk, pingpong_chk, loop_chk, invert_chk,
                    gauss_chk, crosshair_chk, marker_chk, grid_chk, contour_chk)
            chk.size[] = 18
            chk.checkmarksize[] = 0.58
        end
        for sl in (slice_slider, sigma_slider)
            sl.height[] = 20
            sl.linewidth[] = 8
        end
    end

    invert_chk.checked[] = invert_cmap[]
    cmap_menu.selection[] = String(cmap_name[])
    gauss_chk.checked[] = gauss_on[]
    crosshair_chk.checked[] = show_crosshair[]
    marker_chk.checked[] = show_marker[]
    grid_chk.checked[] = show_grid[]
    contour_chk.checked[] = show_contours[]
    loop_chk.checked[] = true
    hint_label = Label(
        main_grid[3, 2];
        text      = "arrows: move crosshair    left-click: pick / draw region    right-drag: zoom    i: invert colormap",
        halign    = :right,
        fontsize  = 13,
        color     = ui_text_muted,
        tellwidth = false,
    )
    ui_status = Observable(" ")
    status_footer_label = Label(
        main_grid[4, 1:2];
        text = ui_status,
        halign = :left,
        tellwidth = false,
    )
    if compact_layout
        hint_label.visible[] = false
        status_footer_label.visible[] = false
    end
    # ---------- Helpers ----------
    set_status!(msg::AbstractString) = (ui_status[] = String(msg); nothing)
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
    if spec_ylimits_source[] !== :auto
        lo, hi = spec_ylimits_value[]
        set_box_text!(spec_ymin_box, string(lo))
        set_box_text!(spec_ymax_box, string(hi))
    end

    set_block_visible!(block, visible::Bool) = begin
        try
            block.visible[] = visible
        catch
        end
        try
            block.scene.visible[] = visible
        catch
        end
        try
            block.blockscene.visible[] = visible
        catch
        end
        nothing
    end

    function set_layout_contents_visible!(layout, visible::Bool)
        for block in try
            contents(layout)
        catch
            Any[]
        end
            set_block_visible!(block, visible)
            block isa GridLayout && set_layout_contents_visible!(block, visible)
        end
        nothing
    end

    nav_cards = (view_card, slice_card, display_card)
    analysis_cards = (contrast_card, region_card, contour_card, hist_card, moment_card)
    export_cards = (output_card, anim_card)

    function set_mode_button_active!(btn, active::Bool)
        btn.buttoncolor[] = active ? ui_theme.surface_active : ui_theme.surface
        btn.buttoncolor_hover[] = active ? ui_theme.surface_active : ui_theme.surface_hover
        btn.labelcolor[] = active ? ui_accent_strong : ui_text
        btn.labelcolor_hover[] = ui_accent_strong
        btn.strokecolor[] = active ? ui_accent : ui_border
        nothing
    end

    function refresh_control_mode!()
        mode = control_mode[]
        for card in nav_cards
            set_layout_contents_visible!(card, mode === :navigation)
        end
        for card in analysis_cards
            set_layout_contents_visible!(card, mode === :analysis)
        end
        for card in export_cards
            set_layout_contents_visible!(card, mode === :export)
        end
        set_mode_button_active!(mode_nav_btn, mode === :navigation)
        set_mode_button_active!(mode_analysis_btn, mode === :analysis)
        set_mode_button_active!(mode_export_btn, mode === :export)
        nothing
    end
    refresh_control_mode!()

    # ---------- Mode-gated event helper ----------
    # Cards from different modes share the same grid cell, so even when
    # `set_block_visible!` hides a card, the underlying Buttons / Checkboxes /
    # Menus / Sliders are still pickable and forward clicks to their listeners.
    # `on_mode` wraps `on(obs)` so the callback fires only when the active mode
    # matches, neutralising leak-through clicks (e.g. clicking a Contours
    # button while in :analysis mode used to also trigger the hidden GIF
    # button stacked underneath).
    # `bypass_mode_gate` is a re-entrant escape hatch for routines that
    # programmatically poke widgets across modes (load state, etc.).
    bypass_mode_gate = Ref(false)
    on_mode(callback, obs, mode::Symbol) = on(obs) do v
        (bypass_mode_gate[] || control_mode[] === mode) || return
        callback(v)
    end

    set_block_visible!(ax_cmp, false)

    function show_compare_loader!()
        btn_show_compare.label[] = ""
        btn_show_compare.width[] = 0
        compare_mode_menu.width[] = 0
        compare_path_box.placeholder[] = "second cube FITS path"
        compare_path_box.width[] = 310
        btn_load_compare.label[] = "Load cube"
        btn_load_compare.width[] = 104
        compare_state_label.text[] = "Comparison: waiting for cube path"
        compare_state_label.color[] = ui_text_muted
        set_status!("Enter the second cube FITS path, then click Load cube.")
        nothing
    end

    function hide_compare_loader!()
        btn_show_compare.label[] = ""
        btn_show_compare.width[] = 0
        compare_path_box.placeholder[] = ""
        compare_path_box.width[] = 0
        btn_load_compare.label[] = ""
        btn_load_compare.width[] = 0
        compare_mode_menu.width[] = compare_visible[] ? 150 : 0
        if !compare_visible[]
            btn_show_compare.label[] = "Compare cube..."
            btn_show_compare.width[] = 138
            compare_state_label.text[] = "Comparison: no cube loaded"
            compare_state_label.color[] = ui_text_muted
        end
        nothing
    end

    function resolve_compare_path(path_txt::AbstractString)
        p = strip(String(path_txt))
        isempty(p) && return ""
        isfile(p) && return p
        beside_primary = joinpath(dirname(abspath(filepath)), p)
        isfile(beside_primary) && return beside_primary
        return p
    end

    # Native OS file picker. Tries platform-appropriate tools without
    # adding a Julia dep. Returns "" when the user cancels or no backend
    # is available — the caller then falls back to the legacy textbox loader.
    function pick_compare_path()
        initial = filepath != "" ? dirname(abspath(filepath)) : pwd()
        out = ""
        try
            if Sys.isapple()
                script = "POSIX path of (choose file with prompt \"Select FITS cube to compare\" default location POSIX file \"$(initial)\")"
                out = readchomp(`osascript -e $script`)
            elseif Sys.islinux()
                zenity = Sys.which("zenity")
                if zenity !== nothing
                    out = readchomp(`$zenity --file-selection --title=Select\ FITS\ cube\ to\ compare --filename=$(initial)/`)
                else
                    kdialog = Sys.which("kdialog")
                    if kdialog !== nothing
                        out = readchomp(`$kdialog --getopenfilename $initial "FITS (*.fits *.fits.gz)"`)
                    end
                end
            elseif Sys.iswindows()
                ps = """
                Add-Type -AssemblyName System.Windows.Forms | Out-Null;
                \$f = New-Object System.Windows.Forms.OpenFileDialog;
                \$f.Filter = 'FITS (*.fits;*.fits.gz)|*.fits;*.fits.gz|All files (*.*)|*.*';
                \$f.InitialDirectory = '$initial';
                if (\$f.ShowDialog() -eq 'OK') { Write-Output \$f.FileName }
                """
                out = readchomp(`powershell -NoProfile -Command $ps`)
            end
        catch e
            @debug "Native file dialog failed; falling back to textbox." exception=e
            return ""
        end
        return strip(String(out))
    end

    # Lightweight WCS compatibility check: when both cubes declare a WCS on a
    # given axis, CTYPE base / CRVAL / CRPIX / CDELT must agree (relative
    # tolerance 1e-6). Axes where neither cube has WCS are accepted. Axes
    # where only one cube has WCS are rejected — that asymmetry would break
    # tick formatting and world-coordinate readouts on `ax_cmp`.
    wcs_axes_compatible(a::AbstractVector, b::AbstractVector) = begin
        length(a) == length(b) || return false
        for (xa, xb) in zip(a, b)
            if !xa.available && !xb.available
                continue
            end
            xa.available == xb.available || return false
            xa.ctype_base == xb.ctype_base || return false
            isapprox(xa.crval, xb.crval; rtol = 1e-6, atol = 0.0) || return false
            isapprox(xa.cdelt, xb.cdelt; rtol = 1e-6, atol = 0.0) || return false
            isapprox(xa.crpix, xb.crpix; rtol = 0.0, atol = 1e-6) || return false
        end
        return true
    end

    function load_compare_cube!(path_txt::AbstractString)
        cmp_path = resolve_compare_path(path_txt)
        if isempty(cmp_path)
            set_status!("Provide a second cube FITS path before enabling dual view.")
            return false
        end
        if !isfile(cmp_path)
            set_status!("Second cube not found: $(cmp_path)")
            return false
        end
        raw_cmp, cmp_header = try
            FITS(cmp_path) do f
                (read(f[1]), read_header(f[1]))
            end
        catch e
            msg = "Failed to read second cube: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
            return false
        end
        if ndims(raw_cmp) != 3
            set_status!("Second cube must be 3D, got ndims=$(ndims(raw_cmp)).")
            return false
        end
        if size(raw_cmp) != siz
            set_status!("Second cube size $(size(raw_cmp)) does not match primary cube size $(siz).")
            return false
        end
        # WCS compatibility check — required since axes/colorbar/spectrum
        # ticks on `ax_cmp` reuse the primary cube's `wcs`.
        cmp_wcs = try
            read_simple_wcs(cmp_header, 3)
        catch
            SimpleWCSAxis[]
        end
        if !wcs_axes_compatible(wcs, cmp_wcs)
            set_status!("Second cube WCS does not match primary cube WCS.")
            return false
        end
        compare_data[] = Float32.(raw_cmp)
        compare_name[] = String(replace(basename(cmp_path), r"\.fits(\.gz)?$" => ""))
        compare_visible[] = true
        colsize!(img_grid, 2, Auto())
        colsize!(img_grid, 4, Auto())
        set_block_visible!(ax_cmp, true)
        set_block_visible!(img_colorbar_cmp, true)
        ax_cmp.xgridvisible[] = show_grid[]
        ax_cmp.ygridvisible[] = show_grid[]
        autolimits!(ax_cmp)
        hide_compare_loader!()
        compare_state_label.text[] = "Comparison: cube loaded ($(compare_name[]))"
        compare_state_label.color[] = ui_success
        # Pull the spectrum line in immediately so the user sees overlay
        # without having to click anything.
        refresh_spectrum!()
        set_status!("Comparison cube loaded: $(cmp_path).")
        return true
    end

    function world_info_string()
        any(has_wcs(wcs, dim) for dim in 1:3) || return ""
        coords = (
            format_world_coord(wcs, 1, i_idx[]),
            format_world_coord(wcs, 2, j_idx[]),
            format_world_coord(wcs, 3, k_idx[]),
        )
        return join(coords, ", ")
    end

    function selection_info_tex()
        if isempty(region_uvs[])
            val = data[i_idx[], j_idx[], k_idx[]]
            base = String(make_info_tex(i_idx[], j_idx[], k_idx[], u_idx[], v_idx[], val))
            winfo = world_info_string()
            isempty(winfo) && return latexstring(base)
            return latexstring(base, "\\quad\\mathbf{WCS}\\,(", latex_safe(winfo), ")")
        else
            npx = length(region_uvs[])
            y = mean_region_spectrum(data, axis[], region_uvs[])
            chan = clamp(idx[], 1, length(y))
            val = y[chan]
            shape = region_shape[] === :circle ? "circle" : "box"
            return latexstring(
                "\\mathbf{region}\\,\\text{", shape, "}\\quad\\mathbf{pixels}=",
                npx,
                "\\quad\\mathbf{slice\\ mean}= ",
                isnan(val) ? "NaN" : string(round(Float32(val); digits = 4)),
            )
        end
    end

    function clear_region!()
        region_uvs[] = Tuple{Int,Int}[]
        region_start[] = Point2f(NaN32, NaN32)
        region_end[] = Point2f(NaN32, NaN32)
        region_drag_active[] = false
        region_count_label.text[] = "0 px"
        nothing
    end

    function update_region_from_drag!(p0::Point2f, p1::Point2f)
        u_max, v_max = slice_dims(axis[])
        uv = region_uv_indices(u_max, v_max, p0[1], p0[2], p1[1], p1[2], region_shape[])
        region_uvs[] = uv
        region_count_label.text[] = "$(length(uv)) px"
        if isempty(uv)
            set_status!("Selection canceled: draw a larger $(String(region_shape[])).")
        else
            set_status!("Selection spectrum averaged over $(length(uv)) pixels.")
        end
        nothing
    end

    function apply_percentile_clims!(lo::Real, hi::Real)
        parsed_clims = percentile_clims(slice_disp[], lo, hi)
        clims_manual[] = parsed_clims
        use_manual[] = true
        set_box_text!(clim_min_box, string(first(parsed_clims)))
        set_box_text!(clim_max_box, string(last(parsed_clims)))
        if spec_ylimits_source[] === :contrast
            spec_ylimits_value[] = parsed_clims
            set_box_text!(spec_ymin_box, string(first(parsed_clims)))
            set_box_text!(spec_ymax_box, string(last(parsed_clims)))
            refresh_spec_ylim!()
        end
        set_status!("Contrast set to p$(lo)-p$(hi).")
        nothing
    end

    function refresh_uv!()
        a = axis[]
        u_max, v_max = slice_dims(a)
        u, v = ijk_to_uv(i_idx[], j_idx[], k_idx[], a)
        u = clamp(u, 1, u_max)
        v = clamp(v, 1, v_max)
        u_idx[] = u; v_idx[] = v
        uv_point[] = Point2f(v, u)
    end

    function refresh_labels!()
        lab_info.text[] = selection_info_tex()
        status_label.text[] = latexstring("\\text{axis } $(axis[]),\\, \\text{index } $(idx[])")
    end

    function refresh_spec_ylim!()
        x_max = Float32(max(0, length(spec_y_buf) - 1))
        if spec_ylimits_source[] === :manual || spec_ylimits_source[] === :contrast
            vmin_, vmax_ = spec_ylimits_value[]
            limits!(ax_spec, nothing, nothing, vmin_, vmax_)
            xlims!(ax_spec, 0f0, x_max)
        else
            autolimits!(ax_spec)
            xlims!(ax_spec, 0f0, x_max)
        end
    end

    function refresh_spectrum!()
        if !isempty(region_uvs[])
            spec_x_raw[] = spec_x_axes[axis[]]
            y = mean_region_spectrum(data, axis[], region_uvs[])
            resize!(spec_y_buf, length(y))
            copyto!(spec_y_buf, y)
            ax_spec.title[] = latexstring("\\text{Mean spectrum in selected region}")
        elseif axis[] == 1
            spec_x_raw[] = spec_x_axes[1]
            resize!(spec_y_buf, siz[1])
            @views copyto!(spec_y_buf, data[:, j_idx[], k_idx[]])
            ax_spec.title[] = L"\text{Spectrum at selected pixel}"
        elseif axis[] == 2
            spec_x_raw[] = spec_x_axes[2]
            resize!(spec_y_buf, siz[2])
            @views copyto!(spec_y_buf, data[i_idx[], :, k_idx[]])
            ax_spec.title[] = L"\text{Spectrum at selected pixel}"
        else
            spec_x_raw[] = spec_x_axes[3]
            resize!(spec_y_buf, siz[3])
            @views copyto!(spec_y_buf, data[i_idx[], j_idx[], :])
            ax_spec.title[] = L"\text{Spectrum at selected pixel}"
        end
        spec_y_raw[] = spec_y_buf
        # Mirror the cube-A extraction on the comparison cube when loaded.
        # Same axis, same voxel / region — only the source array changes.
        cmp = compare_data[]
        if cmp !== nothing
            if !isempty(region_uvs[])
                y2 = mean_region_spectrum(cmp, axis[], region_uvs[])
                resize!(spec_y_compare_buf, length(y2))
                copyto!(spec_y_compare_buf, y2)
            elseif axis[] == 1
                resize!(spec_y_compare_buf, siz[1])
                @views copyto!(spec_y_compare_buf, cmp[:, j_idx[], k_idx[]])
            elseif axis[] == 2
                resize!(spec_y_compare_buf, siz[2])
                @views copyto!(spec_y_compare_buf, cmp[i_idx[], :, k_idx[]])
            else
                resize!(spec_y_compare_buf, siz[3])
                @views copyto!(spec_y_compare_buf, cmp[i_idx[], j_idx[], :])
            end
        else
            resize!(spec_y_compare_buf, length(spec_y_buf))
            fill!(spec_y_compare_buf, NaN32)
        end
        spec_y_compare_raw[] = spec_y_compare_buf
        refresh_spec_ylim!()
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

    function refresh_axis_labels!()
        xlab, ylab = slice_axis_labels(axis[])
        ax_img.xlabel[] = xlab
        ax_img.ylabel[] = ylab
        ax_cmp.xlabel[] = xlab
        ax_cmp.ylabel[] = ylab
        u_dim, v_dim = slice_axis_dims(axis[])
        ax_img.xtickformat[] = pixel_world_tick_formatter(v_dim)
        ax_img.ytickformat[] = pixel_world_tick_formatter(u_dim)
        ax_cmp.xtickformat[] = pixel_world_tick_formatter(v_dim)
        ax_cmp.ytickformat[] = pixel_world_tick_formatter(u_dim)
    end

    refresh_all!() = (refresh_axis_labels!(); refresh_uv!(); refresh_labels!(); refresh_spectrum!())

    # ---------- Reactivity ----------
    on(clims_obs) do (cmin, cmax)
        if spec_ylimits_source[] === :contrast
            spec_ylimits_value[] = (Float32(cmin), Float32(cmax))
            refresh_spec_ylim!()
        end
    end

    on(spec_scale_mode) do _
        refresh_spec_ylim!()
    end

    reset_zoom!() = begin
        autolimits!(ax_img)
        compare_visible[] && autolimits!(ax_cmp)
        nothing
    end

    on_mode(reset_zoom_btn.clicks, :navigation) do _
        reset_zoom!()
    end
    on_mode(reset_zoom_analysis_btn.clicks, :analysis) do _
        reset_zoom!()
    end
    on_mode(reset_zoom_export_btn.clicks, :export) do _
        reset_zoom!()
    end

    # ---------- UI callbacks ----------
    syncing_slice_controls = Ref(false)

    on(mode_nav_btn.clicks) do _
        control_mode[] = :navigation
        refresh_control_mode!()
    end
    on(mode_analysis_btn.clicks) do _
        control_mode[] = :analysis
        refresh_control_mode!()
    end
    on(mode_export_btn.clicks) do _
        control_mode[] = :export
        refresh_control_mode!()
    end

    # Keep the slice slider synced to the active axis (range + knob position)
    on_mode(axis_menu.selection, :navigation) do sel
        sel === nothing && return
        new_axis = findfirst(==(String(sel)), axes_labels)
        new_axis === nothing && return
        axis[] = new_axis
        new_range = 1:siz[new_axis]
        syncing_slice_controls[] = true
        slice_slider.range[] = new_range
        idx[] = clamp(idx[], first(new_range), last(new_range))
        slice_slider.value[] = idx[]  # move the thumb if the old value was out of bounds
        syncing_slice_controls[] = false
        ii, jj, kk = uv_to_ijk(u_idx[], v_idx[], axis[], idx[])
        i_idx[] = clamp(ii, 1, siz[1]); j_idx[] = clamp(jj, 1, siz[2]); k_idx[] = clamp(kk, 1, siz[3])
        clear_region!()
        refresh_all!()
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    on_mode(slice_slider.value, :navigation) do v
        syncing_slice_controls[] && return
        idx[] = Int(round(v))
        ii, jj, kk = uv_to_ijk(u_idx[], v_idx[], axis[], idx[])
        i_idx[] = clamp(ii, 1, siz[1]); j_idx[] = clamp(jj, 1, siz[2]); k_idx[] = clamp(kk, 1, siz[3])
        refresh_labels!(); refresh_spectrum!()
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    on_mode(img_scale_menu.selection, :navigation) do sel
        sel === nothing && return
        img_scale_mode[] = Symbol(sel)
    end

    on_mode(spec_scale_menu.selection, :navigation) do sel
        sel === nothing && return
        spec_scale_mode[] = Symbol(sel)
    end

    on_mode(cmap_menu.selection, :navigation) do sel
        sel === nothing && return
        cmap_name[] = Symbol(sel)
        set_status!("Colormap set to $(String(sel)).")
    end

    on_mode(hist_mode_menu.selection, :analysis) do sel
        sel === nothing && return
        hist_mode_obs[] = normalize_histogram_mode(sel)
        set_status!("Histogram mode set to $(String(hist_mode_obs[])).")
    end

    on_mode(hist_apply_btn.clicks, :analysis) do _
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

    on_mode(hist_auto_btn.clicks, :analysis) do _
        hist_xlimits_manual[] = false
        set_box_text!(hist_xmin_box, "")
        set_box_text!(hist_xmax_box, "")
        set_status!("Automatic histogram x-axis enabled.")
    end

    on_mode(hist_y_auto_btn.clicks, :analysis) do _
        hist_ylimits_manual[] = false
        set_box_text!(hist_ymin_box, "")
        set_box_text!(hist_ymax_box, "")
        refresh_hist_axes!()
        set_status!("Automatic histogram y-axis enabled.")
    end

    on_mode(hist_y_apply_btn.clicks, :analysis) do _
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

    on_mode(spec_y_apply_btn.clicks, :analysis) do _
        ok, manual, ylim, msg = parse_spectrum_ylimits(
            get_box_str(spec_ymin_box),
            get_box_str(spec_ymax_box);
            fallback = spec_ylimits_value[],
        )
        set_status!(msg)
        ok || return
        if manual
            spec_ylimits_value[] = ylim
            spec_ylimits_source[] = :manual
            set_box_text!(spec_ymin_box, string(first(ylim)))
            set_box_text!(spec_ymax_box, string(last(ylim)))
        else
            spec_ylimits_source[] = :auto
            set_box_text!(spec_ymin_box, "")
            set_box_text!(spec_ymax_box, "")
        end
        refresh_spec_ylim!()
    end

    on_mode(spec_y_auto_btn.clicks, :analysis) do _
        spec_ylimits_source[] = :auto
        set_box_text!(spec_ymin_box, "")
        set_box_text!(spec_ymax_box, "")
        refresh_spec_ylim!()
        set_status!("Automatic spectrum y-axis enabled.")
    end

    on(hist_limits_obs) do _
        refresh_hist_axes!()
    end
    on(hist_y_obs) do _
        refresh_hist_axes!()
    end
    on(compare_hist_y_obs) do _
        refresh_hist_axes!()
    end

    on_mode(invert_chk.checked, :navigation) do v
        invert_cmap[] = v
    end

    on_mode(gauss_chk.checked, :navigation) do v
        gauss_on[] = v
        refresh_spectrum!()
    end

    on_mode(crosshair_chk.checked, :navigation) do v
        show_crosshair[] = v
    end

    on_mode(marker_chk.checked, :navigation) do v
        show_marker[] = v
    end

    on_mode(grid_chk.checked, :navigation) do v
        show_grid[] = v
        ax_img.xgridvisible[] = v
        ax_img.ygridvisible[] = v
        ax_cmp.xgridvisible[] = v
        ax_cmp.ygridvisible[] = v
        ax_spec.xgridvisible[] = v
        ax_spec.ygridvisible[] = v
    end

    on_mode(btn_show_compare.clicks, :export) do _
        # Try a native file picker first. If it returns a path we load
        # immediately; otherwise we fall back to the legacy textbox loader
        # (useful in headless CI where no dialog backend is available, or
        # when the user cancels and wants to type a path).
        picked = pick_compare_path()
        if !isempty(picked) && isfile(picked)
            set_box_text!(compare_path_box, picked)
            load_compare_cube!(picked)
        else
            show_compare_loader!()
        end
    end

    on_mode(btn_load_compare.clicks, :export) do _
        btn_load_compare.width[] <= 0 && return
        load_compare_cube!(get_box_str(compare_path_box))
    end

    on_mode(compare_mode_menu.selection, :export) do sel
        sel === nothing && return
        label = String(sel)
        compare_mode[] = label == "A" ? :A :
            label == "B" ? :B :
            label == "A - B" ? :diff :
            label == "A / B" ? :ratio :
            :residuals
        set_status!("Dual product set to $(compare_mode_label(compare_mode[])).")
    end

    on_mode(sigma_slider.value, :navigation) do v
        sigma[] = Float32(v)
        sigma_label.text[] = latexstring("\\sigma = $(round(v; digits = 2))\\,\\text{px}")
    end

    on_mode(clim_apply_btn.clicks, :analysis) do _
        txtmin = get_box_str(clim_min_box)
        txtmax = get_box_str(clim_max_box)
        ok, new_manual, parsed_clims, msg = parse_manual_clims(txtmin, txtmax; fallback = clims_manual[])
        set_status!(msg)
        if !ok
            @warn "Could not apply contrast limits" txtmin txtmax msg
            return
        end
        if new_manual
            clims_manual[] = parsed_clims
            use_manual[] = true
            if spec_ylimits_source[] === :contrast
                spec_ylimits_value[] = parsed_clims
                set_box_text!(spec_ymin_box, string(first(parsed_clims)))
                set_box_text!(spec_ymax_box, string(last(parsed_clims)))
            end
            refresh_spec_ylim!()
            set_box_text!(clim_min_box, string(first(parsed_clims)))
            set_box_text!(clim_max_box, string(last(parsed_clims)))
        else
            use_manual[] = false
            if spec_ylimits_source[] === :contrast
                spec_ylimits_source[] = :auto
                set_box_text!(spec_ymin_box, "")
                set_box_text!(spec_ymax_box, "")
            end
            refresh_spec_ylim!()
        end
    end

    on_mode(clim_auto_btn.clicks, :analysis) do _
        use_manual[] = false
        set_box_text!(clim_min_box, "")
        set_box_text!(clim_max_box, "")
        if spec_ylimits_source[] === :contrast
            spec_ylimits_source[] = :auto
            set_box_text!(spec_ymin_box, "")
            set_box_text!(spec_ymax_box, "")
        end
        refresh_spec_ylim!()
        set_status!("Automatic contrast enabled.")
    end

    on_mode(clim_p1_btn.clicks, :analysis) do _
        apply_percentile_clims!(1, 99)
    end

    on_mode(clim_p5_btn.clicks, :analysis) do _
        apply_percentile_clims!(5, 95)
    end

    on_mode(region_mode_menu.selection, :analysis) do sel
        sel === nothing && return
        mode = Symbol(String(sel))
        if mode in (:point, :box, :circle)
            selection_mode[] = mode
            region_shape[] = mode === :circle ? :circle : :box
            if mode === :point
                clear_region!()
                refresh_labels!()
                refresh_spectrum!()
            else
                set_status!("Draw a $(String(mode)) with left drag on the image.")
            end
        end
    end

    on_mode(region_clear_btn.clicks, :analysis) do _
        clear_region!()
        refresh_labels!()
        refresh_spectrum!()
        set_status!("Selection cleared; spectrum follows the selected pixel.")
    end

    on_mode(contour_chk.checked, :analysis) do v
        show_contours[] = v
        set_status!(v ? "Contours enabled." : "Contours hidden.")
    end

    on_mode(contour_apply_btn.clicks, :analysis) do _
        ok, use_man, levels, colors, msg = parse_contour_specs(
            get_box_str(contour_levels_box);
            fallback_levels = contour_manual_levels[],
            fallback_colors = contour_manual_colors[],
        )
        set_status!(msg)
        if !ok
            @warn "Could not apply contour levels" msg
            return
        end
        contour_use_manual[] = use_man
        contour_manual_levels[] = levels
        contour_manual_colors[] = colors
        if use_man
            set_box_text!(contour_levels_box, format_contour_specs(levels, colors))
        else
            set_box_text!(contour_levels_box, "")
        end
        show_contours[] = true
        contour_chk.checked[] = true
    end

    on_mode(moment_menu.selection, :analysis) do sel
        sel === nothing && return
        label = String(sel)
        moment_order[] = startswith(label, "M1") ? 1 : startswith(label, "M2") ? 2 : 0
        view_product[] === :moment && set_status!("Showing $(label) along axis $(axis[]).")
    end

    on_mode(btn_show_moment.clicks, :analysis) do _
        view_product[] = :moment
        use_manual[] = false
        autolimits!(ax_spec)
        xlims!(ax_spec, 0f0, Float32(max(0, length(spec_y_buf) - 1)))
        autolimits!(ax_img)
        set_status!("Moment map displayed along axis $(axis[]).")
    end

    on_mode(btn_show_slice.clicks, :analysis) do _
        view_product[] = :slice
        use_manual[] = false
        autolimits!(ax_img)
        set_status!("Slice view restored.")
    end

    # Keyboard navigation (+ invert)
    on(events(fig).keyboardbutton) do ev
        ev.action == Keyboard.press || return
        u_max, v_max = slice_dims(axis[])
        if ev.key == Keyboard.i
            invert_cmap[] = !invert_cmap[]
        elseif ev.key == Keyboard.left
            v_idx[] = max(1, v_idx[] - 1)
        elseif ev.key == Keyboard.right
            v_idx[] = min(v_max, v_idx[] + 1)
        elseif ev.key == Keyboard.up
            u_idx[] = min(u_max, u_idx[] + 1)
        elseif ev.key == Keyboard.down
            u_idx[] = max(1, u_idx[] - 1)
        else
            return
        end
        ii, jj, kk = uv_to_ijk(u_idx[], v_idx[], axis[], idx[])
        i_idx[] = clamp(ii, 1, siz[1]); j_idx[] = clamp(jj, 1, siz[2]); k_idx[] = clamp(kk, 1, siz[3])
        refresh_labels!(); refresh_spectrum!()
        uv_point[] = Point2f(v_idx[], u_idx[])
    end

    # Mouse pick
    on(events(ax_img).mousebutton) do ev
        if ev.button == Mouse.right && ev.action == Mouse.press
            p = mouseposition(ax_img)
            if any(isnan, p)
                return
            end
            zoom_drag_start[] = Point2f(p[1], p[2])
            zoom_drag_end[] = Point2f(p[1], p[2])
            zoom_drag_active[] = true
            set_status!("Zoom box: right-drag and release to apply.")
            return
        elseif ev.button == Mouse.right && ev.action == Mouse.release
            if !zoom_drag_active[]
                return
            end
            p = mouseposition(ax_img)
            if !any(isnan, p)
                zoom_drag_end[] = Point2f(p[1], p[2])
            end
            p0 = zoom_drag_start[]
            p1 = zoom_drag_end[]
            zoom_drag_active[] = false
            zoom_drag_start[] = Point2f(NaN32, NaN32)
            zoom_drag_end[] = Point2f(NaN32, NaN32)
            if !(isfinite(p0[1]) && isfinite(p0[2]) && isfinite(p1[1]) && isfinite(p1[2]))
                return
            end
            x0, y0 = p0
            x1, y1 = p1
            xmin, xmax = minmax(x0, x1)
            ymin, ymax = minmax(y0, y1)
            if abs(xmax - xmin) < 1e-3 || abs(ymax - ymin) < 1e-3
                set_status!("Zoom canceled: draw a larger rectangle.")
                return
            end
            limits!(ax_img, xmin, xmax, ymin, ymax)
            compare_visible[] && limits!(ax_cmp, xmin, xmax, ymin, ymax)
            set_status!("Zoom applied.")
            return
        elseif ev.button == Mouse.left && ev.action == Mouse.press && selection_mode[] != :point
            p = mouseposition(ax_img)
            if any(isnan, p)
                return
            end
            u_max, v_max = slice_dims(axis[])
            p0 = Point2f(clamp(p[1], 1, v_max), clamp(p[2], 1, u_max))
            region_start[] = p0
            region_end[] = p0
            region_drag_active[] = true
            region_uvs[] = Tuple{Int,Int}[]
            region_count_label.text[] = "0 px"
            set_status!("Drawing $(String(selection_mode[])) region.")
            return
        elseif ev.button == Mouse.left && ev.action == Mouse.release && region_drag_active[]
            p = mouseposition(ax_img)
            u_max, v_max = slice_dims(axis[])
            if !any(isnan, p)
                region_end[] = Point2f(clamp(p[1], 1, v_max), clamp(p[2], 1, u_max))
            end
            p0 = region_start[]
            p1 = region_end[]
            region_drag_active[] = false
            if !(isfinite(p0[1]) && isfinite(p0[2]) && isfinite(p1[1]) && isfinite(p1[2]))
                clear_region!()
                return
            end
            update_region_from_drag!(p0, p1)
            refresh_labels!()
            refresh_spectrum!()
            return
        elseif ev.button == Mouse.left && ev.action == Mouse.press
            p = mouseposition(ax_img)
            if any(isnan, p)
                return
            end
            clear_region!()
            u_max, v_max = slice_dims(axis[])
            u = Int(round(clamp(p[2], 1, u_max)))
            v = Int(round(clamp(p[1], 1, v_max)))
            u_idx[] = u; v_idx[] = v
            ii, jj, kk = uv_to_ijk(u, v, axis[], idx[])
            i_idx[] = clamp(ii, 1, siz[1]); j_idx[] = clamp(jj, 1, siz[2]); k_idx[] = clamp(kk, 1, siz[3])
            refresh_labels!(); refresh_spectrum!()
            uv_point[] = Point2f(v, u)
        end
    end

    on(events(ax_img).mouseposition) do p
        if zoom_drag_active[] && !any(isnan, p)
            zoom_drag_end[] = Point2f(p[1], p[2])
        elseif region_drag_active[] && !any(isnan, p)
            u_max, v_max = slice_dims(axis[])
            region_end[] = Point2f(clamp(p[1], 1, v_max), clamp(p[2], 1, u_max))
        end
    end

    # ---------- Saving ----------
    default_desktop = joinpath(homedir(), "Desktop")
    save_root = if save_dir === nothing
        isdir(default_desktop) ? default_desktop : pwd()
    else
        path = String(save_dir)
        if !isdir(path)
            mkpath(path)
        end
        path
    end
    resolved_settings_path = settings_path === nothing ?
        joinpath(save_root, "$(fname)_viewer_settings.toml") :
        abspath(String(settings_path))

    current_settings() = Dict{String,Any}(
        "axis" => axis[],
        "index" => idx[],
        "img_scale" => String(img_scale_mode[]),
        "spec_scale" => String(spec_scale_mode[]),
        "colormap" => String(cmap_name[]),
        "invert_colormap" => invert_cmap[],
        "show_crosshair" => show_crosshair[],
        "show_marker" => show_marker[],
        "show_grid" => show_grid[],
        "show_contours" => show_contours[],
        "contour_use_manual" => contour_use_manual[],
        "contour_levels" => collect(contour_manual_levels[]),
        "contour_colors" => collect(contour_manual_colors[]),
        "use_manual_clims" => use_manual[],
        "clim_min" => use_manual[] ? first(clims_manual[]) : first(clims_auto[]),
        "clim_max" => use_manual[] ? last(clims_manual[]) : last(clims_auto[]),
    )

    make_name = function (base::AbstractString, ext::AbstractString)
        b = isempty(base) ? fname : base
        return "$(b)_axis$(axis[])_idx$(idx[])_i$(i_idx[])_j$(j_idx[])_k$(k_idx[])_img$(String(img_scale_mode[]))_spec$(String(spec_scale_mode[])).$(ext)"
    end

    # ---------- Embedded power-spectrum layout ----------
    ps_layout_src = Observable(:zoom)      # :zoom | :full
    ps_layout_window = Observable(:hann)   # :hann | :hamming | :none
    ps_layout_pad = Observable(false)
    ps_layout_nanapo = Observable(false)
    ps_layout_units = Observable(:pixel)   # :pixel | :physical
    ps_layout_fit = Observable(false)
    ps_layout_blocks = Any[]

    function set_embedded_ps_visible!(visible::Bool)
        for block in ps_ui_blocks
            set_block_visible!(block, visible && !compact_layout)
        end
        for block in ps_layout_blocks
            set_block_visible!(block, visible)
        end
        nothing
    end
    set_embedded_ps_visible!(false)
    if compact_layout
        rowsize!(spec_grid, 1, Fixed(0))
        set_block_visible!(info_box, false)
        set_block_visible!(lab_info, false)
    end

    ps_u_dim_now() = slice_axis_dims(axis[])[1]
    ps_v_dim_now() = slice_axis_dims(axis[])[2]
    ps_physical_available() = has_wcs(wcs, ps_u_dim_now()) && has_wcs(wcs, ps_v_dim_now())
    ps_pixel_scales() = begin
        if ps_physical_available()
            dy = abs(wcs[ps_u_dim_now()].cdelt)
            dx = abs(wcs[ps_v_dim_now()].cdelt)
            (dx, dy)
        else
            (1.0, 1.0)
        end
    end
    ps_physical_unit_label() = begin
        if ps_physical_available()
            u = wcs[ps_v_dim_now()].cunit
            isempty(u) ? "1" : u
        else
            ""
        end
    end

    function ps_layout_clear!()
        for b in ps_layout_blocks
            try
                Makie.delete!(b)
            catch
            end
        end
        empty!(ps_layout_blocks)
        nothing
    end

    function ps_layout_subimage()
        M = slice_proc[]
        if ps_layout_src[] === :full
            return M
        end
        fl = ax_img.finallimits[]
        x0 = Float64(fl.origin[1])
        y0 = Float64(fl.origin[2])
        x1 = x0 + Float64(fl.widths[1])
        y1 = y0 + Float64(fl.widths[2])
        i_lo = clamp(Int(floor(min(x0, x1))), 1, size(M, 1))
        i_hi = clamp(Int(ceil(max(x0, x1))),  1, size(M, 1))
        j_lo = clamp(Int(floor(min(y0, y1))), 1, size(M, 2))
        j_hi = clamp(Int(ceil(max(y0, y1))),  1, size(M, 2))
        (i_hi <= i_lo || j_hi <= j_lo) && return M
        return M[i_lo:i_hi, j_lo:j_hi]
    end

    function ps_layout_status_text(meta)
        io = IOBuffer()
        print(io, "size $(meta.ny_in)×$(meta.nx_in)")
        if meta.padded
            print(io, " (pad→$(meta.ny_eff)×$(meta.nx_eff))")
        end
        print(io, " • $(meta.src) • ")
        print(io, meta.window === :none ? "none" : titlecase(String(meta.window)))
        meta.apodized && print(io, " • NaN apod")
        meta.f_sky < 1.0 && print(io, " • f_sky=$(round(meta.f_sky; digits = 3))")
        print(io, " • k=", meta.k_phys ? "1/$(ps_physical_unit_label())" : "cycles/pixel")
        return String(take!(io))
    end

    function render_power_spectrum_layout!()
        ps_layout_clear!()
        sub = ps_layout_subimage()
        ny0, nx0 = size(sub)
        if ny0 < 4 || nx0 < 4
            lab = Label(ps_plot_grid[1, 1]; text = "Selection too small for FFT (need ≥ 4×4).", fontsize = 14, color = ui_text)
            push!(ps_layout_blocks, lab)
            ps_layout_status[] = " "
            return
        end

        src_label = ps_layout_src[] === :full ? "full" : "zoom"
        use_phys = ps_layout_units[] === :physical && ps_physical_available()
        dx, dy = ps_pixel_scales()
        k_unit_lbl = use_phys ? "1/$(ps_physical_unit_label())" : "cycles/pixel"
        # why: factored computation shared with the pop-out window's ps_render!
        bundle = _cube_ps_bundle(sub;
                                 window = ps_layout_window[],
                                 pad_pow2 = ps_layout_pad[],
                                 apodize_nan = ps_layout_nanapo[],
                                 use_phys = use_phys, dx = dx, dy = dy)
        P2d = bundle.P2d
        ny, nx = bundle.meta.ny_eff, bundle.meta.nx_eff
        meta = (; bundle.meta..., k_phys = use_phys, src = src_label)

        # ---- Tailles dynamiques pour stacker dans la rangée plots ----
        ps_inner_gap = compact_layout ? 6 : 12
        ps_avail = max(280, ps_plot_row_height - ps_header_height - 16)
        heatmap_box_h = clamp(round(Int, ps_avail * 0.58), 200, ps_axis_size)
        psd_box_h = max(120, ps_avail - heatmap_box_h - ps_inner_gap)
        ps_box_w = heatmap_box_h  # heatmap carrée → PSD alignée sur la même largeur

        # ---- 2D power spectrum heatmap (row 1) ----
        vis = bundle.P2d_log10
        ax2d = Axis(
            ps_plot_grid[1, 1];
            title = latexstring("\\text{2D power spectrum (log10) — ", latex_safe(src_label), "}"),
            xlabel = use_phys ?
                latexstring("k_x\\;(", latex_safe(k_unit_lbl), ")") :
                L"k_x\;\text{(cycles/pixel)}",
            ylabel = use_phys ?
                latexstring("k_y\\;(", latex_safe(k_unit_lbl), ")") :
                L"k_y\;\text{(cycles/pixel)}",
            aspect = DataAspect(),
            width = ps_box_w,
            height = heatmap_box_h,
            halign = :center,
            valign = :top,
            xtickformat = latex_tick_formatter,
            ytickformat = latex_tick_formatter,
        )
        kx = bundle.kx
        ky = bundle.ky
        hm_ps = heatmap!(ax2d, kx, ky, vis; colormap = cm_obs[])
        cb = Colorbar(
            ps_plot_grid[1, 2],
            hm_ps;
            label = L"\log_{10}|F|^2",
            width = 18,
            height = _axis_render_height(ax2d),
            tellheight = false,
            valign = :top,
        )
        push!(ps_layout_blocks, ax2d)
        push!(ps_layout_blocks, cb)

        # ---- 1D radial PSD (log–log) just below the heatmap ----
        prof = bundle.prof
        k = bundle.k
        p_floored = bundle.prof_floored
        # log–log : on retire le DC (k = 0) et tout k ≤ 0 (cf. convention :log10).
        pos_mask = k .> 0
        k_pos = k[pos_mask]
        p_pos = p_floored[pos_mask]

        ax1d = Axis(
            ps_plot_grid[2, 1];
            title = latexstring("\\text{1D radial power spectrum (log–log) — ", latex_safe(src_label), "}"),
            xlabel = use_phys ?
                latexstring("k\\;(", latex_safe(k_unit_lbl), ")") :
                L"k\;\text{(cycles/pixel)}",
            ylabel = L"\langle|F|^2\rangle",
            xscale = log10,
            yscale = log10,
            width = ps_box_w,
            height = psd_box_h,
            halign = :center,
            valign = :top,
            xtickformat = latex_tick_formatter,
        )
        isempty(k_pos) || lines!(ax1d, k_pos, p_pos; color = ui_accent, linewidth = 1.8)
        push!(ps_layout_blocks, ax1d)

        fit_status = ""
        if ps_layout_fit[] && length(k) >= 3
            valid_k = filter(>(0), k)
            auto_lo = isempty(valid_k) ? 0.0 : Float64(first(valid_k))
            auto_hi = isempty(k) ? Inf : Float64(last(k))
            kmin_txt = get_box_str(ps_kmin_box)
            kmax_txt = get_box_str(ps_kmax_box)
            kmin_v = isempty(kmin_txt) ? auto_lo : something(tryparse(Float64, kmin_txt), auto_lo)
            kmax_v = isempty(kmax_txt) ? auto_hi : something(tryparse(Float64, kmax_txt), auto_hi)
            slope, intercept, n_used = fit_loglog_slope(k, prof; kmin = kmin_v, kmax = kmax_v)
            if isfinite(slope) && n_used >= 2
                kfit = filter(ki -> ki > 0 && ki >= kmin_v && ki <= kmax_v, k)
                if !isempty(kfit)
                    yfit = Float32.(10 .^ (slope .* log10.(Float64.(kfit)) .+ intercept))
                    lines!(ax1d, kfit, yfit; color = :red, linestyle = :dash, linewidth = 1.5)
                    fit_status = " • slope=$(round(slope; digits = 3)) [n=$(n_used)]"
                end
            end
        end

        ps_layout_status[] = ps_layout_status_text(meta) * fit_status
        nothing
    end

    function apply_layout_mode!()
        if layout_mode[] === :power_spectrum
            colsize!(main_grid, 1, Relative(1 / 2))
            colsize!(main_grid, 2, Relative(1 / 2))
            rowsize!(main_grid, 1, Fixed(ps_plot_row_height))
            rowsize!(main_grid, 2, Fixed(controls_height))
            rowsize!(spec_grid, 1, Fixed(0))
            rowsize!(spec_grid, 2, Fixed(0))
            rowsize!(spec_grid, 3, Fixed(0))
            refresh_control_mode!()
            set_block_visible!(ax_img, true)
            set_block_visible!(ax_cmp, false)
            set_block_visible!(img_colorbar, true)
            set_block_visible!(img_colorbar_cmp, false)
            set_block_visible!(info_box, false)
            set_block_visible!(lab_info, false)
            set_block_visible!(ax_spec, false)
            set_block_visible!(ax_hist, false)
            set_embedded_ps_visible!(true)
            render_power_spectrum_layout!()
            set_status!("Power spectrum layout enabled.")
        else
            rowsize!(spec_grid, 1, compact_layout ? Fixed(0) : Auto())
            rowsize!(spec_grid, 2, Auto())
            rowsize!(spec_grid, 3, Auto())
            colsize!(main_grid, 1, Auto())
            colsize!(main_grid, 2, Auto())
            rowsize!(main_grid, 1, compact_layout ? Fixed(plot_row_height) : Auto())
            rowsize!(main_grid, 2, Fixed(controls_height))
            refresh_control_mode!()
            set_block_visible!(ax_img, true)
            set_block_visible!(ax_cmp, compare_visible[])
            set_block_visible!(img_colorbar, true)
            set_block_visible!(img_colorbar_cmp, compare_visible[])
            set_block_visible!(info_box, !compact_layout)
            set_block_visible!(lab_info, !compact_layout)
            set_block_visible!(ax_spec, true)
            set_block_visible!(ax_hist, true)
            ps_layout_clear!()
            set_embedded_ps_visible!(false)
            set_status!("Base layout restored.")
        end
        nothing
    end

    on(layout_mode) do _
        apply_layout_mode!()
    end

    on_mode(ps_btn.clicks, :navigation) do _
        layout_mode[] = :power_spectrum
    end

    on_mode(base_layout_btn.clicks, :navigation) do _
        layout_mode[] = :base
    end

    on(ps_src_menu.selection) do sel
        sel === nothing && return
        ps_layout_src[] = sel == "full" ? :full : :zoom
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    on(ps_win_menu.selection) do sel
        sel === nothing && return
        ps_layout_window[] = sel == "Hamming" ? :hamming : sel == "None" ? :none : :hann
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    on(ps_unit_menu.selection) do sel
        sel === nothing && return
        ps_layout_units[] = sel == "physical" ? :physical : :pixel
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    on(ps_pad_chk.checked) do v
        ps_layout_pad[] = v
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    on(ps_nanapo_chk.checked) do v
        ps_layout_nanapo[] = v
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    on(ps_fit_chk.checked) do v
        ps_layout_fit[] = v
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    on(ps_kmin_box.stored_string) do _
        layout_mode[] === :power_spectrum && ps_layout_fit[] && render_power_spectrum_layout!()
    end

    on(ps_kmax_box.stored_string) do _
        layout_mode[] === :power_spectrum && ps_layout_fit[] && render_power_spectrum_layout!()
    end

    on(ps_refresh_btn.clicks) do _
        render_power_spectrum_layout!()
    end

    on(slice_proc) do _
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    # Unified export (let Makie/Cairo choose the backend)
    save_with_format(path::AbstractString, fig) = CairoMakie.save(String(path), fig)

    moment_label() = moment_order[] == 0 ? "moment0" : moment_order[] == 1 ? "moment1" : "moment2"

    # Write a Float32 array to a FITS primary HDU, optionally preserving the
    # cube's original header (WCS / BUNIT / provenance) via FITSIO.
    function write_fits_array(path::AbstractString, arr; header = nothing)
        FITS(String(path), "w") do f
            if header === nothing
                write(f, Float32.(arr))
            else
                write(f, Float32.(arr); header = header)
            end
        end
        nothing
    end

    function save_moment_png!(out::AbstractString)
        f_mom = CairoMakie.Figure(size = (700, 560))
        colgap!(f_mom.layout, -8)
        xlab_s, ylab_s = slice_axis_labels(axis[])
        axM = CairoMakie.Axis(
            f_mom[1, 1];
            title = latexstring("\\text{", latex_safe(fname), " ", latex_safe(moment_label()), " axis $(axis[])}"),
            xlabel = xlab_s,
            ylabel = ylab_s,
            aspect = CairoMakie.DataAspect(),
            yreversed = true,
            xtickformat = latex_tick_formatter,
            ytickformat = latex_tick_formatter,
        )
        lim = clamped_extrema(moment_raw[])
        hmM = CairoMakie.heatmap!(axM, moment_raw[]; colormap = cm_obs[], colorrange = lim)
        CairoMakie.Colorbar(
            f_mom[1, 2],
            hmM;
            label = moment_label(),
            width = 20,
            height = _axis_render_height(axM),
            tellheight = false,
            valign = :center,
        )
        CairoMakie.save(String(out), f_mom; backend = CairoMakie)
        nothing
    end

    function export_fits_product!(product::AbstractString, out::AbstractString)
        src_id = String(ds.source_id)
        if product == "slice"
            hdr = fits_header_for_slice(header, axis[], idx[]; source_id = src_id)
            write_fits_array(out, get_slice(data, axis[], idx[]); header = hdr)
        elseif product == "region"
            if isempty(region_uvs[])
                throw(ArgumentError("select a box or circle region before exporting the averaged region FITS"))
            end
            spec = mean_region_spectrum(data, axis[], region_uvs[])
            hdr = fits_header_for_region_spectrum(header, axis[], length(region_uvs[]);
                                                  source_id = src_id)
            write_fits_array(out, spec; header = hdr)
        elseif product == "moment"
            hdr = fits_header_for_moment(header, axis[], moment_order[]; source_id = src_id)
            write_fits_array(out, moment_raw[]; header = hdr)
        elseif product == "filtered cube"
            hdr = fits_header_for_filtered_cube(header, axis[], sigma[]; source_id = src_id)
            write_fits_array(out, filtered_cube_by_slice(data, axis[], sigma[]); header = hdr)
        else
            throw(ArgumentError("unknown FITS product: $(product)"))
        end
        nothing
    end

    on_mode(btn_moment_png.clicks, :analysis) do _
        spawn_safely() do
            base = get_box_str(fname_box)
            base = isempty(base) ? "$(fname)_$(moment_label())" : base
            out = joinpath(save_root, make_name(base, "png"))
            try
                save_moment_png!(out)
                set_status!("Saved moment PNG to $(out).")
            catch e
                msg = "Failed to save moment PNG $(out): $(sprint(showerror, e))"
                set_status!(msg)
                @error msg exception=(e, catch_backtrace())
            end
        end
    end

    on_mode(btn_moment_fits.clicks, :analysis) do _
        spawn_safely() do
            base = get_box_str(fname_box)
            base = isempty(base) ? "$(fname)_$(moment_label())" : base
            out = joinpath(save_root, make_name(base, "fits"))
            try
                hdr = fits_header_for_moment(header, axis[], moment_order[];
                                             source_id = String(ds.source_id))
                write_fits_array(out, moment_raw[]; header = hdr)
                set_status!("Saved moment FITS to $(out).")
            catch e
                msg = "Failed to save moment FITS $(out): $(sprint(showerror, e))"
                set_status!(msg)
                @error msg exception=(e, catch_backtrace())
            end
        end
    end

    on_mode(btn_save_fits.clicks, :analysis) do _
        spawn_safely() do
            product = String(something(fits_product_menu.selection[], "slice"))
            clean_product = replace(product, " " => "_")
            base = get_box_str(fname_box)
            base = isempty(base) ? "$(fname)_$(clean_product)" : base
            out = joinpath(save_root, make_name(base, "fits"))
            try
                export_fits_product!(product, out)
                set_status!("Saved $(product) FITS to $(out).")
            catch e
                msg = "Failed to save $(product) FITS $(out): $(sprint(showerror, e))"
                set_status!(msg)
                @error msg exception=(e, catch_backtrace())
            end
        end
    end

    on_mode(btn_save_state.clicks, :export) do _
        try
            save_viewer_settings(resolved_settings_path, current_settings())
            btn_save_state.labelcolor[] = ui_success
            set_status!("Saved settings to $(resolved_settings_path).")
        catch e
            msg = "Failed to save settings: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
        end
    end

    on_mode(btn_load_state.clicks, :export) do _
        if !isfile(resolved_settings_path)
            set_status!("Settings file not found: $(resolved_settings_path)")
            return
        end
        # Loading viewer state pokes widgets that belong to :navigation /
        # :analysis cards (axis_menu, slice_slider, invert_chk, ...). Without
        # the bypass their `on_mode` listeners would silently no-op because
        # we are currently in :export mode.
        bypass_mode_gate[] = true
        try
            st = load_viewer_settings(resolved_settings_path)

            axis_val = clamp(Int(get(st, "axis", axis[])), 1, 3)
            axis_menu.selection[] = axes_labels[axis_val]

            idx_val = clamp(Int(get(st, "index", idx[])), 1, siz[axis_val])
            slice_slider.value[] = idx_val

            img_scale_val = String(get(st, "img_scale", String(img_scale_mode[])))
            if img_scale_val in ("lin", "log10", "ln")
                img_scale_menu.selection[] = img_scale_val
            end

            spec_scale_val = String(get(st, "spec_scale", String(spec_scale_mode[])))
            if spec_scale_val in ("lin", "log10", "ln")
                spec_scale_menu.selection[] = spec_scale_val
            end

            cmap_val = Symbol(String(get(st, "colormap", String(cmap_name[]))))
            try
                to_cmap(cmap_val)
                cmap_name[] = cmap_val
                if String(cmap_val) in MANTA_COLORMAP_OPTIONS
                    cmap_menu.selection[] = String(cmap_val)
                end
            catch
                @warn "Ignoring invalid colormap in settings" colormap=cmap_val
            end

            invert_val = Bool(get(st, "invert_colormap", invert_cmap[]))
            invert_chk.checked[] = invert_val

            crosshair_chk.checked[] = Bool(get(st, "show_crosshair", show_crosshair[]))
            marker_chk.checked[] = Bool(get(st, "show_marker", show_marker[]))
            grid_chk.checked[] = Bool(get(st, "show_grid", show_grid[]))
            contour_chk.checked[] = Bool(get(st, "show_contours", show_contours[]))
            contour_use_manual[] = Bool(get(st, "contour_use_manual", contour_use_manual[]))
            raw_levels = get(st, "contour_levels", contour_manual_levels[])
            raw_colors = get(st, "contour_colors", contour_manual_colors[])
            contour_manual_levels[] = Float32.(raw_levels)
            contour_manual_colors[] = String.(raw_colors)
            if contour_use_manual[] && !isempty(contour_manual_levels[])
                set_box_text!(contour_levels_box, format_contour_specs(contour_manual_levels[], contour_manual_colors[]))
            else
                set_box_text!(contour_levels_box, "")
            end

            use_manual_val = Bool(get(st, "use_manual_clims", use_manual[]))
            if use_manual_val
                cmin = Float32(get(st, "clim_min", first(clims_manual[])))
                cmax = Float32(get(st, "clim_max", last(clims_manual[])))
                ok, new_manual, parsed_clims, msg = parse_manual_clims(string(cmin), string(cmax); fallback = clims_manual[])
                if ok && new_manual
                    clims_manual[] = parsed_clims
                    use_manual[] = true
                    set_box_text!(clim_min_box, string(first(parsed_clims)))
                    set_box_text!(clim_max_box, string(last(parsed_clims)))
                    limits!(ax_spec, nothing, nothing, first(parsed_clims), last(parsed_clims))
                    set_status!(msg)
                end
                else
                    use_manual[] = false
                    set_box_text!(clim_min_box, "")
                    set_box_text!(clim_max_box, "")
                    autolimits!(ax_spec)
                    xlims!(ax_spec, 0f0, Float32(max(0, length(spec_y_buf) - 1)))
            end
            btn_load_state.labelcolor[] = ui_success
            set_status!("Loaded settings from $(resolved_settings_path).")
        catch e
            msg = "Failed to load settings: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
        finally
            bypass_mode_gate[] = false
        end
    end

    # ---------- Save image (slice + colorbar + crosshair) ----------
    on_mode(btn_save_img.clicks, :export) do _
        spawn_safely() do
            ext  = String(something(fmt_menu.selection[], "png"))
            out  = joinpath(save_root, make_name(get_box_str(fname_box), ext))
            try
                f_slice = CairoMakie.Figure(size = (700, 560))
                colgap!(f_slice.layout, -8)
                xlab_s, ylab_s = slice_axis_labels(axis[])
                axS = CairoMakie.Axis(
                    f_slice[1, 1];
                    title     = make_slice_title(fname, axis[], idx[]),
                    xlabel    = xlab_s,
                    ylabel    = ylab_s,
                    aspect    = CairoMakie.DataAspect(),
                    yreversed = true,
                    xtickformat = latex_tick_formatter,
                    ytickformat = latex_tick_formatter,
                )
                hmS = CairoMakie.heatmap!(axS, slice_disp[]; colormap = cm_obs[], colorrange = clims_obs[])
                if show_contours[] && !isempty(contour_levels_obs[])
                    CairoMakie.contour!(axS, slice_disp[]; levels = contour_levels_obs[], color = contour_colors_obs[], linewidth = 1.2)
                end
                axS.xgridvisible[] = show_grid[]
                axS.ygridvisible[] = show_grid[]
                if show_crosshair[]
                    u_max, v_max = slice_dims(axis[])
                    u, v = u_idx[], v_idx[]
                    CairoMakie.linesegments!(
                        axS,
                        Point2f[
                            Point2f(1, u), Point2f(v_max, u),
                            Point2f(v, 1), Point2f(v, u_max),
                        ];
                        color = (:white, 0.9),
                        linewidth = 1.6,
                        linestyle = :dot,
                    )
                end
                if show_marker[]
                    CairoMakie.scatter!(axS, [Point2f(uv_point[]...)], markersize = 10)
                end
                if !isempty(region_uvs[])
                    CairoMakie.lines!(
                        axS,
                        region_segments_from_points(region_start[], region_end[], region_shape[]);
                        color = (RGBf(1.0, 0.78, 0.18), 0.98),
                        linewidth = 2.4,
                    )
                end
                CairoMakie.Colorbar(
                    f_slice[1, 2],
                    hmS;
                    label = unit_label,
                    width = 20,
                    height = _axis_render_height(axS),
                    tellheight = false,
                    valign = :center,
                )

                CairoMakie.save(String(out), f_slice; backend = CairoMakie)
                @info "Saved image" out
                set_status!("Saved image to $(out).")
            catch e
                msg = "Failed to save image $(out): $(sprint(showerror, e))"
                set_status!(msg)
                @error msg exception=(e, catch_backtrace())
            end
        end
    end

    # ---------- Save spectrum (lines plot) ----------
    on_mode(btn_save_spec.clicks, :export) do _
        spawn_safely() do
            ext = String(something(fmt_menu.selection[], "png"))
            base = get_box_str(fname_box)
            base = isempty(base) ? "$(fname)_spectrum" : base
            out = joinpath(save_root, make_name(base, ext))

            try
                f_spec = CairoMakie.Figure(size = (600, 400))
                axP = CairoMakie.Axis(
                    f_spec[1, 1];
                    title  = isempty(region_uvs[]) ? make_spec_title(i_idx[], j_idx[], k_idx[]) : L"\text{Mean spectrum in selected region}",
                    xlabel = L"\text{index along slice axis}",
                    ylabel = unit_label_tex,
                    xtickformat = latex_tick_formatter,
                    ytickformat = latex_tick_formatter,
                )
                CairoMakie.lines!(axP, spec_x_raw[], spec_y_disp[])
                CairoMakie.xlims!(axP, 0f0, Float32(max(0, length(spec_x_raw[]) - 1)))

                CairoMakie.save(String(out), f_spec; backend = CairoMakie)
                @info "Saved spectrum" out
                set_status!("Saved spectrum to $(out).")
            catch e
                msg = "Failed to save spectrum $(out): $(sprint(showerror, e))"
                set_status!(msg)
                @error msg exception=(e, catch_backtrace())
            end
        end
    end

    # `current_animation_request` is now a module-level helper in
    # `src/views/cube/AnimationRequest.jl`. It takes the captured pieces of
    # state (axis, size, the four textboxes and the pingpong checkbox)
    # explicitly so the GIF/animation pipeline is decoupled from this
    # closure.

    function start_channel_animation!(frames::Vector{Int}, fps::Int)
        anim_playing[] = true
        play_btn.label[] = "Pause"
        spawn_safely() do
            delay = 1 / max(1, fps)
            try
                while anim_playing[]
                    for fidx in frames
                        anim_playing[] || break
                        slice_slider.value[] = fidx
                        sleep(delay)
                    end
                    loop_chk.checked[] || break
                end
            finally
                anim_playing[] = false
                play_btn.label[] = "Play"
            end
        end
        nothing
    end

    on_mode(play_btn.clicks, :export) do _
        if anim_playing[]
            anim_playing[] = false
            play_btn.label[] = "Play"
            set_status!("Animation paused.")
            return
        end
        ok, frames, fps, msg = current_animation_request(
            axis[], siz, start_box, stop_box, step_box, fps_box, pingpong_chk,
        )
        set_status!(ok ? "Interactive animation playing at $(fps) FPS." : msg)
        ok || return
        start_channel_animation!(frames, fps)
    end


    # ---------- GIF export  ----------
    on_mode(anim_btn.clicks, :export) do _
        a = axis[]; amax = siz[a]
        ok, frames, fps, msg = current_animation_request(
            a, siz, start_box, stop_box, step_box, fps_box, pingpong_chk,
        )
        set_status!(msg)
        if !ok
            @warn "Invalid GIF parameters" msg axis=a amax=amax
            return
        end

        # strict name: <fits_name>.gif (e.g., synthetic_cube.gif)
        outfile = joinpath(save_root, "$(fname).gif")
        ny, nx = slice_dims(axis[])
        w_img = 640
        h_img = Int(round(w_img * ny / nx))
        extra_for_cb = 80 # colorbar space
        fig_gif = CairoMakie.Figure(size = (w_img + extra_for_cb, h_img))
        colgap!(fig_gif.layout, -8)
        axG = CairoMakie.Axis(fig_gif[1, 1]; aspect = DataAspect(), yreversed = true)
        Makie.hidedecorations!(axG, grid = false)

        hmG = CairoMakie.heatmap!(axG, slice_disp; colormap = cm_obs, colorrange = clims_obs)
        CairoMakie.Colorbar(
            fig_gif[1, 2],
            hmG;
            label = "intensity",
            width = 20,
            height = _axis_render_height(axG),
            tellheight = false,
            valign = :center,
        )

        try
            record(fig_gif, outfile, frames; framerate = fps) do fidx
                idx[] = fidx
            end
            @info "Animation saved: $outfile"
            set_status!("GIF exported to $(outfile).")
        catch e
            msg2 = "Failed to export animation $(outfile): $(sprint(showerror, e))"
            set_status!(msg2)
            @error msg2 exception=(e, catch_backtrace())
        end
    end


    # ---------- Power spectrum window ----------
    ps_fig_ref = Ref{Any}(nothing)
    ps_alive_ref = Ref(false)

    function open_power_spectrum_window!()
        if ps_alive_ref[] && ps_fig_ref[] !== nothing
            try
                display(ps_fig_ref[])
                return ps_fig_ref[]
            catch
                ps_alive_ref[] = false
                ps_fig_ref[] = nothing
            end
        end

        ps_mode    = Observable(:two_d)   # :two_d | :one_d
        ps_src     = Observable(:zoom)    # :zoom  | :full
        ps_window  = Observable(:hann)    # :hann | :hamming | :none
        ps_pad     = Observable(false)
        ps_nanapo  = Observable(false)
        ps_units   = Observable(:pixel)   # :pixel | :physical
        ps_fit_on  = Observable(false)

        fig_ps = Figure(size = (1000, 760))
        header = fig_ps[1, 1] = GridLayout(; alignmode = Outside(8))

        # Row 1: mode/source/window/units/refresh
        Label(header[1, 1]; text = "Mode", halign = :right, fontsize = 13)
        mode_menu  = Menu(header[1, 2]; options = ["2D", "1D"], prompt = "2D", width = 80)
        Label(header[1, 3]; text = "Source", halign = :right, fontsize = 13)
        src_menu   = Menu(header[1, 4]; options = ["zoom", "full"], prompt = "zoom", width = 80)
        Label(header[1, 5]; text = "Window", halign = :right, fontsize = 13)
        win_menu   = Menu(header[1, 6]; options = ["Hann", "Hamming", "None"], prompt = "Hann", width = 96)
        Label(header[1, 7]; text = "Units", halign = :right, fontsize = 13)
        unit_menu  = Menu(header[1, 8]; options = ["pixel", "physical"], prompt = "pixel", width = 96)
        refresh_btn = Button(header[1, 9]; label = "Refresh", width = 88, height = 30)

        # Row 2: toggles + fit band
        pad_chk    = Checkbox(header[2, 1]); Label(header[2, 2]; text = "Pad pow2", halign = :left, fontsize = 13, color = ui_text)
        nanapo_chk = Checkbox(header[2, 3]); Label(header[2, 4]; text = "NaN apod.", halign = :left, fontsize = 13, color = ui_text)
        fit_chk    = Checkbox(header[2, 5]); Label(header[2, 6]; text = "Fit slope", halign = :left, fontsize = 13, color = ui_text)
        Label(header[2, 7]; text = "k range", halign = :right, fontsize = 13)
        kmin_box   = Textbox(header[2, 8]; placeholder = "k_min", width = 84, height = 28)
        kmax_box   = Textbox(header[2, 9]; placeholder = "k_max", width = 84, height = 28)

        # Row 3: save buttons
        save_png_btn = Button(header[3, 1:2]; label = "Save PNG", width = 120, height = 28)
        save_pdf_btn = Button(header[3, 3:4]; label = "Save PDF", width = 120, height = 28)
        save_csv_btn = Button(header[3, 5:6]; label = "Save CSV (1D)", width = 140, height = 28)

        ps_status = Observable(" ")
        Label(header[3, 7:9]; text = ps_status, halign = :left, fontsize = 12, color = ui_text_muted, tellwidth = false)

        style_menu!(mode_menu); style_menu!(src_menu); style_menu!(win_menu); style_menu!(unit_menu)
        style_button!(refresh_btn); style_button!(save_png_btn); style_button!(save_pdf_btn); style_button!(save_csv_btn)
        style_checkbox!(pad_chk); style_checkbox!(nanapo_chk); style_checkbox!(fit_chk)
        style_textbox!(kmin_box); style_textbox!(kmax_box)

        plot_grid = fig_ps[2, 1] = GridLayout()
        colgap!(plot_grid, -8)
        rowsize!(fig_ps.layout, 1, Fixed(120))

        # WCS-derived physical pixel scale (for 1D + 2D physical units).
        u_dim_now() = slice_axis_dims(axis[])[1]
        v_dim_now() = slice_axis_dims(axis[])[2]
        physical_available() = has_wcs(wcs, u_dim_now()) && has_wcs(wcs, v_dim_now())
        pixel_scales() = begin
            if physical_available()
                dy = abs(wcs[u_dim_now()].cdelt)
                dx = abs(wcs[v_dim_now()].cdelt)
                (dx, dy)
            else
                (1.0, 1.0)
            end
        end
        physical_unit_label() = begin
            if physical_available()
                u = wcs[v_dim_now()].cunit
                isempty(u) ? "1" : u
            else
                ""
            end
        end

        # Cache of the latest 1D points, used by Save CSV and Fit.
        last_1d_k     = Float32[]
        last_1d_p     = Float32[]
        last_1d_units = "cycles/pixel"
        last_meta     = (; ny_in = 0, nx_in = 0, ny_eff = 0, nx_eff = 0,
                          padded = false, window = :none, apodized = false,
                          f_sky = 1.0, w_norm = 0.0, k_phys = false, src = "zoom")

        ps_blocks = Any[]
        clear_plot!() = begin
            for b in ps_blocks
                try
                    Makie.delete!(b)
                catch
                end
            end
            empty!(ps_blocks)
        end

        function ps_subimage()
            M = slice_proc[]
            if ps_src[] === :full
                return M
            end
            fl = ax_img.finallimits[]
            x0 = Float64(fl.origin[1])
            y0 = Float64(fl.origin[2])
            x1 = x0 + Float64(fl.widths[1])
            y1 = y0 + Float64(fl.widths[2])
            i_lo = clamp(Int(floor(min(x0, x1))), 1, size(M, 1))
            i_hi = clamp(Int(ceil(max(x0, x1))),  1, size(M, 1))
            j_lo = clamp(Int(floor(min(y0, y1))), 1, size(M, 2))
            j_hi = clamp(Int(ceil(max(y0, y1))),  1, size(M, 2))
            (i_hi <= i_lo || j_hi <= j_lo) && return M
            return M[i_lo:i_hi, j_lo:j_hi]
        end

        function format_status(meta)
            io = IOBuffer()
            print(io, "size $(meta.ny_in)×$(meta.nx_in)")
            if meta.padded
                print(io, " (pad→$(meta.ny_eff)×$(meta.nx_eff))")
            end
            print(io, " • $(meta.src) • ")
            print(io, meta.window === :none ? "none" : titlecase(String(meta.window)))
            if meta.apodized
                print(io, " • NaN apod")
            end
            if meta.f_sky < 1.0
                print(io, " • f_sky=$(round(meta.f_sky; digits = 3))")
            end
            print(io, " • k=", meta.k_phys ? "1/$(physical_unit_label())" : "cycles/pixel")
            return String(take!(io))
        end

        function ps_render!()
            clear_plot!()
            sub = ps_subimage()
            ny0, nx0 = size(sub)
            if ny0 < 4 || nx0 < 4
                lab = Label(plot_grid[1, 1]; text = "Selection too small for FFT (need ≥ 4×4).", fontsize = 14)
                push!(ps_blocks, lab)
                ps_status[] = " "
                empty!(last_1d_k); empty!(last_1d_p)
                return
            end
            src_label = ps_src[] === :full ? "full" : "zoom"
            use_phys = ps_units[] === :physical && physical_available()
            dx, dy = pixel_scales()
            k_unit_lbl = use_phys ? "1/$(physical_unit_label())" : "cycles/pixel"
            # why: factored computation shared with render_power_spectrum_layout!
            bundle = _cube_ps_bundle(sub;
                                     window = ps_window[],
                                     pad_pow2 = ps_pad[],
                                     apodize_nan = ps_nanapo[],
                                     use_phys = use_phys, dx = dx, dy = dy)
            P2d = bundle.P2d
            ny, nx = bundle.meta.ny_eff, bundle.meta.nx_eff
            meta = (; bundle.meta..., k_phys = use_phys, src = src_label)
            last_meta = meta

            if ps_mode[] === :two_d
                empty!(last_1d_k); empty!(last_1d_p)
                last_1d_units = k_unit_lbl
                vis = bundle.P2d_log10
                ax = Axis(
                    plot_grid[1, 1];
                    title  = latexstring("\\text{2D power spectrum (log10) — ", latex_safe(src_label), "}"),
                    xlabel = use_phys ?
                        latexstring("k_x\\;(", latex_safe(k_unit_lbl), ")") :
                        L"k_x\;\text{(cycles/pixel)}",
                    ylabel = use_phys ?
                        latexstring("k_y\\;(", latex_safe(k_unit_lbl), ")") :
                        L"k_y\;\text{(cycles/pixel)}",
                    aspect = DataAspect(),
                    xtickformat = latex_tick_formatter,
                    ytickformat = latex_tick_formatter,
                )
                kx = bundle.kx
                ky = bundle.ky
                hm = heatmap!(ax, kx, ky, vis; colormap = cm_obs[])
                cb = Colorbar(
                    plot_grid[1, 2],
                    hm;
                    label = L"\log_{10}|F|^2",
                    width = 18,
                    height = _axis_render_height(ax),
                    tellheight = false,
                    valign = :center,
                )
                push!(ps_blocks, ax); push!(ps_blocks, cb)
            else
                prof = bundle.prof
                k = bundle.k
                p_floored = bundle.prof_floored
                resize!(last_1d_k, length(k));  copyto!(last_1d_k, k)
                resize!(last_1d_p, length(prof)); copyto!(last_1d_p, prof)
                last_1d_units = k_unit_lbl

                ax = Axis(
                    plot_grid[1, 1];
                    title  = latexstring("\\text{1D radial power spectrum — ", latex_safe(src_label), "}"),
                    xlabel = use_phys ?
                        latexstring("k\\;(", latex_safe(k_unit_lbl), ")") :
                        L"k\;\text{(cycles/pixel)}",
                    ylabel = L"\langle|F|^2\rangle",
                    yscale = log10,
                    xtickformat = latex_tick_formatter,
                )
                if !isempty(k)
                    lines!(ax, k, p_floored; color = ui_accent, linewidth = 1.8)
                end
                push!(ps_blocks, ax)

                if ps_fit_on[] && length(k) >= 3
                    kmin_txt = get_box_str(kmin_box)
                    kmax_txt = get_box_str(kmax_box)
                    valid_k = filter(>(0), k)
                    auto_lo = isempty(valid_k) ? 0.0 : Float64(first(valid_k))
                    auto_hi = isempty(k) ? Inf : Float64(last(k))
                    kmin_v = isempty(kmin_txt) ? auto_lo : something(tryparse(Float64, kmin_txt), auto_lo)
                    kmax_v = isempty(kmax_txt) ? auto_hi : something(tryparse(Float64, kmax_txt), auto_hi)
                    slope, intercept, n_used = fit_loglog_slope(k, prof; kmin = kmin_v, kmax = kmax_v)
                    if isfinite(slope) && n_used >= 2
                        kfit = filter(ki -> ki > 0 && ki >= kmin_v && ki <= kmax_v, k)
                        if !isempty(kfit)
                            yfit = Float32.(10 .^ (slope .* log10.(Float64.(kfit)) .+ intercept))
                            lines!(ax, kfit, yfit; color = :red, linestyle = :dash, linewidth = 1.5)
                            slope_str = "slope=$(round(slope; digits = 3)) [n=$(n_used)]"
                            ps_status[] = format_status(meta) * " • " * slope_str
                            return
                        end
                    end
                end
            end
            ps_status[] = format_status(meta)
        end

        on(mode_menu.selection)  do sel; sel === nothing || (ps_mode[] = sel == "1D" ? :one_d : :two_d; ps_render!()); end
        on(src_menu.selection)   do sel; sel === nothing || (ps_src[] = sel == "full" ? :full : :zoom; ps_render!()); end
        on(win_menu.selection)   do sel
            sel === nothing && return
            ps_window[] = sel == "Hamming" ? :hamming : sel == "None" ? :none : :hann
            ps_render!()
        end
        on(unit_menu.selection)  do sel
            sel === nothing && return
            ps_units[] = sel == "physical" ? :physical : :pixel
            ps_render!()
        end
        on(pad_chk.checked)    do v; ps_pad[]    = v; ps_render!(); end
        on(nanapo_chk.checked) do v; ps_nanapo[] = v; ps_render!(); end
        on(fit_chk.checked)    do v; ps_fit_on[] = v; ps_render!(); end
        on(kmin_box.stored_string) do _; ps_fit_on[] && ps_render!(); end
        on(kmax_box.stored_string) do _; ps_fit_on[] && ps_render!(); end
        on(refresh_btn.clicks) do _; ps_render!(); end

        ps_window_alive = Ref(true)
        on(slice_proc) do _
            ps_window_alive[] && ps_render!()
        end

        ps_save_path(ext) = joinpath(save_root, make_name(get_box_str(fname_box), "powerspec.$(ext)"))
        on(save_png_btn.clicks) do _
            try
                out = ps_save_path("png")
                CairoMakie.save(String(out), fig_ps; backend = CairoMakie)
                set_status!("Saved power spectrum to $(out).")
            catch e
                set_status!("Failed to save PNG: $(sprint(showerror, e))")
            end
        end
        on(save_pdf_btn.clicks) do _
            try
                out = ps_save_path("pdf")
                CairoMakie.save(String(out), fig_ps; backend = CairoMakie)
                set_status!("Saved power spectrum to $(out).")
            catch e
                set_status!("Failed to save PDF: $(sprint(showerror, e))")
            end
        end
        on(save_csv_btn.clicks) do _
            if isempty(last_1d_k)
                set_status!("No 1D points to save (switch to 1D mode first).")
                return
            end
            try
                out = ps_save_path("csv")
                open(String(out), "w") do io
                    println(io, "# window=$(last_meta.window) pad=$(last_meta.padded) nan_apod=$(last_meta.apodized) f_sky=$(last_meta.f_sky) src=$(last_meta.src)")
                    println(io, "k_$(replace(last_1d_units, ' ' => '_')),power")
                    for i in eachindex(last_1d_k)
                        println(io, last_1d_k[i], ",", last_1d_p[i])
                    end
                end
                set_status!("Saved 1D PS CSV to $(out).")
            catch e
                set_status!("Failed to save CSV: $(sprint(showerror, e))")
            end
        end

        ps_render!()
        keepalive!(fig_ps)
        ps_fig_ref[]   = fig_ps
        ps_alive_ref[] = true
        on(fig_ps.scene.events.window_open) do is_open
            if !is_open
                ps_window_alive[] = false
                ps_alive_ref[] = false
                ps_fig_ref[]   = nothing
                forget!(fig_ps)
            end
        end
        display(fig_ps)
        return fig_ps
    end

    on(ps_popout_btn.clicks) do _
        try
            open_power_spectrum_window!()
        catch e
            msg = "Failed to open power spectrum: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
        end
    end

    # ---------- Init ----------
    refresh_all!()
    refresh_hist_axes!()
    # Optional preload of a comparison cube passed via the `compare=` kwarg
    # (used by scripted invocations and headless tests). Silently no-ops if
    # the path can't be loaded — `load_compare_cube!` already logs through
    # `set_status!` so we don't double-report here.
    if compare !== nothing
        try
            load_compare_cube!(String(compare))
        catch e
            @warn "Failed to preload comparison cube" path=compare exception=e
        end
    end
    keepalive!(fig)

    on(fig.scene.events.window_open) do is_open
        if !is_open
            forget!(fig)
        end
    end
    if display_fig
        display(fig)
    end
    return fig
end

# Public alias for the dataset-driven cube viewer.
const view_cube = _view_cube
