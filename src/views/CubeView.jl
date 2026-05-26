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
    state = nothing,
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
    compare_path_current = Observable("")
    compare_mode = Observable(:B)
    view_product = Observable(:slice)
    moment_order = Observable(0)
    layout_mode = Observable(:base)
    anim_playing = Observable(false)

    # Keep raw slice observables on a single concrete array type. Observables.jl
    # stores the concrete type of the initial lifted value; returning views here
    # makes later updates fail when smoothing / moment products emit Matrix
    # buffers or when the active axis changes the concrete SubArray type.
    slice_raw = lift(axis, idx) do a, id
        get_slice_copy(data, a, clamp(id, 1, siz[a]))
    end

    compare_slice_raw = lift(compare_data, axis, idx) do cmp, a, id
        cmp === nothing && return fill(NaN32, slice_dims(a))
        get_slice_copy(cmp, a, clamp(id, 1, siz[a]))
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

    # ---------- Mask state ----------
    # The mask system keeps a *declarative* source (a `MaskSource` subtype that
    # describes how the mask was built) alongside the concretized
    # `BitArray{3}`. The bits drive runtime kernels (moments / spectra /
    # histogram); the source is what gets persisted to viewer_settings.toml so
    # the mask can be regenerated on reload — we never serialize the BitArray
    # itself.
    mask_source_obs = Observable{MaskSource}(NoMaskSource())
    mask_bits_obs   = Observable{Union{Nothing,BitArray{3}}}(nothing)
    mask_status_obs = Observable("No mask applied")

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
    # channels are passed once at construction. Cache by (axis, order, mask)
    # so toggling axis or moment order doesn't recompute the whole map.
    # Changing the persistent mask invalidates the cache entirely (see
    # `apply_mask_source!`) — we don't try to key by mask identity here
    # because that would force a fresh entry per BitArray instance even
    # when the bits are identical.
    _moment_cache = Dict{Tuple{Int,Int}, Matrix{Float32}}()
    moment_raw = lift(axis, moment_order, mask_bits_obs) do a, ord, mbits
        get!(_moment_cache, (Int(a), Int(ord))) do
            moment_map(data, a, ord; coords = spectral_coords(a),
                       threshold = moment_threshold,
                       nsigma = moment_nsigma,
                       channels = moment_channels === nothing ? (1:siz[a]) : moment_channels,
                       dx = moment_dx_for(a),
                       mask = mbits)
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

    # ---------- Mask slice / histogram input ----------
    # Pull the 2-D slice of the mask aligned with the current (axis, idx).
    # Returns `nothing` when no mask is active, so downstream lifts can
    # short-circuit to the unmasked path.
    mask_slice = lift(mask_bits_obs, axis, idx) do bits, a, id
        bits === nothing && return nothing
        id_safe = clamp(id, 1, size(bits, a))
        if a == 1
            collect(@view bits[id_safe, :, :])
        elseif a == 2
            collect(@view bits[:, id_safe, :])
        else
            collect(@view bits[:, :, id_safe])
        end
    end

    # Histogram input: when a mask is active, NaN-out the rejected voxels so
    # `histogram_profile` (which is binning a single 2-D matrix) excludes
    # them naturally — its internal `histogram_counts` already ignores
    # non-finite samples.
    hist_slice_obs = lift(slice_disp, mask_slice) do s, ms
        ms === nothing && return s
        out = Matrix{Float32}(undef, size(s))
        @inbounds for i in eachindex(s, out, ms)
            out[i] = ms[i] ? Float32(s[i]) : Float32(NaN)
        end
        out
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

    hist_pair_obs = lift(hist_slice_obs, hist_limits_obs, hist_bins_obs, hist_mode_obs) do s, lim, bins, mode
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
    pick_backend!(activate_gl)
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
    # valign=:center : centre le bloc image+colorbar verticalement dans sa
    # cellule de main_grid. Sans ce réglage, la row 1 est aussi haute que
    # spec_grid (spectre + histo + info ≈ 550 px) et l'espace vide s'accumule
    # au-dessus de l'image pour les cartes non carrées (DataAspect).
    # NB : on ne met PAS tellwidth=false ici — CompareBundle redimensionne
    # les colonnes 2/4 de img_grid dynamiquement et doit pouvoir propager
    # la largeur totale vers main_grid col 1.
    img_grid  = main_grid[1, 1] = GridLayout(; valign = :center)
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
    help_btn = Button(mode_bar[1, 4]; label = "Help", width = 78, height = 32)
    foreach(c -> colsize!(mode_bar, c, Auto()), 1:4)
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
    btn_copy_code = Button(output_card[3, 5]; label = "Copy code", width = 112, height = 32)
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

    hist_card = control_card!(analysis_bottom, 1, 6, "Histogram"; rows = 5, cols = 5)
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

    moment_card = control_card!(analysis_bottom, 1, 4, "Products"; rows = 4, cols = 5)
    moment_menu = Menu(moment_card[2, 1]; options = ["M0 integrated", "M1 mean", "M2 dispersion"], prompt = "M0 integrated", width = compact_layout ? 124 : 138)
    btn_show_moment = Button(moment_card[2, 2]; label = "Show", width = compact_layout ? 72 : 82, height = 32)
    btn_show_slice = Button(moment_card[2, 3]; label = "Slice", width = compact_layout ? 72 : 82, height = 32)
    btn_moment_png = Button(moment_card[2, 4]; label = "PNG", width = compact_layout ? 64 : 74, height = 32)
    btn_moment_fits = Button(moment_card[2, 5]; label = "FITS", width = compact_layout ? 64 : 74, height = 32)
    fits_product_menu = Menu(moment_card[3, 1:2]; options = ["slice", "region", "moment", "filtered cube", "mask"], prompt = "slice", width = compact_layout ? 136 : 150)
    btn_save_fits = Button(moment_card[3, 3]; label = "Export FITS", width = compact_layout ? 104 : 118, height = 32)
    foreach(c -> colsize!(moment_card, c, Auto()), 1:5)

    # ---- Mask card ----
    # Lives in analysis_bottom alongside Products/Histogram so the user can
    # build masks while inspecting moments and the histogram. Widget layout:
    #   row 2 : Source menu | Op menu | lo  | hi
    #   row 3 : i range box | j range | k range
    #   row 4 : Apply | Reset | Stats label
    mask_card = control_card!(analysis_bottom, 1, 2, "Mask"; rows = 4, cols = 5)
    control_label!(mask_card, (2, 1), "Source")
    mask_source_menu = Menu(mask_card[2, 2]; options = ["none", "finite", "threshold", "rectangle"], prompt = "none", width = compact_layout ? 114 : 128)
    mask_op_menu = Menu(mask_card[2, 3]; options = ["≥", "≤", "range", "outside"], prompt = "≥", width = compact_layout ? 88 : 100)
    mask_lo_box = Textbox(mask_card[2, 4]; placeholder = "lo", width = compact_layout ? 64 : 74, height = 32)
    mask_hi_box = Textbox(mask_card[2, 5]; placeholder = "hi", width = compact_layout ? 64 : 74, height = 32)
    mask_i_box = Textbox(mask_card[3, 1:2]; placeholder = "i range a:b", width = compact_layout ? 130 : 150, height = 32)
    mask_j_box = Textbox(mask_card[3, 3]; placeholder = "j range a:b", width = compact_layout ? 92 : 108, height = 32)
    mask_k_box = Textbox(mask_card[3, 4]; placeholder = "k range a:b", width = compact_layout ? 92 : 108, height = 32)
    mask_apply_btn = Button(mask_card[4, 1]; label = "Apply", width = compact_layout ? 72 : 82, height = 32)
    mask_reset_btn = Button(mask_card[4, 2]; label = "Reset", width = compact_layout ? 72 : 82, height = 32)
    mask_status_label = Label(mask_card[4, 3:5]; text = mask_status_obs[], halign = :left, tellwidth = false, fontsize = 13, color = ui_text_muted)
    foreach(c -> colsize!(mask_card, c, Auto()), 1:5)

    # Finalise analysis_bottom: transparent Boxes force spacer columns to exist
    # so colsize! can address them. Cards live in cols 2, 4, 6.
    Box(analysis_bottom[1, 1]; color = :transparent, strokewidth = 0)
    Box(analysis_bottom[1, 3]; color = :transparent, strokewidth = 0)
    Box(analysis_bottom[1, 5]; color = :transparent, strokewidth = 0)
    Box(analysis_bottom[1, 7]; color = :transparent, strokewidth = 0)
    colsize!(analysis_bottom, 1, Relative(1))
    colsize!(analysis_bottom, 2, Fixed(compact_layout ? 360 : 420))   # Mask
    colsize!(analysis_bottom, 3, Fixed(compact_layout ? 14 : 22))
    colsize!(analysis_bottom, 4, Fixed(compact_layout ? 460 : 520))   # Products
    colsize!(analysis_bottom, 5, Fixed(compact_layout ? 14 : 22))
    colsize!(analysis_bottom, 6, Fixed(compact_layout ? 430 : 500))   # Histogram
    colsize!(analysis_bottom, 7, Relative(1))

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
    style_button!(help_btn)
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
    style_button!(btn_copy_code)
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
    style_menu!(mask_source_menu)
    style_menu!(mask_op_menu)
    style_textbox!(mask_lo_box)
    style_textbox!(mask_hi_box)
    style_textbox!(mask_i_box)
    style_textbox!(mask_j_box)
    style_textbox!(mask_k_box)
    style_button!(mask_apply_btn)
    style_button!(mask_reset_btn)
    style_slider!(slice_slider)
    style_slider!(sigma_slider)

    if compact_layout
        for btn in (help_btn, reset_zoom_btn, reset_zoom_analysis_btn, reset_zoom_export_btn,
                    ps_btn, base_layout_btn, ps_refresh_btn, ps_popout_btn,
                    btn_save_img, btn_save_spec, btn_save_state, btn_load_state, btn_copy_code,
                    btn_show_compare, btn_load_compare, play_btn, anim_btn, clim_apply_btn,
                    clim_auto_btn, clim_p1_btn, clim_p5_btn, region_clear_btn, contour_apply_btn,
                    spec_y_apply_btn, spec_y_auto_btn, hist_apply_btn, hist_auto_btn, hist_y_apply_btn, hist_y_auto_btn,
                    btn_show_moment, btn_show_slice, btn_moment_png, btn_moment_fits, btn_save_fits,
                    mask_apply_btn, mask_reset_btn)
            btn.height[] = 30
            btn.fontsize[] = 13
            btn.padding[] = (9, 9, 5, 5)
        end
        for menu in (img_scale_menu, spec_scale_menu, cmap_menu, ps_src_menu, ps_win_menu,
                     ps_unit_menu, fmt_menu, compare_mode_menu, axis_menu, region_mode_menu, hist_mode_menu,
                     moment_menu, fits_product_menu, mask_source_menu, mask_op_menu)
            menu.height[] = 30
            menu.fontsize[] = 13
            menu.textpadding[] = (8, 8, 5, 5)
            menu.dropdown_arrow_size[] = 10
        end
        for tb in (ps_kmin_box, ps_kmax_box, clim_min_box, clim_max_box, fname_box,
                   compare_path_box, contour_levels_box, spec_ymin_box, spec_ymax_box,
                   hist_bins_box, hist_xmin_box, hist_xmax_box, hist_ymin_box, hist_ymax_box,
                   start_box, stop_box, step_box, fps_box,
                   mask_lo_box, mask_hi_box, mask_i_box, mask_j_box, mask_k_box)
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
    analysis_cards = (contrast_card, region_card, contour_card, hist_card, moment_card, mask_card)
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

    # ---------- Spectrum helpers (defined early — used by CompareBundle below) ----------
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
        mbits = mask_bits_obs[]
        if !isempty(region_uvs[])
            spec_x_raw[] = spec_x_axes[axis[]]
            y = mean_region_spectrum(data, axis[], region_uvs[]; mask = mbits)
            resize!(spec_y_buf, length(y))
            copyto!(spec_y_buf, y)
            ax_spec.title[] = latexstring("\\text{Mean spectrum in selected region}")
        elseif axis[] == 1
            spec_x_raw[] = spec_x_axes[1]
            resize!(spec_y_buf, siz[1])
            @views copyto!(spec_y_buf, data[:, j_idx[], k_idx[]])
            if mbits !== nothing
                @inbounds for c in 1:siz[1]
                    mbits[c, j_idx[], k_idx[]] || (spec_y_buf[c] = NaN32)
                end
            end
            ax_spec.title[] = L"\text{Spectrum at selected pixel}"
        elseif axis[] == 2
            spec_x_raw[] = spec_x_axes[2]
            resize!(spec_y_buf, siz[2])
            @views copyto!(spec_y_buf, data[i_idx[], :, k_idx[]])
            if mbits !== nothing
                @inbounds for c in 1:siz[2]
                    mbits[i_idx[], c, k_idx[]] || (spec_y_buf[c] = NaN32)
                end
            end
            ax_spec.title[] = L"\text{Spectrum at selected pixel}"
        else
            spec_x_raw[] = spec_x_axes[3]
            resize!(spec_y_buf, siz[3])
            @views copyto!(spec_y_buf, data[i_idx[], j_idx[], :])
            if mbits !== nothing
                @inbounds for c in 1:siz[3]
                    mbits[i_idx[], j_idx[], c] || (spec_y_buf[c] = NaN32)
                end
            end
            ax_spec.title[] = L"\text{Spectrum at selected pixel}"
        end
        spec_y_raw[] = spec_y_buf
        # Mirror the cube-A extraction on the comparison cube when loaded.
        # Same axis, same voxel / region — only the source array changes.
        cmp = compare_data[]
        if cmp !== nothing
            cmp_mask = (mbits !== nothing && size(cmp) == size(data)) ? mbits : nothing
            if !isempty(region_uvs[])
                y2 = mean_region_spectrum(cmp, axis[], region_uvs[]; mask = cmp_mask)
                resize!(spec_y_compare_buf, length(y2))
                copyto!(spec_y_compare_buf, y2)
            elseif axis[] == 1
                resize!(spec_y_compare_buf, siz[1])
                @views copyto!(spec_y_compare_buf, cmp[:, j_idx[], k_idx[]])
                if cmp_mask !== nothing
                    @inbounds for c in 1:siz[1]
                        cmp_mask[c, j_idx[], k_idx[]] || (spec_y_compare_buf[c] = NaN32)
                    end
                end
            elseif axis[] == 2
                resize!(spec_y_compare_buf, siz[2])
                @views copyto!(spec_y_compare_buf, cmp[i_idx[], :, k_idx[]])
                if cmp_mask !== nothing
                    @inbounds for c in 1:siz[2]
                        cmp_mask[i_idx[], c, k_idx[]] || (spec_y_compare_buf[c] = NaN32)
                    end
                end
            else
                resize!(spec_y_compare_buf, siz[3])
                @views copyto!(spec_y_compare_buf, cmp[i_idx[], j_idx[], :])
                if cmp_mask !== nothing
                    @inbounds for c in 1:siz[3]
                        cmp_mask[i_idx[], j_idx[], c] || (spec_y_compare_buf[c] = NaN32)
                    end
                end
            end
        else
            resize!(spec_y_compare_buf, length(spec_y_buf))
            fill!(spec_y_compare_buf, NaN32)
        end
        spec_y_compare_raw[] = spec_y_compare_buf
        refresh_spec_ylim!()
    end

    # ---------- Compare: loader UI + cube alignment ----------
    # See src/views/cube/CompareBundle.jl
    (; show_compare_loader!, hide_compare_loader!, resolve_compare_path, pick_compare_path, load_compare_cube!) =
        _cube_compare_bundle(;
            filepath, wcs, data, siz,
            compare_data, compare_visible, compare_name, compare_path_current,
            btn_show_compare, compare_mode_menu, compare_path_box, btn_load_compare,
            compare_state_label, ax_cmp, img_colorbar_cmp, img_grid,
            show_grid, ui_text_muted, ui_success,
            refresh_spectrum!, set_status!, set_block_visible!, set_box_text!,
        )

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
            # Reconstruction inline plutôt que de re-concat un LaTeXString déjà
            # entouré de $...$ : sinon LaTeXStrings.latexstring détecte les $
            # de tête/queue et insère la suite hors mode math, ce qui fait
            # planter MathTeXEngine (\quad, \mathbf, \^{} hors $$).
            info = string(
                "\\mathbf{pixel}\\,(i,j,k)=(",
                i_idx[], ",", j_idx[], ",", k_idx[], ")",
                "\\quad\\mathbf{slice}\\,(\\text{row},\\text{col})=(",
                u_idx[], ",", v_idx[], ")",
                "\\quad\\mathbf{intensity}= ",
                isnan(val) ? "NaN" : string(round(Float32(val); digits = 4)),
            )
            winfo = world_info_string()
            isempty(winfo) && return latexstring(info)
            # On enveloppe le bloc WCS dans \text{} pour que les échappements
            # text-mode produits par latex_safe (\^{}, \_, ...) soient interprétés
            # dans le bon contexte plutôt qu'envoyés bruts en mode math.
            return latexstring(info, "\\quad\\mathbf{WCS}\\,(\\text{", latex_safe(winfo), "})")
        else
            npx = length(region_uvs[])
            y = mean_region_spectrum(data, axis[], region_uvs[]; mask = mask_bits_obs[])
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

    # ---------- Mask: parsing helpers + apply/reset ----------
    # See src/views/cube/MaskBundle.jl
    (; _parse_int_range, _set_mask_status!, apply_mask_source!, reset_mask!, build_mask_source_from_ui) =
        _cube_mask_bundle(;
            data, _moment_cache,
            mask_source_obs, mask_bits_obs, mask_status_obs,
            mask_status_label,
            mask_lo_box, mask_hi_box, mask_i_box, mask_j_box, mask_k_box,
            mask_source_menu, mask_op_menu,
            refresh_spectrum!, set_status!, set_box_text!,
        )

    on_mode(mask_apply_btn.clicks, :analysis) do _
        src, err = build_mask_source_from_ui()
        if src === nothing
            set_status!("Mask not applied: $(err)")
            return
        end
        try
            apply_mask_source!(src)
            if src isa NoMaskSource
                set_status!("Mask cleared (source = none).")
            else
                set_status!("Mask applied. " * mask_status_obs[])
            end
        catch e
            msg = "Failed to build mask: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
        end
    end

    on_mode(mask_reset_btn.clicks, :analysis) do _
        reset_mask!()
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

    # ---------- Keyboard shortcuts + mouse pick ----------
    # See src/views/cube/KeyboardBundle.jl
    _cube_keyboard_bundle!(fig, ax_img;
        axis, idx, siz, u_idx, v_idx, i_idx, j_idx, k_idx, uv_point,
        slice_dims, uv_to_ijk,
        invert_cmap, img_scale_mode, show_contours, compare_visible, layout_mode,
        zoom_drag_active, zoom_drag_start, zoom_drag_end,
        region_drag_active, region_start, region_end, region_uvs, selection_mode, anim_playing,
        slice_slider, axis_menu, axes_labels, img_scale_menu, contour_chk,
        clim_auto_btn, btn_save_img, help_btn, region_count_label, ax_cmp,
        refresh_labels!, refresh_spectrum!, apply_percentile_clims!,
        render_power_spectrum_layout!, reset_zoom!, clear_region!, update_region_from_drag!,
        set_status!, bypass_mode_gate, syncing_slice_controls,
        textboxes = (clim_min_box, clim_max_box, fname_box, compare_path_box,
                     spec_ymin_box, spec_ymax_box, contour_levels_box,
                     hist_bins_box, hist_xmin_box, hist_xmax_box,
                     hist_ymin_box, hist_ymax_box,
                     start_box, stop_box, step_box, fps_box,
                     mask_lo_box, mask_hi_box, ps_kmin_box, ps_kmax_box),
        ui_theme,
    )

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
        # The mask source is the *declarative* description (kind + params);
        # the materialised BitArray is regenerated from `data` on reload.
        "mask" => mask_source_to_toml(mask_source_obs[]),
    )

    state_get(st, key::AbstractString, fallback) = begin
        if st isa AbstractDict
            return get(st, key, get(st, Symbol(key), fallback))
        elseif st isa NamedTuple
            sym = Symbol(key)
            return hasproperty(st, sym) ? getproperty(st, sym) : fallback
        else
            return fallback
        end
    end

    function apply_inline_state!(st; announce::Bool = true)
        st === nothing && return nothing

        axis_val = clamp(Int(state_get(st, "axis", axis[])), 1, 3)
        axis_menu.selection[] = axes_labels[axis_val]

        idx_val = clamp(Int(state_get(st, "index", idx[])), 1, siz[axis_val])
        slice_slider.value[] = idx_val

        img_scale_val = String(state_get(st, "img_scale", String(img_scale_mode[])))
        img_scale_val in ("lin", "log10", "ln") && (img_scale_menu.selection[] = img_scale_val)

        spec_scale_val = String(state_get(st, "spec_scale", String(spec_scale_mode[])))
        spec_scale_val in ("lin", "log10", "ln") && (spec_scale_menu.selection[] = spec_scale_val)

        cmap_val = Symbol(String(state_get(st, "colormap", String(cmap_name[]))))
        try
            to_cmap(cmap_val)
            cmap_name[] = cmap_val
            String(cmap_val) in MANTA_COLORMAP_OPTIONS && (cmap_menu.selection[] = String(cmap_val))
        catch
            @warn "Ignoring invalid colormap in inline state" colormap=cmap_val
        end

        invert_chk.checked[] = Bool(state_get(st, "invert_colormap", invert_cmap[]))
        crosshair_chk.checked[] = Bool(state_get(st, "show_crosshair", show_crosshair[]))
        marker_chk.checked[] = Bool(state_get(st, "show_marker", show_marker[]))
        grid_chk.checked[] = Bool(state_get(st, "show_grid", show_grid[]))
        contour_chk.checked[] = Bool(state_get(st, "show_contours", show_contours[]))

        use_manual_val = Bool(state_get(st, "use_manual_clims", use_manual[]))
        if use_manual_val
            cmin = Float32(state_get(st, "clim_min", first(clims_manual[])))
            cmax = Float32(state_get(st, "clim_max", last(clims_manual[])))
            ok, new_manual, parsed_clims, _ = parse_manual_clims(string(cmin), string(cmax); fallback = clims_manual[])
            if ok && new_manual
                clims_manual[] = parsed_clims
                use_manual[] = true
                set_box_text!(clim_min_box, string(first(parsed_clims)))
                set_box_text!(clim_max_box, string(last(parsed_clims)))
            end
        else
            use_manual[] = false
            set_box_text!(clim_min_box, "")
            set_box_text!(clim_max_box, "")
        end

        mask_dict = state_get(st, "mask", nothing)
        if mask_dict isa AbstractDict
            try
                apply_mask_source!(mask_source_from_toml(mask_dict))
            catch e
                @warn "Could not restore mask from inline state" exception=(e, catch_backtrace())
            end
        end

        announce && set_status!("Applied inline viewer state.")
        return nothing
    end

    function current_recipe_snippet()
        source_arg = filepath != "" ? filepath : JuliaExpr("data")
        kwargs_pairs = Pair{Symbol,Any}[
            :cmap => cmap_name[],
            :invert => invert_cmap[],
            :state => current_settings(),
        ]
        if compare_visible[] && !isempty(compare_path_current[])
            push!(kwargs_pairs, :compare => compare_path_current[])
        end
        return manta_recipe(source_arg; kwargs_pairs...)
    end

    make_name = function (base::AbstractString, ext::AbstractString)
        b = isempty(base) ? fname : base
        return "$(b)_axis$(axis[])_idx$(idx[])_i$(i_idx[])_j$(j_idx[])_k$(k_idx[])_img$(String(img_scale_mode[]))_spec$(String(spec_scale_mode[])).$(ext)"
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

    # ---------- Export: write_fits_array, save_moment_png!, export_fits_product!,
    # analysis/export mode callbacks (save image, spectrum, animation, GIF)
    # See src/views/cube/ExportBundle.jl
    _cube_export_bundle!(;
        ds, header, data, fname, save_root, make_name,
        axis, idx, siz, moment_order, sigma,
        moment_raw, slice_disp, slice_proc, cm_obs, clims_obs,
        contour_levels_obs, contour_colors_obs,
        region_uvs, mask_bits_obs, region_start, region_end, region_shape,
        u_idx, v_idx, uv_point, show_crosshair, show_marker, show_grid, show_contours,
        spec_x_raw, spec_y_disp, i_idx, j_idx, k_idx, spec_y_buf,
        unit_label, unit_label_tex, slice_axis_labels, slice_dims,
        anim_playing,
        btn_moment_png, btn_moment_fits, btn_save_fits, fits_product_menu,
        btn_save_img, btn_save_spec, play_btn, anim_btn,
        fmt_menu, fname_box, start_box, stop_box, step_box, fps_box,
        pingpong_chk, loop_chk, slice_slider, ax_spec,
        region_segments_from_points,
        on_mode, set_status!,
    )

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

    on_mode(btn_copy_code.clicks, :export) do _
        snippet = current_recipe_snippet()
        if copy_text_to_clipboard(snippet)
            btn_copy_code.labelcolor[] = ui_success
            set_status!("Copied reproducible MANTA.manta(...) recipe to clipboard.")
        else
            out = joinpath(save_root, "$(fname)_manta_recipe.jl")
            try
                open(out, "w") do io
                    write(io, snippet)
                    write(io, "\n")
                end
                set_status!("Clipboard unavailable; wrote recipe to $(out).")
            catch e
                msg = "Failed to copy recipe: $(sprint(showerror, e))"
                set_status!(msg)
                @error msg exception=(e, catch_backtrace())
            end
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

            mask_dict = get(st, "mask", nothing)
            try
                if mask_dict isa AbstractDict
                    src = mask_source_from_toml(mask_dict)
                    apply_mask_source!(src)
                    # Sync UI widgets to reflect the restored source so
                    # subsequent edits start from the right state.
                    if src isa NoMaskSource
                        mask_source_menu.selection[] = "none"
                    elseif src isa FiniteSource
                        mask_source_menu.selection[] = "finite"
                    elseif src isa ThresholdSource
                        mask_source_menu.selection[] = "threshold"
                        mask_op_menu.selection[] = src.op === :ge ? "≥" :
                                                   src.op === :le ? "≤" :
                                                   src.op === :range ? "range" : "outside"
                        set_box_text!(mask_lo_box, string(src.lo))
                        set_box_text!(mask_hi_box, string(src.hi))
                    elseif src isa RectangleSource
                        mask_source_menu.selection[] = "rectangle"
                        _rng(a, b) = (a === nothing && b === nothing) ? "" :
                                     "$(a === nothing ? "" : a):$(b === nothing ? "" : b)"
                        set_box_text!(mask_i_box, _rng(src.i1, src.i2))
                        set_box_text!(mask_j_box, _rng(src.j1, src.j2))
                        set_box_text!(mask_k_box, _rng(src.k1, src.k2))
                    end
                end
            catch e
                @warn "Could not restore mask from settings" exception=(e, catch_backtrace())
                apply_mask_source!(NoMaskSource())
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

    # ---------- Power spectrum window ----------
    ps_fig_ref = Ref{Any}(nothing)
    ps_alive_ref = Ref(false)

    # See src/views/cube/PSWindowBundle.jl
    (; open_power_spectrum_window!) = _cube_ps_window_bundle(;
        ps_fig_ref, ps_alive_ref,
        slice_proc, wcs, axis, slice_axis_dims,
        save_root, fname, fname_box, make_name,
        ui_text, ui_text_muted, ui_accent,
        style_menu!, style_button!, style_checkbox!, style_textbox!,
        set_status!, cm_obs,
    )

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
    if state !== nothing
        try
            apply_inline_state!(state; announce = false)
        catch e
            @warn "Failed to apply inline MANTA state" exception=(e, catch_backtrace())
        end
    end
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
