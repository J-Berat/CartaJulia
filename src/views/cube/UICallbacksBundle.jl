# path: src/views/cube/UICallbacksBundle.jl
#
# Navigation / analysis / export UI callbacks for the 3D cube viewer.
#
# Extracted from src/views/CubeView.jl during a structural refactor
# (no behaviour change). Contains all the `on_mode` / `on` registrations
# for the control-panel widgets:
#
#   • Mode-tab buttons (navigation / analysis / export)
#   • Axis menu + slice slider (with syncing guard)
#   • Image scale, spectrum scale, colourmap menus
#   • Histogram (bins, x-limits, y-limits, mode)
#   • Spectrum y-axis limits
#   • Gaussian smoothing, crosshair/marker/grid checkboxes
#   • Comparison cube loader + display mode menu
#   • Contrast limits (manual, auto, percentile)
#   • Region selection + clear
#   • Contour overlay
#
# NOTE: the call site in CubeView.jl must appear AFTER
# `render_power_spectrum_layout!` is defined (inside the embedded PS layout
# section, ~line 1870) because two slice-navigation callbacks invoke it.
#
# Entry point: `_cube_ui_callbacks_bundle!(; kwargs...)`.
# Returns `syncing_slice_controls::Ref{Bool}` (needed by KeyboardBundle).

function _apply_invert_colormap_toggle!(v, invert_cmap)
    enabled = Bool(v)
    invert_cmap[] = enabled
    return enabled
end

function _apply_gaussian_smoothing_toggle!(v, gauss_on, refresh_spectrum!,
                                           layout_mode, render_power_spectrum_layout!,
                                           set_status!)
    enabled = Bool(v)
    gauss_on[] = enabled
    refresh_spectrum!()
    layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    set_status!(enabled ? "Gaussian smoothing enabled." : "Gaussian smoothing disabled.")
    return enabled
end

"""
    _cube_ui_callbacks_bundle!(; kwargs...) -> Ref{Bool}

Register all navigation/analysis/export UI callbacks for the cube viewer.
Returns `syncing_slice_controls`, a `Ref{Bool}` flag that prevents the
axis-menu and slice-slider callbacks from re-entering each other.

The `!` suffix signals that this function has side effects (observer
registrations only — no widgets are created here).
"""
function _cube_ui_callbacks_bundle!(;
    # --- mode infrastructure ---
    control_mode,
    on_mode,
    bypass_mode_gate,
    refresh_control_mode!,
    set_status!,
    set_box_text!,
    flag_box!,
    # --- mode buttons ---
    mode_nav_btn,
    mode_analysis_btn,
    mode_export_btn,
    # --- axis / slice ---
    axis,
    idx,
    compare_idx,
    siz,
    axes_labels,
    axis_menu,
    slice_slider,
    compare_slice_slider,
    compare_slice_label,
    uv_to_ijk,
    u_idx,
    v_idx,
    i_idx,
    j_idx,
    k_idx,
    clear_region!,
    refresh_all!,
    refresh_labels!,
    render_power_spectrum_layout!,
    layout_mode,
    view_product,
    data,
    slice_plot_preview,
    # --- scale / colourmap ---
    img_scale_mode,
    spec_scale_mode,
    cmap_name,
    img_scale_menu,
    spec_scale_menu,
    cmap_menu,
    # --- histogram ---
    hist_mode_obs,
    hist_bins_obs,
    hist_xlimits_manual,
    hist_xlimits_manual_value,
    hist_ylimits_manual,
    hist_ylimits_manual_value,
    hist_limits_obs,
    hist_y_obs,
    compare_hist_y_obs,
    hist_mode_menu,
    hist_apply_btn,
    hist_auto_btn,
    hist_y_apply_btn,
    hist_y_auto_btn,
    hist_bins_box,
    hist_xmin_box,
    hist_xmax_box,
    hist_ymin_box,
    hist_ymax_box,
    refresh_hist_axes!,
    # --- spectrum y-axis ---
    spec_ylimits_source,
    spec_ylimits_value,
    spec_y_apply_btn,
    spec_y_auto_btn,
    spec_ymin_box,
    spec_ymax_box,
    refresh_spec_ylim!,
    # --- image display toggles ---
    invert_cmap,
    gauss_on,
    show_crosshair,
    show_marker,
    show_grid,
    invert_chk,
    gauss_chk,
    crosshair_chk,
    marker_chk,
    grid_chk,
    ax_img,
    ax_cmp,
    ax_spec,
    refresh_spectrum!,
    # --- comparison cube ---
    compare_mode,
    compare_visible,
    compare_path_box,
    btn_show_compare,
    btn_load_compare,
    compare_mode_menu,
    pick_compare_path,
    load_compare_cube!,
    show_compare_loader!,
    # --- gaussian smoothing ---
    sigma,
    sigma_label,
    sigma_slider,
    # --- contrast limits ---
    use_manual,
    clims_manual,
    clims_safe,
    clim_apply_btn,
    clim_auto_btn,
    clim_fix_btn,
    clim_auto_nav_btn,
    clim_p1_btn,
    clim_p5_btn,
    clim_min_box,
    clim_max_box,
    apply_percentile_clims!,
    # --- region selection ---
    selection_mode,
    region_shape,
    region_mode_menu,
    region_clear_btn,
    # --- contour overlay ---
    show_contours,
    contour_use_manual,
    contour_manual_levels,
    contour_manual_colors,
    contour_chk,
    contour_apply_btn,
    contour_levels_box,
    asinh_softening::Real = ASINH_SOFTENING_DEFAULT,
    slice_debounce_seconds::Real = 0.12,
    slice_preview_max_pixels::Integer = 1024,
    slice_preview_min_interval::Real = 1 / 30,
)
    # ------------------------------------------------------------------ #
    # Helpers local to this bundle
    # ------------------------------------------------------------------ #
    compare_mode_label(mode::Symbol) = mode === :A ? "A" :
        mode === :B ? "B" :
        mode === :diff ? "A - B" :
        mode === :ratio ? "A / B" :
        "normalized residuals"

    # ------------------------------------------------------------------ #
    # Slice-control sync guard
    # ------------------------------------------------------------------ #
    # Prevents axis_menu and slice_slider from recursively updating each other.
    syncing_slice_controls = Ref(false)

    set_visible!(block, visible::Bool) = begin
        try; block.visible[] = visible; catch; end
        try; block.scene.visible[] = visible; catch; end
        try; block.blockscene.visible[] = visible; catch; end
        nothing
    end

    # ------------------------------------------------------------------ #
    # Debounced slice navigation
    # ------------------------------------------------------------------ #
    _slice_commit_generation = Ref(0)
    _last_preview_idx = Ref(idx[])
    _last_preview_time_ns = Ref(0)

    function _commit_slice_index!(n::Integer)
        n_clamped = clamp(Int(n), 1, siz[axis[]])
        if n_clamped == idx[]
            slice_plot_preview[] = nothing
            refresh_labels!()
            return nothing
        end
        idx[] = n_clamped
        ii, jj, kk = uv_to_ijk(u_idx[], v_idx[], axis[], n_clamped)
        i_idx[] = clamp(ii, 1, siz[1])
        j_idx[] = clamp(jj, 1, siz[2])
        k_idx[] = clamp(kk, 1, siz[3])
        refresh_labels!()
        refresh_spectrum!()
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
        nothing
    end

    function _preview_slice_index!(n::Integer)
        try
            if view_product[] !== :slice
                return nothing
            end
            if slice_preview_max_pixels <= 0
                return nothing
            end
            a = axis[]
            n_clamped = clamp(Int(n), 1, siz[a])
            now_ns = time_ns()
            min_interval_ns = round(Int, max(0, Float64(slice_preview_min_interval)) * 1e9)
            if n_clamped == _last_preview_idx[] || now_ns - _last_preview_time_ns[] < min_interval_ns
                return nothing
            end
            raw = get_slice_view(data, a, n_clamped)
            stride = downsample_factor(size(raw), slice_preview_max_pixels)
            small = stride == 1 ? Float32.(raw) : downsample_subsample(raw, stride)
            scaled = apply_scale_display(small, img_scale_mode[]; asinh_softening = asinh_softening)
            slice_plot_preview[] = (;
                x = downsample_centers(size(raw, 2), stride),
                y = downsample_centers(size(raw, 1), stride),
                z = permutedims(scaled),
            )
            _last_preview_idx[] = n_clamped
            _last_preview_time_ns[] = now_ns
        catch e
            @debug "Slice preview failed" exception=(e, catch_backtrace())
        end
        nothing
    end

    function _schedule_slice_commit!(n::Integer)
        _slice_commit_generation[] += 1
        generation = _slice_commit_generation[]
        delay = max(0, Float64(slice_debounce_seconds))
        @async begin
            try
                sleep(delay)
                generation == _slice_commit_generation[] || return
                _commit_slice_index!(n)
            catch e
                @warn "Debounced slice update failed" exception=(e, catch_backtrace())
            end
        end
        nothing
    end

    on(compare_visible; update = true) do visible
        show_compare_index = visible && control_mode[] === :navigation
        set_visible!(compare_slice_label, show_compare_index)
        set_visible!(compare_slice_slider, show_compare_index)
    end

    # ------------------------------------------------------------------ #
    # Mode-tab buttons
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Axis menu  — updates slicing axis + re-ranges the slider
    # ------------------------------------------------------------------ #
    on_mode(axis_menu.selection, :navigation) do sel
        sel === nothing && return
        new_axis = findfirst(==(String(sel)), axes_labels)
        new_axis === nothing && return
        _slice_commit_generation[] += 1
        slice_plot_preview[] = nothing
        axis[] = new_axis
        new_range = 1:siz[new_axis]
        syncing_slice_controls[] = true
        slice_slider.range[] = new_range
        compare_slice_slider.range[] = new_range
        idx[] = clamp(idx[], first(new_range), last(new_range))
        compare_idx[] = clamp(compare_idx[], first(new_range), last(new_range))
        slice_slider.value[] = idx[]      # move the thumb if out of bounds
        compare_slice_slider.value[] = compare_idx[]
        syncing_slice_controls[] = false
        ii, jj, kk = uv_to_ijk(u_idx[], v_idx[], axis[], idx[])
        i_idx[] = clamp(ii, 1, siz[1])
        j_idx[] = clamp(jj, 1, siz[2])
        k_idx[] = clamp(kk, 1, siz[3])
        clear_region!()
        refresh_all!()
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    # ------------------------------------------------------------------ #
    # Slice slider
    # ------------------------------------------------------------------ #
    on_mode(slice_slider.value, :navigation) do v
        syncing_slice_controls[] && return
        n = clamp(Int(round(v)), 1, siz[axis[]])
        if bypass_mode_gate[]
            slice_plot_preview[] = nothing
            _slice_commit_generation[] += 1
            _commit_slice_index!(n)
            return
        end
        _preview_slice_index!(n)
        set_status!("Preview slice $(n) / $(siz[axis[]]) along axis $(axis[]).")
        _schedule_slice_commit!(n)
    end

    on_mode(compare_slice_slider.value, :navigation) do v
        syncing_slice_controls[] && return
        compare_idx[] = Int(round(v))
        set_status!("Compare slice $(compare_idx[]) / $(siz[axis[]]) along axis $(axis[]).")
    end

    # ------------------------------------------------------------------ #
    # Scale and colourmap menus
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Histogram callbacks
    # ------------------------------------------------------------------ #
    on_mode(hist_mode_menu.selection, :analysis) do sel
        sel === nothing && return
        hist_mode_obs[] = normalize_histogram_mode(sel)
        set_status!("Histogram mode set to $(String(hist_mode_obs[])).")
    end

    on_mode(hist_apply_btn.clicks, :analysis) do _
        ok_bins, bins, bins_msg = parse_histogram_bins(
            get_box_str(hist_bins_box); fallback = hist_bins_obs[])
        ok_x, manual_x, xlim, x_msg = parse_histogram_xlimits(
            get_box_str(hist_xmin_box),
            get_box_str(hist_xmax_box);
            fallback = hist_xlimits_manual_value[])
        flag_box!(hist_bins_box, ok_bins)
        flag_box!(hist_xmin_box, ok_x)
        flag_box!(hist_xmax_box, ok_x)
        if !ok_bins
            set_status!(bins_msg); return
        end
        if !ok_x
            set_status!(x_msg); return
        end
        hist_bins_obs[]             = bins
        hist_xlimits_manual_value[] = xlim
        hist_xlimits_manual[]       = manual_x
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
        flag_box!(hist_xmin_box, true)
        flag_box!(hist_xmax_box, true)
        set_status!("Automatic histogram x-axis enabled.")
    end

    on_mode(hist_y_auto_btn.clicks, :analysis) do _
        hist_ylimits_manual[] = false
        set_box_text!(hist_ymin_box, "")
        set_box_text!(hist_ymax_box, "")
        flag_box!(hist_ymin_box, true)
        flag_box!(hist_ymax_box, true)
        refresh_hist_axes!()
        set_status!("Automatic histogram y-axis enabled.")
    end

    on_mode(hist_y_apply_btn.clicks, :analysis) do _
        ok_y, manual_y, ylim, y_msg = parse_histogram_ylimits(
            get_box_str(hist_ymin_box),
            get_box_str(hist_ymax_box);
            fallback = hist_ylimits_manual_value[])
        flag_box!(hist_ymin_box, ok_y)
        flag_box!(hist_ymax_box, ok_y)
        set_status!(y_msg)
        ok_y || return
        hist_ylimits_manual_value[] = ylim
        hist_ylimits_manual[]       = manual_y
        if manual_y
            set_box_text!(hist_ymin_box, string(first(ylim)))
            set_box_text!(hist_ymax_box, string(last(ylim)))
        else
            set_box_text!(hist_ymin_box, "")
            set_box_text!(hist_ymax_box, "")
        end
        refresh_hist_axes!()
    end

    # Histogram auto-refresh on computed limit / data changes
    on(hist_limits_obs)       do _ ; refresh_hist_axes!() end
    on(hist_y_obs)            do _ ; refresh_hist_axes!() end
    on(compare_hist_y_obs)    do _ ; refresh_hist_axes!() end

    # ------------------------------------------------------------------ #
    # Spectrum y-axis callbacks
    # ------------------------------------------------------------------ #
    on_mode(spec_y_apply_btn.clicks, :analysis) do _
        ok, manual, ylim, msg = parse_spectrum_ylimits(
            get_box_str(spec_ymin_box),
            get_box_str(spec_ymax_box);
            fallback = spec_ylimits_value[])
        flag_box!(spec_ymin_box, ok)
        flag_box!(spec_ymax_box, ok)
        set_status!(msg)
        ok || return
        if manual
            spec_ylimits_value[]  = ylim
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
        flag_box!(spec_ymin_box, true)
        flag_box!(spec_ymax_box, true)
        refresh_spec_ylim!()
        set_status!("Automatic spectrum y-axis enabled.")
    end

    # ------------------------------------------------------------------ #
    # Display-toggle checkboxes
    # ------------------------------------------------------------------ #
    on(invert_chk.checked) do v
        _apply_invert_colormap_toggle!(v, invert_cmap)
    end

    on(gauss_chk.checked) do v
        _apply_gaussian_smoothing_toggle!(
            v, gauss_on, refresh_spectrum!,
            layout_mode, render_power_spectrum_layout!, set_status!)
    end

    on_mode(crosshair_chk.checked, :navigation) do v
        show_crosshair[] = v
    end

    on_mode(marker_chk.checked, :navigation) do v
        show_marker[] = v
    end

    on_mode(grid_chk.checked, :navigation) do v
        show_grid[] = v
        ax_img.xgridvisible[] = v; ax_img.ygridvisible[] = v
        ax_cmp.xgridvisible[] = v; ax_cmp.ygridvisible[] = v
        ax_spec.xgridvisible[] = v; ax_spec.ygridvisible[] = v
    end

    # ------------------------------------------------------------------ #
    # Comparison cube loader
    # ------------------------------------------------------------------ #
    on_mode(btn_show_compare.clicks, :export) do _
        # Try a native file picker first. If it returns a path we load
        # immediately; otherwise fall back to the legacy textbox loader
        # (useful in headless CI or when the user cancels).
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

    # ------------------------------------------------------------------ #
    # Gaussian-smoothing slider
    # ------------------------------------------------------------------ #
    on_mode(sigma_slider.value, :navigation) do v
        sigma[] = Float32(v)
        sigma_label.text[] = latexstring("\\sigma = $(round(v; digits = 2))\\,\\text{px}")
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    # ------------------------------------------------------------------ #
    # Contrast / colour-limit callbacks
    # ------------------------------------------------------------------ #
    function apply_auto_contrast!()
        use_manual[] = false
        set_box_text!(clim_min_box, "")
        set_box_text!(clim_max_box, "")
        flag_box!(clim_min_box, true)
        flag_box!(clim_max_box, true)
        if spec_ylimits_source[] === :contrast
            spec_ylimits_source[] = :auto
            set_box_text!(spec_ymin_box, "")
            set_box_text!(spec_ymax_box, "")
            flag_box!(spec_ymin_box, true)
            flag_box!(spec_ymax_box, true)
        end
        refresh_spec_ylim!()
        set_status!("Automatic contrast enabled.")
        nothing
    end

    function fix_current_colorbar!()
        cmin, cmax = clims_safe[]
        fixed_clims = (Float32(cmin), Float32(cmax))
        clims_manual[] = fixed_clims
        use_manual[] = true
        set_box_text!(clim_min_box, string(first(fixed_clims)))
        set_box_text!(clim_max_box, string(last(fixed_clims)))
        flag_box!(clim_min_box, true)
        flag_box!(clim_max_box, true)
        if spec_ylimits_source[] === :contrast
            spec_ylimits_value[] = fixed_clims
            set_box_text!(spec_ymin_box, string(first(fixed_clims)))
            set_box_text!(spec_ymax_box, string(last(fixed_clims)))
            flag_box!(spec_ymin_box, true)
            flag_box!(spec_ymax_box, true)
        end
        refresh_spec_ylim!()
        set_status!("Colorbar fixed to current limits.")
        nothing
    end

    on_mode(clim_fix_btn.clicks, :navigation) do _
        fix_current_colorbar!()
    end

    on_mode(clim_auto_nav_btn.clicks, :navigation) do _
        apply_auto_contrast!()
    end

    on_mode(clim_apply_btn.clicks, :analysis) do _
        txtmin = get_box_str(clim_min_box)
        txtmax = get_box_str(clim_max_box)
        ok, new_manual, parsed_clims, msg = parse_manual_clims(
            txtmin, txtmax; fallback = clims_manual[])
        flag_box!(clim_min_box, ok)
        flag_box!(clim_max_box, ok)
        set_status!(msg)
        if !ok
            @warn "Could not apply contrast limits" txtmin txtmax msg; return
        end
        if new_manual
            clims_manual[] = parsed_clims
            use_manual[]   = true
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
        apply_auto_contrast!()
    end

    on_mode(clim_p1_btn.clicks, :analysis) do _
        apply_percentile_clims!(1, 99)
    end

    on_mode(clim_p5_btn.clicks, :analysis) do _
        apply_percentile_clims!(5, 95)
    end

    # ------------------------------------------------------------------ #
    # Region-selection callbacks
    # ------------------------------------------------------------------ #
    on_mode(region_mode_menu.selection, :analysis) do sel
        sel === nothing && return
        mode = Symbol(String(sel))
        if mode in (:point, :box, :circle)
            selection_mode[] = mode
            region_shape[]   = mode === :circle ? :circle : :box
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

    # ------------------------------------------------------------------ #
    # Contour overlay callbacks
    # ------------------------------------------------------------------ #
    on_mode(contour_chk.checked, :analysis) do v
        show_contours[] = v
        set_status!(v ? "Contours enabled." : "Contours hidden.")
    end

    on_mode(contour_apply_btn.clicks, :analysis) do _
        ok, use_man, levels, colors, msg = parse_contour_specs(
            get_box_str(contour_levels_box);
            fallback_levels = contour_manual_levels[],
            fallback_colors = contour_manual_colors[])
        flag_box!(contour_levels_box, ok)
        set_status!(msg)
        if !ok
            @warn "Could not apply contour levels" msg; return
        end
        contour_use_manual[]    = use_man
        contour_manual_levels[] = levels
        contour_manual_colors[] = colors
        if use_man
            set_box_text!(contour_levels_box,
                format_contour_specs(levels, colors))
        else
            set_box_text!(contour_levels_box, "")
        end
        show_contours[]        = true
        contour_chk.checked[]  = true
    end

    return syncing_slice_controls
end
