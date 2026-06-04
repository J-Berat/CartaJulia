# path: src/views/cube/SettingsBundle.jl
#
# Viewer-state persistence helpers for the 3D cube viewer.
#
# Extracted from src/views/CubeView.jl during a structural refactor
# (no behaviour change). Contains:
#
#   save_root              — resolved output directory (Desktop or pwd())
#   resolved_settings_path — absolute path to the TOML settings file
#   make_name              — filename generator (embeds axis/idx/scale)
#   current_settings()     — snapshot of the current viewer state as a Dict
#   apply_inline_state!()  — replay a state dict / NamedTuple on the viewer
#   current_recipe_snippet() — Julia manta(…) code that reproduces the view
#
# Also registers three on_mode callbacks (btn_save_state, btn_copy_code,
# btn_load_state) as a side effect.
#
# NOTE: `apply_inline_state!` must be called AFTER `_cube_mask_bundle`
# because it uses `apply_mask_source!` from that bundle.
#
# Entry point: `_cube_settings_bundle!(; kwargs...)`.
# Returns `(; save_root, resolved_settings_path, make_name, current_settings,
#             apply_inline_state!, current_recipe_snippet)`.

"""
    _cube_settings_bundle!(; kwargs...) -> NamedTuple

Set up save-root resolution, state helpers, and the save/load/copy callbacks
for the cube viewer.  Registering the `on_mode` listeners is a side effect;
all pure helper functions are returned in the named tuple.

The `!` suffix signals that this function has side effects (observer
registrations).  Unlike the pure-data bundles (`SlicePipelineBundle`,
`SpectrumBundle`), it pokes Makie widgets and must run after those bundles.
"""
function _cube_settings_bundle!(;
    # --- source identification ---
    filepath,
    fname,
    # --- viewer kwargs forwarded verbatim ---
    save_dir,
    settings_path,
    # --- observable state ---
    axis,
    idx,
    compare_idx,
    siz,
    img_scale_mode,
    spec_scale_mode,
    cmap_name,
    invert_cmap,
    show_crosshair,
    show_marker,
    show_grid,
    show_contours,
    contour_use_manual,
    contour_manual_levels,
    contour_manual_colors,
    use_manual,
    clims_manual,
    clims_auto,
    mask_source_obs,
    compare_visible,
    compare_path_current,
    spec_ylimits_source,
    spec_ylimits_value,
    spec_y_buf,
    refresh_spec_ylim!,
    # --- voxel indices (used by make_name) ---
    i_idx, j_idx, k_idx,
    # --- axes ---
    ax_spec,
    # --- menu / widget objects (selections poked by apply_inline_state!) ---
    axes_labels,
    axis_menu,
    slice_slider,
    compare_slice_slider,
    img_scale_menu,
    spec_scale_menu,
    cmap_menu,
    invert_chk,
    crosshair_chk,
    marker_chk,
    grid_chk,
    contour_chk,
    contour_levels_box,
    clim_min_box,
    clim_max_box,
    spec_ymin_box,
    spec_ymax_box,
    mask_source_menu,
    mask_op_menu,
    mask_lo_box,
    mask_hi_box,
    mask_i_box,
    mask_j_box,
    mask_k_box,
    # --- export-mode buttons ---
    btn_save_state,
    btn_copy_code,
    btn_load_state,
    # --- mode infrastructure ---
    control_mode,
    bypass_mode_gate,
    on_mode,
    # --- helper closures from other bundles ---
    apply_mask_source!,
    set_status!,
    set_box_text!,
    # --- UI theme constants ---
    ui_success,
)
    # ------------------------------------------------------------------ #
    # Resolve the output directory
    # ------------------------------------------------------------------ #
    default_desktop = joinpath(homedir(), "Desktop")
    save_root = if save_dir === nothing
        isdir(default_desktop) ? default_desktop : pwd()
    else
        path = String(save_dir)
        isdir(path) || mkpath(path)
        path
    end

    resolved_settings_path = settings_path === nothing ?
        joinpath(save_root, "$(fname)_viewer_settings.toml") :
        abspath(String(settings_path))

    # ------------------------------------------------------------------ #
    # make_name  — filename with embedded viewer state tokens
    # ------------------------------------------------------------------ #
    make_name = function (base::AbstractString, ext::AbstractString)
        b = isempty(base) ? fname : base
        return "$(b)_axis$(axis[])_idx$(idx[])_i$(i_idx[])_j$(j_idx[])_k$(k_idx[])_img$(String(img_scale_mode[]))_spec$(String(spec_scale_mode[])).$(ext)"
    end

    # ------------------------------------------------------------------ #
    # current_settings  — declarative snapshot of the viewer state
    # ------------------------------------------------------------------ #
    current_settings() = Dict{String,Any}(
        "axis"              => axis[],
        "index"             => idx[],
        "compare_index"     => compare_idx[],
        "img_scale"         => String(img_scale_mode[]),
        "spec_scale"        => String(spec_scale_mode[]),
        "colormap"          => String(cmap_name[]),
        "invert_colormap"   => invert_cmap[],
        "show_crosshair"    => show_crosshair[],
        "show_marker"       => show_marker[],
        "show_grid"         => show_grid[],
        "show_contours"     => show_contours[],
        "contour_use_manual"=> contour_use_manual[],
        "contour_levels"    => collect(contour_manual_levels[]),
        "contour_colors"    => collect(contour_manual_colors[]),
        "use_manual_clims"  => use_manual[],
        "clim_min"          => use_manual[] ? first(clims_manual[]) : first(clims_auto[]),
        "clim_max"          => use_manual[] ? last(clims_manual[])  : last(clims_auto[]),
        # Spectrum y-axis policy: :auto (autolimits), :manual (user limits)
        # or :contrast (limits track the colour-scale clims). The numeric
        # bounds are only meaningful for the manual/contrast sources but are
        # always stored so a saved manual range survives an auto round-trip.
        "spec_ylimits_source" => String(spec_ylimits_source[]),
        "spec_ymin"           => first(spec_ylimits_value[]),
        "spec_ymax"           => last(spec_ylimits_value[]),
        # The mask source is the *declarative* description (kind + params);
        # the materialised BitArray is regenerated from `data` on reload.
        "mask"              => mask_source_to_toml(mask_source_obs[]),
    )

    # ------------------------------------------------------------------ #
    # state_get  — safe key lookup that accepts both Dict and NamedTuple
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # _safe_* helpers  — type-checked TOML field accessors
    #
    # Each returns the fallback value and emits @warn instead of throwing
    # when the stored value cannot be converted to the target type.  This
    # matches the permissive strategy used by `mask_source_from_toml` and
    # ensures that a single corrupt field never aborts the whole load.
    # ------------------------------------------------------------------ #
    function _safe_int(val, fb::Int)::Int
        try
            return Int(val)
        catch
            @warn "MANTA settings: expected integer, using fallback" value=val fallback=fb
            return fb
        end
    end

    function _safe_float32(val, fb::Float32)::Float32
        try
            return Float32(val)
        catch
            @warn "MANTA settings: expected number, using fallback" value=val fallback=fb
            return fb
        end
    end

    function _safe_bool(val, fb::Bool)::Bool
        try
            return Bool(val)
        catch
            @warn "MANTA settings: expected boolean, using fallback" value=val fallback=fb
            return fb
        end
    end

    # ------------------------------------------------------------------ #
    # _restore_spec_ylimits!  — replay the persisted spectrum y-axis policy
    #
    # Shared by both `apply_inline_state!` (recipe `state=`) and
    # `_apply_loaded_settings!` (TOML load) so the two paths stay in sync.
    # Accepts the same Dict/NamedTuple shape via `state_get`. Unknown or
    # missing sources fall back to :auto. The numeric bounds are routed
    # through `parse_spectrum_ylimits` so they are normalised/ordered the
    # same way as live UI input before being applied.
    # ------------------------------------------------------------------ #
    function _restore_spec_ylimits!(st)
        src = lowercase(String(state_get(st, "spec_ylimits_source", String(spec_ylimits_source[]))))
        if src == "manual" || src == "contrast"
            ymin = _safe_float32(state_get(st, "spec_ymin", first(spec_ylimits_value[])), first(spec_ylimits_value[]))
            ymax = _safe_float32(state_get(st, "spec_ymax", last(spec_ylimits_value[])),  last(spec_ylimits_value[]))
            ok, manual, ylim, _ = parse_spectrum_ylimits(string(ymin), string(ymax); fallback = spec_ylimits_value[])
            if ok && manual
                spec_ylimits_value[]  = ylim
                spec_ylimits_source[] = src == "contrast" ? :contrast : :manual
                set_box_text!(spec_ymin_box, string(first(ylim)))
                set_box_text!(spec_ymax_box, string(last(ylim)))
            else
                spec_ylimits_source[] = :auto
                set_box_text!(spec_ymin_box, "")
                set_box_text!(spec_ymax_box, "")
            end
        else
            spec_ylimits_source[] = :auto
            set_box_text!(spec_ymin_box, "")
            set_box_text!(spec_ymax_box, "")
        end
        refresh_spec_ylim!()
        nothing
    end

    # ------------------------------------------------------------------ #
    # apply_inline_state!  — replay a Dict / NamedTuple on the live viewer
    # ------------------------------------------------------------------ #
    # Writes widget selections, Observables, and textbox texts so the viewer
    # looks as if the user had interacted with it.  Tolerant of missing keys
    # (falls back to the current value) and invalid values (warns and skips).
    function apply_inline_state!(st; announce::Bool = true)
        st === nothing && return nothing

        axis_val = clamp(_safe_int(state_get(st, "axis", axis[]), axis[]), 1, 3)
        axis_menu.selection[] = axes_labels[axis_val]

        idx_val = clamp(_safe_int(state_get(st, "index", idx[]), idx[]), 1, siz[axis_val])
        slice_slider.value[] = idx_val
        compare_idx_val = clamp(_safe_int(state_get(st, "compare_index", compare_idx[]), compare_idx[]), 1, siz[axis_val])
        compare_slice_slider.value[] = compare_idx_val

        img_scale_val = String(state_get(st, "img_scale", String(img_scale_mode[])))
        img_scale_val in scale_menu_options() && (img_scale_menu.selection[] = img_scale_val)

        spec_scale_val = String(state_get(st, "spec_scale", String(spec_scale_mode[])))
        spec_scale_val in scale_menu_options() && (spec_scale_menu.selection[] = spec_scale_val)

        cmap_val = Symbol(String(state_get(st, "colormap", String(cmap_name[]))))
        try
            to_cmap(cmap_val)
            cmap_name[] = cmap_val
            String(cmap_val) in MANTA_COLORMAP_OPTIONS && (cmap_menu.selection[] = String(cmap_val))
        catch
            @warn "Ignoring invalid colormap in inline state" colormap=cmap_val
        end

        invert_chk.checked[]    = _safe_bool(state_get(st, "invert_colormap", invert_cmap[]),  invert_cmap[])
        crosshair_chk.checked[] = _safe_bool(state_get(st, "show_crosshair",  show_crosshair[]), show_crosshair[])
        marker_chk.checked[]    = _safe_bool(state_get(st, "show_marker",     show_marker[]), show_marker[])
        grid_chk.checked[]      = _safe_bool(state_get(st, "show_grid",       show_grid[]), show_grid[])
        contour_chk.checked[]   = _safe_bool(state_get(st, "show_contours",   show_contours[]), show_contours[])

        use_manual_val = _safe_bool(state_get(st, "use_manual_clims", use_manual[]), use_manual[])
        if use_manual_val
            cmin = _safe_float32(state_get(st, "clim_min", first(clims_manual[])), first(clims_manual[]))
            cmax = _safe_float32(state_get(st, "clim_max", last(clims_manual[])),  last(clims_manual[]))
            ok, new_manual, parsed_clims, _ = parse_manual_clims(string(cmin), string(cmax); fallback = clims_manual[])
            if ok && new_manual
                clims_manual[] = parsed_clims
                use_manual[]   = true
                set_box_text!(clim_min_box, string(first(parsed_clims)))
                set_box_text!(clim_max_box, string(last(parsed_clims)))
            end
        else
            use_manual[] = false
            set_box_text!(clim_min_box, "")
            set_box_text!(clim_max_box, "")
        end

        # Spectrum y-axis policy is restored after the clims so a :contrast
        # source replays against the just-applied colour-scale limits.
        _restore_spec_ylimits!(st)

        mask_dict = state_get(st, "mask", nothing)
        if mask_dict isa AbstractDict
            try
                apply_mask_source!(mask_source_from_toml(mask_dict))
            catch e
                @warn "Could not restore mask from inline state" exception=(e, catch_backtrace())
                apply_mask_source!(NoMaskSource())
            end
        end

        announce && set_status!("Applied inline viewer state.")
        return nothing
    end

    # ------------------------------------------------------------------ #
    # current_recipe_snippet  — Julia one-liner to reproduce the current view
    # ------------------------------------------------------------------ #
    function current_recipe_snippet()
        source_arg = filepath != "" ? filepath : JuliaExpr("data")
        kwargs_pairs = Pair{Symbol,Any}[
            :cmap   => cmap_name[],
            :invert => invert_cmap[],
            :state  => current_settings(),
        ]
        if compare_visible[] && !isempty(compare_path_current[])
            push!(kwargs_pairs, :compare => compare_path_current[])
        end
        return manta_recipe(source_arg; kwargs_pairs...)
    end

    # ------------------------------------------------------------------ #
    # load_state_callback  — shared body for btn_load_state
    # (extracted as a named function to keep the on_mode closure short)
    # ------------------------------------------------------------------ #
    function _apply_loaded_settings!(st)
        axis_val = clamp(_safe_int(get(st, "axis", axis[]), axis[]), 1, 3)
        axis_menu.selection[] = axes_labels[axis_val]

        idx_val = clamp(_safe_int(get(st, "index", idx[]), idx[]), 1, siz[axis_val])
        slice_slider.value[] = idx_val
        compare_idx_val = clamp(_safe_int(get(st, "compare_index", compare_idx[]), compare_idx[]), 1, siz[axis_val])
        compare_slice_slider.value[] = compare_idx_val

        img_scale_val = String(get(st, "img_scale", String(img_scale_mode[])))
        img_scale_val in scale_menu_options() && (img_scale_menu.selection[] = img_scale_val)

        spec_scale_val = String(get(st, "spec_scale", String(spec_scale_mode[])))
        spec_scale_val in scale_menu_options() && (spec_scale_menu.selection[] = spec_scale_val)

        cmap_val = Symbol(String(get(st, "colormap", String(cmap_name[]))))
        try
            to_cmap(cmap_val)
            cmap_name[] = cmap_val
            String(cmap_val) in MANTA_COLORMAP_OPTIONS && (cmap_menu.selection[] = String(cmap_val))
        catch
            @warn "Ignoring invalid colormap in settings" colormap=cmap_val
        end

        invert_chk.checked[]    = _safe_bool(get(st, "invert_colormap", invert_cmap[]),  invert_cmap[])
        crosshair_chk.checked[] = _safe_bool(get(st, "show_crosshair",  show_crosshair[]), show_crosshair[])
        marker_chk.checked[]    = _safe_bool(get(st, "show_marker",     show_marker[]), show_marker[])
        grid_chk.checked[]      = _safe_bool(get(st, "show_grid",       show_grid[]), show_grid[])
        contour_chk.checked[]   = _safe_bool(get(st, "show_contours",   show_contours[]), show_contours[])
        contour_use_manual[]    = _safe_bool(get(st, "contour_use_manual", contour_use_manual[]), contour_use_manual[])
        raw_levels = get(st, "contour_levels", contour_manual_levels[])
        raw_colors = get(st, "contour_colors", contour_manual_colors[])
        contour_manual_levels[] = try
            Float32.(raw_levels)
        catch
            @warn "MANTA settings: ignoring non-numeric contour_levels" value=raw_levels
            contour_manual_levels[]
        end
        contour_manual_colors[] = try
            String.(raw_colors)
        catch
            @warn "MANTA settings: ignoring non-string contour_colors" value=raw_colors
            contour_manual_colors[]
        end
        if contour_use_manual[] && !isempty(contour_manual_levels[])
            set_box_text!(contour_levels_box,
                format_contour_specs(contour_manual_levels[], contour_manual_colors[]))
        else
            set_box_text!(contour_levels_box, "")
        end

        mask_dict = get(st, "mask", nothing)
        try
            if mask_dict isa AbstractDict
                src = mask_source_from_toml(mask_dict)
                apply_mask_source!(src)
                # Sync mask UI widgets to reflect the restored source so
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

        use_manual_val = _safe_bool(get(st, "use_manual_clims", use_manual[]), use_manual[])
        if use_manual_val
            cmin = _safe_float32(get(st, "clim_min", first(clims_manual[])), first(clims_manual[]))
            cmax = _safe_float32(get(st, "clim_max", last(clims_manual[])),  last(clims_manual[]))
            ok, new_manual, parsed_clims, msg = parse_manual_clims(
                string(cmin), string(cmax); fallback = clims_manual[])
            if ok && new_manual
                clims_manual[] = parsed_clims
                use_manual[]   = true
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

        # Restore the spectrum y-axis policy last so it has the final say on
        # ax_spec limits (overriding the contrast-driven defaults above when
        # the user had pinned a manual/contrast range).
        _restore_spec_ylimits!(st)
    end

    # ------------------------------------------------------------------ #
    # on_mode callbacks — save / copy / load
    # ------------------------------------------------------------------ #
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
            _apply_loaded_settings!(st)
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

    return (;
        save_root,
        resolved_settings_path,
        make_name,
        current_settings,
        apply_inline_state!,
        current_recipe_snippet,
    )
end
