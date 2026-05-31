# path: src/views/cube/ExportBundle.jl
#
# Export / save callbacks for the 3D cube viewer.
#
# Extracted from src/views/CubeView.jl during a structural refactor
# (no behaviour change). Contains:
#   - `write_fits_array`      : low-level FITS writer
#   - `save_moment_png!`      : export the active moment map to PNG
#   - `export_fits_product!`  : dispatch slice / region / moment / cube / mask FITS
#   - analysis-mode export callbacks (moment PNG, moment FITS, generic FITS)
#   - `start_channel_animation!` : drive the interactive slice animation
#   - export-mode callbacks (save image, save spectrum, play animation, GIF)
#
# NOTE: the "Save/Load settings" and "Copy recipe" buttons remain in
# CubeView.jl because they poke nearly every widget in the viewer and are
# more "state management" than pure file export.
#
# Entry point: `_cube_export_bundle!(; kwargs...)`.
# Registers all `on_mode(…)` listeners as a side effect; returns nothing.

"""
    _cube_export_bundle!(; kwargs...)

Register all file-export callbacks for the cube viewer.  All callbacks
capture their dependencies through the keyword arguments.

Returns `nothing` — pure side effects (observer registrations).
"""
function _cube_export_bundle!(;
    # --- source identification ---
    ds,
    header,
    data,
    fname,
    save_root,
    make_name,
    # --- slice / state observables ---
    axis,
    idx,
    siz,
    moment_order,
    sigma,
    moment_raw,
    slice_disp,
    slice_proc,
    cm_obs,
    clims_obs,
    contour_levels_obs,
    contour_colors_obs,
    # --- selection / mask ---
    region_uvs,
    mask_bits_obs,
    region_start,
    region_end,
    region_shape,
    # --- display state ---
    u_idx,
    v_idx,
    uv_point,
    show_crosshair,
    show_marker,
    show_grid,
    show_contours,
    # --- spectrum state ---
    spec_x_raw,
    spec_y_disp,
    i_idx,
    j_idx,
    k_idx,
    spec_y_buf,
    # --- label helpers ---
    unit_label,
    unit_label_tex,
    slice_axis_labels,
    slice_dims,
    # --- animation state ---
    anim_playing,
    # --- widgets (analysis mode) ---
    btn_moment_png,
    btn_moment_fits,
    btn_save_fits,
    fits_product_menu,
    # --- widgets (export mode) ---
    btn_save_img,
    btn_save_spec,
    play_btn,
    anim_btn,
    fmt_menu,
    fname_box,
    start_box,
    stop_box,
    step_box,
    fps_box,
    pingpong_chk,
    loop_chk,
    slice_slider,
    ax_spec,
    # --- pure-function local helpers ---
    region_segments_from_points,
    # --- closures from earlier sections ---
    on_mode,
    set_status!,
)
    moment_label() = moment_order[] == 0 ? "moment0" :
                     moment_order[] == 1 ? "moment1" : "moment2"

    # ------------------------------------------------------------------
    # Write a Float32 array to a FITS primary HDU, optionally preserving
    # the cube's original header (WCS / BUNIT / provenance).
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Export the current moment map as a standalone PNG figure.
    # ------------------------------------------------------------------
    function save_moment_png!(out::AbstractString)
        f_mom = CairoMakie.Figure(size = (700, 560))
        colgap!(f_mom.layout, -8)
        xlab_s, ylab_s = slice_axis_labels(axis[])
        axM = CairoMakie.Axis(
            f_mom[1, 1];
            title    = latexstring("\\text{", latex_safe(fname), " ",
                                   latex_safe(moment_label()), " axis $(axis[])}"),
            xlabel   = xlab_s,
            ylabel   = ylab_s,
            aspect   = CairoMakie.DataAspect(),
            yreversed = true,
            xtickformat = latex_tick_formatter,
            ytickformat = latex_tick_formatter,
        )
        lim = clamped_extrema(moment_raw[])
        hmM = CairoMakie.heatmap!(axM, moment_raw[]; colormap = cm_obs[], colorrange = lim)
        CairoMakie.Colorbar(
            f_mom[1, 2],
            hmM;
            label      = moment_label(),
            width      = 20,
            height     = _axis_render_height(axM),
            tellheight = false,
            valign     = :center,
        )
        CairoMakie.save(String(out), f_mom; backend = CairoMakie)
        nothing
    end

    # ------------------------------------------------------------------
    # Dispatch FITS product export by product name.
    # ------------------------------------------------------------------
    function export_fits_product!(product::AbstractString, out::AbstractString)
        src_id = String(ds.source_id)
        mbits  = mask_bits_obs[]
        if product == "slice"
            hdr = fits_header_for_slice(header, axis[], idx[]; source_id = src_id)
            write_fits_array(out, get_slice(data, axis[], idx[]); header = hdr)
        elseif product == "region"
            if isempty(region_uvs[])
                throw(ArgumentError("select a box or circle region before exporting the averaged region FITS"))
            end
            spec = mean_region_spectrum(data, axis[], region_uvs[]; mask = mbits)
            hdr  = fits_header_for_region_spectrum(header, axis[], length(region_uvs[]);
                                                   source_id = src_id)
            write_fits_array(out, spec; header = hdr)
        elseif product == "moment"
            # moment_raw[] already incorporates the active mask.
            hdr = fits_header_for_moment(header, axis[], moment_order[]; source_id = src_id)
            write_fits_array(out, moment_raw[]; header = hdr)
        elseif product == "filtered cube"
            hdr = fits_header_for_filtered_cube(header, axis[], sigma[]; source_id = src_id)
            write_fits_array(out, filtered_cube_by_slice(data, axis[], sigma[]); header = hdr)
        elseif product == "mask"
            mbits === nothing && throw(ArgumentError("no mask is currently active — apply a mask before exporting"))
            # FITS booleans → Int8 (1 = retained, 0 = rejected).
            mask_i8 = Int8.(mbits)
            hdr = fits_header_for_slice(header, 3, 1; source_id = src_id)
            write_fits_array(out, mask_i8; header = hdr)
        else
            throw(ArgumentError("unknown FITS product: $(product)"))
        end
        nothing
    end

    # ------------------------------------------------------------------
    # Analysis-mode export callbacks
    # ------------------------------------------------------------------
    on_mode(btn_moment_png.clicks, :analysis) do _
        spawn_safely() do
            base = get_box_str(fname_box)
            base = isempty(base) ? "$(fname)_$(moment_label())" : base
            out  = joinpath(save_root, make_name(base, "png"))
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
            out  = joinpath(save_root, make_name(base, "fits"))
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
            product       = String(something(fits_product_menu.selection[], "slice"))
            clean_product = replace(product, " " => "_")
            base = get_box_str(fname_box)
            base = isempty(base) ? "$(fname)_$(clean_product)" : base
            out  = joinpath(save_root, make_name(base, "fits"))
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

    # ------------------------------------------------------------------
    # Save image (slice + colorbar + optional crosshair / contours)
    # ------------------------------------------------------------------
    on_mode(btn_save_img.clicks, :export) do _
        spawn_safely() do
            ext = String(something(fmt_menu.selection[], "png"))
            out = joinpath(save_root, make_name(get_box_str(fname_box), ext))
            try
                f_slice = CairoMakie.Figure(size = (700, 560))
                colgap!(f_slice.layout, -8)
                xlab_s, ylab_s = slice_axis_labels(axis[])
                axS = CairoMakie.Axis(
                    f_slice[1, 1];
                    title       = make_slice_title(fname, axis[], idx[]),
                    xlabel      = xlab_s,
                    ylabel      = ylab_s,
                    aspect      = CairoMakie.DataAspect(),
                    yreversed   = true,
                    xtickformat = latex_tick_formatter,
                    ytickformat = latex_tick_formatter,
                )
                hmS = CairoMakie.heatmap!(axS, slice_disp[];
                                          colormap = cm_obs[], colorrange = clims_obs[])
                if show_contours[] && !isempty(contour_levels_obs[])
                    CairoMakie.contour!(axS, slice_disp[];
                                        levels    = contour_levels_obs[],
                                        color     = contour_colors_obs[],
                                        linewidth = CONTOUR_LW)
                end
                axS.xgridvisible[] = show_grid[]
                axS.ygridvisible[] = show_grid[]
                if show_crosshair[]
                    u_max, v_max = slice_dims(axis[])
                    u, v = u_idx[], v_idx[]
                    _ch_segs = Point2f[
                        Point2f(1, u), Point2f(v_max, u),
                        Point2f(v, 1), Point2f(v, u_max),
                    ]
                    # Halo noir + trait blanc fin (cohérent avec la vue interactive).
                    CairoMakie.linesegments!(axS, _ch_segs;
                        color = (:black, CROSSHAIR_ALPHA_DARK),  linewidth = CROSSHAIR_LW_DARK,  linestyle = :solid)
                    CairoMakie.linesegments!(axS, _ch_segs;
                        color = (:white, CROSSHAIR_ALPHA_LIGHT), linewidth = CROSSHAIR_LW_LIGHT, linestyle = :solid)
                end
                if show_marker[]
                    CairoMakie.scatter!(axS, [Point2f(uv_point[]...)], markersize = MARKER_SIZE)
                end
                if !isempty(region_uvs[])
                    CairoMakie.lines!(
                        axS,
                        region_segments_from_points(region_start[], region_end[], region_shape[]);
                        color     = (RGBf(1.0, 0.78, 0.18), 0.98),
                        linewidth = REGION_LW,
                    )
                end
                CairoMakie.Colorbar(
                    f_slice[1, 2],
                    hmS;
                    label      = unit_label,
                    width      = 20,
                    height     = _axis_render_height(axS),
                    tellheight = false,
                    valign     = :center,
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

    # ------------------------------------------------------------------
    # Save spectrum (lines plot)
    # ------------------------------------------------------------------
    on_mode(btn_save_spec.clicks, :export) do _
        spawn_safely() do
            ext  = String(something(fmt_menu.selection[], "png"))
            base = get_box_str(fname_box)
            base = isempty(base) ? "$(fname)_spectrum" : base
            out  = joinpath(save_root, make_name(base, ext))
            try
                f_spec = CairoMakie.Figure(size = (600, 400))
                axP = CairoMakie.Axis(
                    f_spec[1, 1];
                    title  = isempty(region_uvs[]) ?
                             make_spec_title(i_idx[], j_idx[], k_idx[]) :
                             L"\text{Mean spectrum in selected region}",
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

    # ------------------------------------------------------------------
    # Interactive channel animation (play / pause)
    # ------------------------------------------------------------------

    # `current_animation_request` is a module-level helper in
    # src/views/cube/AnimationRequest.jl.

    function start_channel_animation!(frames::Vector{Int}, fps::Int)
        anim_playing[]   = true
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
                anim_playing[]   = false
                play_btn.label[] = "Play"
            end
        end
        nothing
    end

    on_mode(play_btn.clicks, :export) do _
        if anim_playing[]
            anim_playing[]   = false
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

    # ------------------------------------------------------------------
    # GIF export
    # ------------------------------------------------------------------
    on_mode(anim_btn.clicks, :export) do _
        a    = axis[]
        amax = siz[a]
        ok, frames, fps, msg = current_animation_request(
            a, siz, start_box, stop_box, step_box, fps_box, pingpong_chk,
        )
        set_status!(msg)
        if !ok
            @warn "Invalid GIF parameters" msg axis=a amax=amax
            return
        end
        outfile = joinpath(save_root, "$(fname).gif")
        ny, nx  = slice_dims(axis[])
        w_img   = 640
        h_img   = Int(round(w_img * ny / nx))
        extra_for_cb = 80
        fig_gif = CairoMakie.Figure(size = (w_img + extra_for_cb, h_img))
        colgap!(fig_gif.layout, -8)
        axG = CairoMakie.Axis(fig_gif[1, 1]; aspect = DataAspect(), yreversed = true)
        Makie.hidedecorations!(axG, grid = false)
        hmG = CairoMakie.heatmap!(axG, slice_disp; colormap = cm_obs, colorrange = clims_obs)
        CairoMakie.Colorbar(
            fig_gif[1, 2],
            hmG;
            label      = "intensity",
            width      = 20,
            height     = _axis_render_height(axG),
            tellheight = false,
            valign     = :center,
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

    return nothing
end
