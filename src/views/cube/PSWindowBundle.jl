# path: src/views/cube/PSWindowBundle.jl
#
# Pop-out power-spectrum window for the 3D cube viewer.
#
# Extracted from src/views/CubeView.jl during a structural refactor
# (no behaviour change). Contains the full `open_power_spectrum_window!`
# closure — an independent Makie Figure with its own UI, callbacks, and
# export buttons that is opened on demand from the main viewer.
#
# Entry point: `_cube_ps_window_bundle(; ...)` returns
# `(; open_power_spectrum_window!)`.
#
# NOTE: this file covers only the **pop-out window**.  The embedded
# power-spectrum layout (inline in the main figure) remains in
# CubeView.jl because it is too tightly coupled to the main grid.

"""
    _cube_ps_window_bundle(; kwargs...) -> NamedTuple

Return `(; open_power_spectrum_window!)`, a closure that opens (or
re-focuses) an independent Makie figure showing the 2D/1D power spectrum
of the currently visible slice.  The window reacts to `slice_proc` updates
and provides save-PNG / save-PDF / save-CSV buttons.

Captured state is passed explicitly as keyword arguments so this file
can be compiled independently of `_view_cube`'s scope.
"""
function _cube_ps_window_bundle(;
    ps_fig_ref,
    ps_alive_ref,
    slice_proc,
    wcs,
    axis,
    slice_axis_dims,
    save_root,
    fname,
    fname_box,
    make_name,
    ui_text,
    ui_text_muted,
    ui_accent,
    ui_error,
    style_menu!,
    style_button!,
    style_button_primary!,
    style_button_ghost!,
    style_checkbox!,
    style_textbox!,
    set_status!,
    cm_obs,
)
    function open_power_spectrum_window!()
        # Re-focus existing window instead of opening a second one.
        if ps_alive_ref[] && ps_fig_ref[] !== nothing
            try
                display(ps_fig_ref[])
                return ps_fig_ref[]
            catch
                ps_alive_ref[] = false
                ps_fig_ref[]   = nothing
            end
        end

        ps_mode   = Observable(:two_d)   # :two_d | :one_d
        ps_src    = Observable(:zoom)    # :zoom  | :full
        ps_window = Observable(:hann)    # :hann | :hamming | :none
        ps_pad    = Observable(false)
        ps_nanapo = Observable(false)
        ps_units  = Observable(:pixel)   # :pixel | :physical
        ps_fit_on = Observable(false)

        fig_ps = Figure(size = (1000, 760))
        header = fig_ps[1, 1] = GridLayout(; alignmode = Outside(8), tellheight = false)
        colgap!(header, 12)
        rowgap!(header, 6)

        settings_bar = header[1, 1] = GridLayout(; alignmode = Outside(7), tellheight = false)
        Box(settings_bar[1, 1]; color = current_ui_theme().surface, strokecolor = current_ui_theme().border,
            strokewidth = 1.0, cornerradius = 8, z = -5)
        settings = settings_bar[1, 1] = GridLayout(; alignmode = Inside(), tellheight = false)
        colgap!(settings, 8)
        rowgap!(settings, 6)

        # Settings bar: mode/source/window/units on top, processing toggles below.
        Label(settings[1, 1]; text = "Mode",   halign = :right, fontsize = 13, color = ui_text_muted)
        mode_menu    = Menu(settings[1, 2]; options = ["2D", "1D"],                prompt = "2D",   width = 80)
        Label(settings[1, 3]; text = "Source", halign = :right, fontsize = 13, color = ui_text_muted)
        src_menu     = Menu(settings[1, 4]; options = ["zoom", "full"],            prompt = "zoom", width = 80)
        Label(settings[1, 5]; text = "Window", halign = :right, fontsize = 13, color = ui_text_muted)
        win_menu     = Menu(settings[1, 6]; options = ["Hann", "Hamming", "None"], prompt = "Hann", width = 96)
        Label(settings[1, 7]; text = "Units",  halign = :right, fontsize = 13, color = ui_text_muted)
        unit_menu    = Menu(settings[1, 8]; options = ["pixel", "physical"],       prompt = "pixel", width = 96)

        pad_chk    = Checkbox(settings[2, 1]); Label(settings[2, 2]; text = "Pad pow2",  halign = :left, fontsize = 13, color = ui_text)
        nanapo_chk = Checkbox(settings[2, 3]); Label(settings[2, 4]; text = "NaN apod.", halign = :left, fontsize = 13, color = ui_text)
        fit_chk    = Checkbox(settings[2, 5]); Label(settings[2, 6]; text = "$(MANTA_ICONS.fit) Fit", halign = :left, fontsize = 13, color = ui_text)
        Label(settings[2, 7]; text = "k range", halign = :right, fontsize = 13, color = ui_text_muted)
        kmin_box   = Textbox(settings[2, 8]; placeholder = "k_min", width = 84, height = 28)
        kmax_box   = Textbox(settings[2, 9]; placeholder = "k_max", width = 84, height = 28)

        action_bar = header[1, 2] = GridLayout(; halign = :right, valign = :top, tellwidth = false, tellheight = false)
        colgap!(action_bar, 6)
        rowgap!(action_bar, 6)
        refresh_btn  = Button(action_bar[1, 1:3]; label = "Refresh", width = 88, height = 30)
        save_png_btn = Button(action_bar[2, 1]; label = "$(MANTA_ICONS.export_icon) PNG", width = 96, height = 28)
        save_pdf_btn = Button(action_bar[2, 2]; label = "$(MANTA_ICONS.export_icon) PDF", width = 96, height = 28)
        save_csv_btn = Button(action_bar[2, 3]; label = "$(MANTA_ICONS.export_icon) CSV", width = 96, height = 28)

        ps_status = Observable(" ")
        Label(header[2, 1:2]; text = ps_status, halign = :left, fontsize = 12,
              color = ui_text_muted, tellwidth = false)

        style_menu!(mode_menu); style_menu!(src_menu); style_menu!(win_menu); style_menu!(unit_menu)
        style_button_ghost!(refresh_btn)
        style_button_primary!(save_png_btn); style_button_primary!(save_pdf_btn); style_button_primary!(save_csv_btn)
        style_checkbox!(pad_chk); style_checkbox!(nanapo_chk); style_checkbox!(fit_chk)
        style_textbox!(kmin_box); style_textbox!(kmax_box)

        plot_grid = fig_ps[2, 1] = GridLayout()
        colgap!(plot_grid, -8)
        rowsize!(fig_ps.layout, 1, Fixed(120))

        # WCS-derived physical pixel scale (for 1D + 2D physical-unit axes).
        u_dim_now()          = slice_axis_dims(axis[])[1]
        v_dim_now()          = slice_axis_dims(axis[])[2]
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

        # Cache of the latest 1D points, used by Save CSV and slope-fit.
        last_1d_k     = Float32[]
        last_1d_p     = Float32[]
        last_1d_units = "cycles/pixel"
        last_meta     = (; ny_in = 0, nx_in = 0, ny_eff = 0, nx_eff = 0,
                          padded = false, window = :none, apodized = false,
                          f_sky = 1.0, w_norm = 0.0, k_phys = false, src = "zoom")

        ps_blocks = Any[]
        clear_plot!() = begin
            for b in ps_blocks
                try; Makie.delete!(b); catch; end
            end
            empty!(ps_blocks)
        end

        function ps_subimage()
            M = slice_proc[]
            if ps_src[] === :full
                return M
            else
                # Crop to the current ax_img viewport if available;
                # fall back to the full slice.
                try
                    lims = current_axis_limits(fig_ps)
                    lims === nothing && return M
                    x0, y0 = lims[1], lims[2]
                    x1, y1 = lims[3], lims[4]
                    i_lo = clamp(Int(floor(min(x0, x1))), 1, size(M, 1))
                    i_hi = clamp(Int(ceil(max(x0, x1))),  1, size(M, 1))
                    j_lo = clamp(Int(floor(min(y0, y1))), 1, size(M, 2))
                    j_hi = clamp(Int(ceil(max(y0, y1))),  1, size(M, 2))
                    (i_hi <= i_lo || j_hi <= j_lo) && return M
                    return M[i_lo:i_hi, j_lo:j_hi]
                catch
                    return M
                end
            end
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
            use_phys  = ps_units[] === :physical && physical_available()
            dx, dy    = pixel_scales()
            k_unit_lbl = use_phys ? "1/$(physical_unit_label())" : "cycles/pixel"

            bundle = _cube_ps_bundle(sub;
                                     window      = ps_window[],
                                     pad_pow2    = ps_pad[],
                                     apodize_nan = ps_nanapo[],
                                     use_phys    = use_phys,
                                     dx          = dx,
                                     dy          = dy)
            P2d  = bundle.P2d
            ny   = bundle.meta.ny_eff
            nx   = bundle.meta.nx_eff
            meta = (; bundle.meta..., k_phys = use_phys, src = src_label)
            last_meta = meta

            if ps_mode[] === :two_d
                empty!(last_1d_k); empty!(last_1d_p)
                last_1d_units = k_unit_lbl
                vis = bundle.P2d_log10
                ax  = Axis(
                    plot_grid[1, 1];
                    title  = latexstring("\\text{2D power spectrum (log10) — ", latex_safe(src_label), "}"),
                    xlabel = use_phys ?
                        latexstring("k_x\\;(", latex_safe(k_unit_lbl), ")") :
                        L"k_x\;\text{(cycles/pixel)}",
                    ylabel = use_phys ?
                        latexstring("k_y\\;(", latex_safe(k_unit_lbl), ")") :
                        L"k_y\;\text{(cycles/pixel)}",
                    aspect      = DataAspect(),
                    xtickformat = latex_tick_formatter,
                    ytickformat = latex_tick_formatter,
                )
                kx = bundle.kx
                ky = bundle.ky
                hm = heatmap!(ax, kx, ky, vis; colormap = cm_obs[])
                cb = Colorbar(
                    plot_grid[1, 2],
                    hm;
                    label      = L"\log_{10}|F|^2",
                    width      = 18,
                    height     = _axis_render_height(ax),
                    tellheight = false,
                    valign     = :center,
                )
                push!(ps_blocks, ax); push!(ps_blocks, cb)
            else
                prof     = bundle.prof
                k        = bundle.k
                p_floored = bundle.prof_floored
                resize!(last_1d_k, length(k));    copyto!(last_1d_k, k)
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
                    lines!(ax, k, p_floored; color = ui_accent, linewidth = PS_LINE_LW)
                end
                push!(ps_blocks, ax)

                if ps_fit_on[] && length(k) >= 3
                    kmin_txt  = get_box_str(kmin_box)
                    kmax_txt  = get_box_str(kmax_box)
                    valid_k   = filter(>(0), k)
                    auto_lo   = isempty(valid_k) ? 0.0 : Float64(first(valid_k))
                    auto_hi   = isempty(k) ? Inf  : Float64(last(k))
                    kmin_v    = isempty(kmin_txt) ? auto_lo : something(tryparse(Float64, kmin_txt), auto_lo)
                    kmax_v    = isempty(kmax_txt) ? auto_hi : something(tryparse(Float64, kmax_txt), auto_hi)
                    slope, intercept, n_used = fit_loglog_slope(k, prof; kmin = kmin_v, kmax = kmax_v)
                    if isfinite(slope) && n_used >= 2
                        kfit = filter(ki -> ki > 0 && ki >= kmin_v && ki <= kmax_v, k)
                        if !isempty(kfit)
                            yfit = Float32.(10 .^ (slope .* log10.(Float64.(kfit)) .+ intercept))
                            lines!(ax, kfit, yfit; color = ui_error, linestyle = :dash, linewidth = 1.5)
                            slope_str = "slope=$(round(slope; digits = 3)) [n=$(n_used)]"
                            ps_status[] = format_status(meta) * " • " * slope_str
                            return
                        end
                    end
                end
            end
            ps_status[] = format_status(meta)
        end

        # ------------------------------------------------------------------
        # Widget callbacks
        # ------------------------------------------------------------------
        on(mode_menu.selection)  do sel; sel === nothing || (ps_mode[] = sel == "1D" ? :one_d : :two_d; ps_render!()); end
        on(src_menu.selection)   do sel; sel === nothing || (ps_src[] = sel == "full" ? :full : :zoom; ps_render!()); end
        on(win_menu.selection) do sel
            sel === nothing && return
            ps_window[] = sel == "Hamming" ? :hamming : sel == "None" ? :none : :hann
            ps_render!()
        end
        on(unit_menu.selection) do sel
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

        # Re-render when the main viewer's slice changes.
        ps_window_alive = Ref(true)
        on(slice_proc) do _
            ps_window_alive[] && ps_render!()
        end

        # ------------------------------------------------------------------
        # Save callbacks
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Finalise
        # ------------------------------------------------------------------
        ps_render!()
        ps_fig_ref[]   = fig_ps
        ps_alive_ref[] = true
        register_window_close!(fig_ps) do
            ps_window_alive[] = false
            ps_alive_ref[]    = false
            ps_fig_ref[]      = nothing
        end
        display(fig_ps)
        return fig_ps
    end  # open_power_spectrum_window!

    return (; open_power_spectrum_window!)
end
