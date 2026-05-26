# path: src/views/cube/KeyboardBundle.jl
#
# Keyboard shortcuts and mouse-pick callbacks for the 3D cube viewer.
#
# Extracted from src/views/CubeView.jl during a structural refactor
# (no behaviour change). Contains:
#   - navigation helper functions (_move_uv!, set_slice!, etc.)
#   - the shortcut binding table and registration
#   - the help-window opener
#   - left/right mouse-button handlers (zoom-drag, region-drag, pixel pick)
#   - mouse-position handler (live zoom/region rubber-band update)
#
# Entry point: `_cube_keyboard_bundle!(fig, ax_img; ...)`.
# Registers all `on(…)` listeners as a side effect; returns nothing.

"""
    _cube_keyboard_bundle!(fig, ax_img; kwargs...)

Register keyboard shortcuts and mouse-pick event handlers on `fig` and
`ax_img`.  All handlers are mode-aware via the shared `bypass_mode_gate`
Ref and the `on_mode` wrapper that is passed as a keyword argument.

Returns `nothing` — the function is called purely for its side effects
(observer registrations).
"""
function _cube_keyboard_bundle!(fig, ax_img;
    # --- slice / pixel state ---
    axis,
    idx,
    siz,
    u_idx,
    v_idx,
    i_idx,
    j_idx,
    k_idx,
    uv_point,
    # --- pure-function local helpers ---
    slice_dims,
    uv_to_ijk,
    # --- observables ---
    invert_cmap,
    img_scale_mode,
    show_contours,
    compare_visible,
    layout_mode,
    zoom_drag_active,
    zoom_drag_start,
    zoom_drag_end,
    region_drag_active,
    region_start,
    region_end,
    region_uvs,
    selection_mode,
    anim_playing,
    # --- widgets needed by shortcuts ---
    slice_slider,
    axis_menu,
    axes_labels,
    img_scale_menu,
    contour_chk,
    clim_auto_btn,
    btn_save_img,
    help_btn,
    region_count_label,
    ax_cmp,
    # --- callbacks from earlier sections ---
    refresh_labels!,
    refresh_spectrum!,
    apply_percentile_clims!,
    render_power_spectrum_layout!,
    reset_zoom!,
    clear_region!,
    update_region_from_drag!,
    set_status!,
    # --- mode / re-entrancy guards ---
    bypass_mode_gate,
    syncing_slice_controls,
    # --- all textboxes for shortcut-focus guard ---
    textboxes,
    # --- theme for help window ---
    ui_theme,
)
    # ------------------------------------------------------------------
    # Pixel-position helpers
    # ------------------------------------------------------------------
    function _move_uv!(du::Int, dv::Int)
        u_max, v_max = slice_dims(axis[])
        u_idx[] = clamp(u_idx[] + du, 1, u_max)
        v_idx[] = clamp(v_idx[] + dv, 1, v_max)
        ii, jj, kk = uv_to_ijk(u_idx[], v_idx[], axis[], idx[])
        i_idx[] = clamp(ii, 1, siz[1])
        j_idx[] = clamp(jj, 1, siz[2])
        k_idx[] = clamp(kk, 1, siz[3])
        refresh_labels!(); refresh_spectrum!()
        uv_point[] = Point2f(v_idx[], u_idx[])
    end

    # ------------------------------------------------------------------
    # Jump to slice `n` along the current axis.
    # Keeps the slider thumb in sync via a re-entrancy guard.
    # ------------------------------------------------------------------
    function set_slice!(n::Integer)
        n_clamped = clamp(Int(n), 1, siz[axis[]])
        syncing_slice_controls[] = true
        try
            slice_slider.value[] = n_clamped
        finally
            syncing_slice_controls[] = false
        end
        idx[] = n_clamped
        ii, jj, kk = uv_to_ijk(u_idx[], v_idx[], axis[], idx[])
        i_idx[] = clamp(ii, 1, siz[1])
        j_idx[] = clamp(jj, 1, siz[2])
        k_idx[] = clamp(kk, 1, siz[3])
        refresh_labels!(); refresh_spectrum!()
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
        set_status!("Slice $(idx[]) / $(siz[axis[]]) along axis $(axis[]).")
    end

    # ------------------------------------------------------------------
    # Switch the active axis through the axis menu (respects mode gate).
    # ------------------------------------------------------------------
    function switch_axis_shortcut!(new_axis::Int)
        new_axis in 1:3 || return
        new_axis == axis[] && return
        bypass_mode_gate[] = true
        try
            axis_menu.selection[] = axes_labels[new_axis]
        finally
            bypass_mode_gate[] = false
        end
    end

    # Programmatically fire a button click while bypassing the mode gate.
    function _trigger_button!(btn)
        bypass_mode_gate[] = true
        try
            btn.clicks[] = btn.clicks[] + 1
        finally
            bypass_mode_gate[] = false
        end
    end

    function toggle_contours_shortcut!()
        new_val = !show_contours[]
        bypass_mode_gate[] = true
        try
            contour_chk.checked[] = new_val
        finally
            bypass_mode_gate[] = false
        end
    end

    function cycle_log_scale!()
        next = img_scale_mode[] === :lin   ? :log10 :
               img_scale_mode[] === :log10 ? :ln    : :lin
        bypass_mode_gate[] = true
        try
            img_scale_menu.selection[] = String(next)
        finally
            bypass_mode_gate[] = false
        end
        set_status!("Image scale: $(String(next)).")
    end

    # ------------------------------------------------------------------
    # Shortcut table
    # ------------------------------------------------------------------
    shortcut_bindings = ShortcutBinding[
        ShortcutBinding(Keyboard.i,         () -> (invert_cmap[] = !invert_cmap[]);
                        description = "invert cmap"),
        ShortcutBinding(Keyboard.left,      () -> _move_uv!(0, -1)),
        ShortcutBinding(Keyboard.right,     () -> _move_uv!(0,  1)),
        ShortcutBinding(Keyboard.up,        () -> _move_uv!( 1, 0);
                        description = "pixel ←/→/↑/↓"),
        ShortcutBinding(Keyboard.down,      () -> _move_uv!(-1, 0)),
        ShortcutBinding(Keyboard.page_up,   () -> set_slice!(idx[] - 1);
                        description = "prev slice"),
        ShortcutBinding(Keyboard.page_down, () -> set_slice!(idx[] + 1);
                        description = "next slice"),
        ShortcutBinding(Keyboard.home,      () -> set_slice!(1);
                        description = "first slice"),
        ShortcutBinding(Keyboard.x,         () -> switch_axis_shortcut!(1);
                        description = "axis X"),
        ShortcutBinding(Keyboard.y,         () -> switch_axis_shortcut!(2);
                        description = "axis Y"),
        ShortcutBinding(Keyboard.z,         () -> switch_axis_shortcut!(3);
                        description = "axis Z"),
        ShortcutBinding(Keyboard.tab,       () -> switch_axis_shortcut!(mod1(axis[] + 1, 3));
                        description = "cycle axis"),
        ShortcutBinding(Keyboard.a,         () -> _trigger_button!(clim_auto_btn);
                        description = "auto contrast"),
        ShortcutBinding(Keyboard._1,        () -> apply_percentile_clims!(1, 99);
                        description = "p1-p99"),
        ShortcutBinding(Keyboard._5,        () -> apply_percentile_clims!(5, 95);
                        description = "p5-p95"),
        ShortcutBinding(Keyboard.r,         () -> reset_zoom!();
                        description = "reset zoom"),
        ShortcutBinding(Keyboard.s,         () -> _trigger_button!(btn_save_img);
                        description = "save image"),
        ShortcutBinding(Keyboard.c,         () -> toggle_contours_shortcut!();
                        description = "contours"),
        ShortcutBinding(Keyboard.l,         () -> cycle_log_scale!();
                        description = "cycle scale"),
    ]

    # `?` (Shift + /) opens a dedicated figure listing every binding.
    function _open_help_cube!()
        try
            open_shortcut_help_window(shortcut_bindings;
                title = "MANTA — Cube view shortcuts", theme = ui_theme)
        catch e
            @warn "Could not open shortcut help window" exception = (e, catch_backtrace())
        end
        set_status!(shortcut_help_message(shortcut_bindings))
    end
    push!(shortcut_bindings,
          ShortcutBinding(Keyboard.slash,
                          _open_help_cube!;
                          description = "this help",
                          modifier    = :shift))
    on(help_btn.clicks) do _
        _open_help_cube!()
    end

    register_shortcuts!(fig, shortcut_bindings;
        textboxes  = textboxes,
        is_blocked = () -> zoom_drag_active[] || region_drag_active[] || anim_playing[],
    )

    # ------------------------------------------------------------------
    # Mouse-button handler (left click = pixel pick / region drag;
    #                       right drag = zoom box)
    # ------------------------------------------------------------------
    on(events(ax_img).mousebutton) do ev
        if ev.button == Mouse.right && ev.action == Mouse.press
            p = mouseposition(ax_img)
            any(isnan, p) && return
            zoom_drag_start[] = Point2f(p[1], p[2])
            zoom_drag_end[]   = Point2f(p[1], p[2])
            zoom_drag_active[] = true
            set_status!("Zoom box: right-drag and release to apply.")
            return

        elseif ev.button == Mouse.right && ev.action == Mouse.release
            zoom_drag_active[] || return
            p = mouseposition(ax_img)
            if !any(isnan, p)
                zoom_drag_end[] = Point2f(p[1], p[2])
            end
            p0 = zoom_drag_start[]
            p1 = zoom_drag_end[]
            zoom_drag_active[]  = false
            zoom_drag_start[] = Point2f(NaN32, NaN32)
            zoom_drag_end[]   = Point2f(NaN32, NaN32)
            if !(isfinite(p0[1]) && isfinite(p0[2]) && isfinite(p1[1]) && isfinite(p1[2]))
                return
            end
            x0, y0 = p0;  x1, y1 = p1
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
            any(isnan, p) && return
            u_max, v_max = slice_dims(axis[])
            p0 = Point2f(clamp(p[1], 1, v_max), clamp(p[2], 1, u_max))
            region_start[]      = p0
            region_end[]        = p0
            region_drag_active[] = true
            region_uvs[]        = Tuple{Int,Int}[]
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
            refresh_labels!(); refresh_spectrum!()
            return

        elseif ev.button == Mouse.left && ev.action == Mouse.press
            p = mouseposition(ax_img)
            any(isnan, p) && return
            clear_region!()
            u_max, v_max = slice_dims(axis[])
            u = Int(round(clamp(p[2], 1, u_max)))
            v = Int(round(clamp(p[1], 1, v_max)))
            u_idx[] = u; v_idx[] = v
            ii, jj, kk = uv_to_ijk(u, v, axis[], idx[])
            i_idx[] = clamp(ii, 1, siz[1])
            j_idx[] = clamp(jj, 1, siz[2])
            k_idx[] = clamp(kk, 1, siz[3])
            refresh_labels!(); refresh_spectrum!()
            uv_point[] = Point2f(v, u)
        end
    end

    # Live rubber-band update while dragging.
    on(events(ax_img).mouseposition) do p
        if zoom_drag_active[] && !any(isnan, p)
            zoom_drag_end[] = Point2f(p[1], p[2])
        elseif region_drag_active[] && !any(isnan, p)
            u_max, v_max = slice_dims(axis[])
            region_end[] = Point2f(clamp(p[1], 1, v_max), clamp(p[2], 1, u_max))
        end
    end

    return nothing
end
