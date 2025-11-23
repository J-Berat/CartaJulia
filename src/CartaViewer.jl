# path: src/CartaViewer.jl
module CartaViewer

using GLMakie
using CairoMakie
using Makie
using Observables
using ImageFiltering
using LaTeXStrings
using FITSIO
using GLFW

# ---- helpers (no TeX newlines; only inline LaTeX) ----
include("helpers/Helpers.jl")

export carta

"""
    carta(filepath::String;
          cmap::Symbol = :viridis,
          vmin = nothing, vmax = nothing,
          invert::Bool = false,
          fullscreen::Bool = false,
          size::Union{Nothing,Tuple{Int,Int}} = nothing)

Interactive 3D FITS viewer (slice + per-voxel spectrum).
- Manual color limits when `vmin` & `vmax` set (also sync spectrum Y).
- Window sized by `fullscreen=true` or explicit `size=(w,h)`.
"""
function carta(
    filepath::String;
    cmap::Symbol = :viridis,
    vmin = nothing,
    vmax = nothing,
    invert::Bool = false,
    fullscreen::Bool = false,
    size::Union{Nothing,Tuple{Int,Int}} = nothing
)
    # ---------- Load ----------
    cube = FITS(filepath) do f
        read(f[1])
    end
    @assert ndims(cube) == 3 "Not a 3D cube"

    data = Float32.(cube)
    siz  = size(data)  # (nx, ny, nz)

    fname_full = basename(filepath)
    fname = replace(fname_full, r"\.fits$" => ""; count = 1)

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

    slice_raw = lift(axis, idx) do a, id
        get_slice(data, a, clamp(id, 1, siz[a]))
    end

    gauss_on = Observable(false)
    sigma    = Observable(1.5f0)

    slice_proc = lift(slice_raw, gauss_on, sigma) do s, on, σ
        if on && σ > 0
            k = ImageFiltering.Kernel.gaussian((σ, σ))
            imfilter(Float32.(s), k)
        else
            Float32.(s)
        end
    end

    slice_disp = lift(slice_proc, img_scale_mode) do s, m
        apply_scale(s, m)
    end

    clims_auto = lift(slice_disp) do s
        clamped_extrema(vec(s))
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

    spec_x_raw  = Observable(collect(1:siz[3]))
    spec_y_raw  = Observable(zeros(Float32, siz[3]))
    spec_y_disp = lift(spec_y_raw, spec_scale_mode) do y, m
        apply_scale(y, m)
    end

    # ---------- Figure & layout ----------
    GLMakie.activate!()
    fig = Figure(size = _pick_fig_size(fullscreen, size))

    main_grid = fig[1, 1] = GridLayout()
    top_grid  = main_grid[1, 1] = GridLayout()

    # Image + colorbar
    img_grid  = top_grid[1, 1] = GridLayout()
    ax_img = Axis(
        img_grid[1, 1];
        title     = latexstring(fname),
        xlabel    = L"\text{pixel } x",
        ylabel    = L"\text{pixel } y",
        aspect    = DataAspect(),
        yreversed = true,
    )

    uv_point = Observable(Point2f(1, 1))
    hm = heatmap!(ax_img, slice_disp; colormap = cm_obs, colorrange = clims_obs)
    scatter!(ax_img, lift(p -> [p], uv_point); markersize = 10)

    Colorbar(img_grid[1, 2], hm; label = L"\text{intensity (scaled)}", width = 20)

    # Info + spectrum
    spec_grid = top_grid[1, 2] = GridLayout()
    lab_info = Label(
        spec_grid[1, 1];
        text     = make_info_tex(1, 1, 1, 1, 1, 0f0),
        halign   = :left,
        valign   = :top,
        fontsize = 14,
    )

    ax_spec = Axis(
        spec_grid[2, 1];
        title  = L"\text{Spectrum at selected pixel}",
        xlabel = L"\text{index along slice axis}",
        ylabel = L"\text{intensity (scaled)}",
        width  = 600,
        height = 400,
    )
    lines!(ax_spec, spec_x_raw, spec_y_disp)

    # Controls
    bottom_grid    = main_grid[2, 1] = GridLayout()
    img_ctrl_grid  = bottom_grid[1, 1] = GridLayout()
    spec_ctrl_grid = bottom_grid[1, 2] = GridLayout()

    # Image controls (row1)
    im_row1_left  = img_ctrl_grid[1, 1] = GridLayout()
    im_row1_right = img_ctrl_grid[1, 2] = GridLayout()

    Label(im_row1_left[1, 1], text = L"\text{Image scale}")
    img_scale_menu = Menu(im_row1_left[1, 2]; options = ["lin", "log10", "ln"], prompt = "lin", width = 60)

    Label(im_row1_left[1, 3], text = L"\text{Spectrum scale}")
    spec_scale_menu = Menu(im_row1_left[1, 4]; options = ["lin", "log10", "ln"], prompt = "lin", width = 60)

    invert_chk = Checkbox(im_row1_left[1, 5])
    Label(im_row1_left[1, 6], text = L"\text{Invert colormap}")

    Label(im_row1_right[1, 1], text = L"\text{Colorbar limits}")
    clim_min_box   = Textbox(im_row1_right[1, 2], placeholder = "min")
    clim_max_box   = Textbox(im_row1_right[1, 3], placeholder = "max")
    clim_apply_btn = Button(im_row1_right[1, 4], label = "Apply")

    if use_manual[]
        mn, mx = clims_manual[]
        s_mn, s_mx = string(mn), string(mx)
        clim_min_box.displayed_string[] = s_mn; clim_min_box.stored_string[] = s_mn
        clim_max_box.displayed_string[] = s_mx; clim_max_box.stored_string[] = s_mx
    end

    # Image controls (row2)
    im_row2_left  = img_ctrl_grid[2, 1] = GridLayout()
    im_row2_right = img_ctrl_grid[2, 2] = GridLayout()

    Label(im_row2_left[1, 1], text = L"\text{Save}")
    fmt_menu  = Menu(im_row2_left[1, 2]; options = ["png", "pdf"], prompt = "png", width = 70)
    fname_box = Textbox(im_row2_left[1, 3], placeholder = "filename base")

    btn_save_fig  = Button(im_row2_right[1, 1], label = "Save fig")
    btn_save_both = Button(im_row2_right[1, 2], label = "Save slice+spec")

    # Image controls (row3)
    im_row3_left  = img_ctrl_grid[3, 1] = GridLayout()
    im_row3_right = img_ctrl_grid[3, 2] = GridLayout()

    Label(im_row3_left[1, 1], text = L"\text{GIF indices}")
    start_box = Textbox(im_row3_left[1, 2], placeholder = "start")
    stop_box  = Textbox(im_row3_left[1, 3], placeholder = "stop")
    step_box  = Textbox(im_row3_left[1, 4], placeholder = "step")
    fps_box   = Textbox(im_row3_left[1, 5], placeholder = "fps")

    pingpong_chk = Checkbox(im_row3_left[1, 6])
    Label(im_row3_left[1, 7], text = L"\text{Back-and-forth mode}")

    anim_btn = Button(im_row3_right[1, 1], label = "Export GIF")

    # Spectrum controls
    sp_row1_left  = spec_ctrl_grid[1, 1] = GridLayout()
    sp_row1_right = spec_ctrl_grid[1, 2] = GridLayout()

    Label(sp_row1_left[1, 1], text = L"\text{Slice axis}")
    axes_labels = ["dim1 (x)", "dim2 (y)", "dim3 (z)"]
    axis_menu = Menu(sp_row1_left[1, 2]; options = axes_labels, prompt = "dim3 (z)", width = 90)

    status_label = Label(sp_row1_left[1, 3], text = latexstring("\\text{axis } 3,\\, \\text{index } 1"), fontsize = 12)

    Label(sp_row1_right[1, 1], text = L"\text{Index}")
    slice_slider = Slider(sp_row1_right[1, 2]; range = 1:siz[3], startvalue = 1, width = 200, height = 10)

    sp_row2_left  = spec_ctrl_grid[2, 1] = GridLayout()
    sp_row2_right = spec_ctrl_grid[2, 2] = GridLayout()

    Label(sp_row2_left[1, 1], text = L"\text{Gaussian filter}")
    gauss_chk   = Checkbox(sp_row2_left[1, 2])
    sigma_label = Label(sp_row2_left[1, 3], text = latexstring("\\sigma = 1.5\\,\\text{px}"), fontsize = 12)

    sigma_slider = Slider(sp_row2_right[1, 1]; range = LinRange(0, 10, 101), startvalue = 1.5, width = 200, height = 10)

    # ---------- Helpers ----------
    function refresh_uv!()
        a = axis[]
        u, v = ijk_to_uv(i_idx[], j_idx[], k_idx[], a)
        u = clamp(u, 1, size(slice_raw[], 1))
        v = clamp(v, 1, size(slice_raw[], 2))
        u_idx[] = u; v_idx[] = v
        uv_point[] = Point2f(v, u)
    end

    function refresh_labels!()
        val = data[i_idx[], j_idx[], k_idx[]]
        lab_info.text = make_info_tex(i_idx[], j_idx[], k_idx[], u_idx[], v_idx[], val)
        status_label.text = latexstring("\\text{axis } $(axis[]),\\, \\text{index } $(idx[])")
    end

    function refresh_spectrum!()
        if axis[] == 1
            spec_x_raw[] = collect(1:siz[1]); spec_y_raw[] = data[:, j_idx[], k_idx[]]
        elseif axis[] == 2
            spec_x_raw[] = collect(1:siz[2]); spec_y_raw[] = data[i_idx[], :, k_idx[]]
        else
            spec_x_raw[] = collect(1:siz[3]); spec_y_raw[] = data[i_idx[], j_idx[], :]
        end
        if use_manual[]
            vmin_, vmax_ = clims_manual[]; limits!(ax_spec, nothing, nothing, vmin_, vmax_)
        else
            autolimits!(ax_spec)
        end
    end

    refresh_all!() = (refresh_uv!(); refresh_labels!(); refresh_spectrum!())

    # ---------- Reactivity ----------
    on(clims_obs) do (cmin, cmax)
        if use_manual[]
            limits!(ax_spec, nothing, nothing, cmin, cmax)
        end
    end

    on(spec_scale_mode) do _
        if use_manual[]
            vmin_, vmax_ = clims_manual[]; limits!(ax_spec, nothing, nothing, vmin_, vmax_)
        else
            autolimits!(ax_spec)
        end
    end

    # ---------- UI callbacks ----------
    on(axis_menu.selection) do sel
        sel === nothing && return
        new_axis = findfirst(==(sel), axes_labels)
        new_axis === nothing && return
        axis[] = new_axis
        slice_slider.range[] = 1:siz[new_axis]
        idx[] = min(idx[], siz[new_axis])
        ii, jj, kk = uv_to_ijk(u_idx[], v_idx[], axis[], idx[])
        i_idx[] = clamp(ii, 1, siz[1]); j_idx[] = clamp(jj, 1, siz[2]); k_idx[] = clamp(kk, 1, siz[3])
        refresh_all!()
    end

    on(slice_slider.value) do v
        idx[] = Int(round(v))
        ii, jj, kk = uv_to_ijk(u_idx[], v_idx[], axis[], idx[])
        i_idx[] = clamp(ii, 1, siz[1]); j_idx[] = clamp(jj, 1, siz[2]); k_idx[] = clamp(kk, 1, siz[3])
        refresh_labels!(); refresh_spectrum!()
    end

    on(img_scale_menu.selection) do sel
        sel === nothing && return
        img_scale_mode[] = Symbol(sel)
    end

    on(spec_scale_menu.selection) do sel
        sel === nothing && return
        spec_scale_mode[] = Symbol(sel)
    end

    on(invert_chk.checked) do v
        invert_cmap[] = v
    end

    on(gauss_chk.checked) do v
        gauss_on[] = v
        refresh_spectrum!()
    end

    on(sigma_slider.value) do v
        sigma[] = Float32(v)
        sigma_label.text = latexstring("\\sigma = $(round(v; digits = 2))\\,\\text{px}")
    end

    on(clim_apply_btn.clicks) do _
        txtmin = get_box_str(clim_min_box)
        txtmax = get_box_str(clim_max_box)
        if isempty(txtmin) || isempty(txtmax)
            use_manual[] = false
            autolimits!(ax_spec)
        else
            vmin_p = tryparse(Float32, txtmin); vmax_p = tryparse(Float32, txtmax)
            if vmin_p === nothing || vmax_p === nothing
                @warn "Could not parse colorbar limits from '$txtmin' '$txtmax'"
            else
                if vmin_p == vmax_p
                    vmin_p = prevfloat(vmin_p); vmax_p = nextfloat(vmax_p)
                end
                clims_manual[] = (vmin_p, vmax_p)
                use_manual[]   = true
                limits!(ax_spec, nothing, nothing, vmin_p, vmax_p)
            end
        end
    end

    # Keyboard navigation (+ invert)
    on(events(fig).keyboardbutton) do ev
        ev.action == Keyboard.press || return
        if ev.key == Keyboard.i
            invert_cmap[] = !invert_cmap[]
        elseif ev.key == Keyboard.left
            v_idx[] = max(1, v_idx[] - 1)
        elseif ev.key == Keyboard.right
            v_idx[] = min(size(slice_raw[], 2), v_idx[] + 1)
        elseif ev.key == Keyboard.up
            u_idx[] = max(1, u_idx[] - 1)
        elseif ev.key == Keyboard.down
            u_idx[] = min(size(slice_raw[], 1), u_idx[] + 1)
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
        if ev.button == Mouse.left && ev.action == Mouse.press
            mp = events(ax_img).mouseposition[]
            p  = to_world(ax_img.scene, mp)
            u = Int(round(clamp(p[2], 1, size(slice_raw[], 1))))
            v = Int(round(clamp(p[1], 1, size(slice_raw[], 2))))
            u_idx[] = u; v_idx[] = v
            ii, jj, kk = uv_to_ijk(u, v, axis[], idx[])
            i_idx[] = clamp(ii, 1, siz[1]); j_idx[] = clamp(jj, 1, siz[2]); k_idx[] = clamp(kk, 1, siz[3])
            refresh_labels!(); refresh_spectrum!()
            uv_point[] = Point2f(v, u)
        end
    end

    # ---------- Saving ----------
    save_dir = isdir(joinpath(homedir(), "Desktop")) ? joinpath(homedir(), "Desktop") : pwd()

    make_name = function (base::String, ext::String)
        b = isempty(base) ? fname : base
        return "$(b)_axis$(axis[])_idx$(idx[])_i$(i_idx[])_j$(j_idx[])_k$(k_idx[])_img$(String(img_scale_mode[]))_spec$(String(spec_scale_mode[])).$(ext)"
    end

    save_with_format = function (path::String, obj)
        if endswith(lowercase(path), ".pdf")
            CairoMakie.save(path, obj)  # vector when PDF
        else
            save(path, obj)
        end
    end

    on(btn_save_fig.clicks) do _
        ext  = something(fmt_menu.selection[], "png")
        base = get_box_str(fname_box)
        outfile = joinpath(save_dir, make_name(base, ext))
        try
            save_with_format(outfile, fig)
            @info "Saved figure: $outfile"
        catch e
            @error "Failed to save $outfile: $e"
        end
    end

    on(btn_save_both.clicks) do _
        ext  = something(fmt_menu.selection[], "png")
        base = get_box_str(fname_box)

        # Slice figure
        try
            f_slice = Figure(size = (700, 560))
            axS = Axis(
                f_slice[1, 1];
                title     = make_slice_title(fname, axis[], idx[]),
                xlabel    = L"\text{pixel } x",
                ylabel    = L"\text{pixel } y",
                aspect    = DataAspect(),
                yreversed = true,
            )
            hmS = heatmap!(axS, slice_disp[]; colormap = cm_obs[], colorrange = clims_obs[])
            scatter!(axS, [Point2f(uv_point[]...)], markersize = 10)
            Colorbar(f_slice[1, 2], hmS; label = L"\text{intensity (scaled)}", width = 20)

            out_slice = joinpath(save_dir, make_name(base * "_slice", ext))
            save_with_format(out_slice, f_slice)
            @info "Saved slice: $out_slice"
        catch e
            @error "Failed to save slice: $e"
        end

        # Spectrum figure
        try
            f_spec = Figure(size = (600, 400))
            axP = Axis(
                f_spec[1, 1];
                title  = make_spec_title(i_idx[], j_idx[], k_idx[]),
                xlabel = L"\text{index along slice axis}",
                ylabel = L"\text{intensity (scaled)}",
            )
            lines!(axP, spec_x_raw[], spec_y_disp[])

            out_spec = joinpath(save_dir, make_name(base * "_spectrum", ext))
            save_with_format(out_spec, f_spec)
            @info "Saved spectrum: $out_spec"
        catch e
            @error "Failed to save spectrum: $e"
        end
    end

    # ---------- GIF export ----------
    on(anim_btn.clicks) do _
        a = axis[]; amax = siz[a]

        start = let v = get_box_str(start_box); isempty(v) ? 1 : clamp(something(tryparse(Int, v), 1), 1, amax) end
        stop  = let v = get_box_str(stop_box);  isempty(v) ? amax : clamp(something(tryparse(Int, v), amax), 1, amax) end
        step  = let v = get_box_str(step_box);  isempty(v) ? 1 : max(1, something(tryparse(Int, v), 1)) end
        fps   = let v = get_box_str(fps_box);   isempty(v) ? 12 : max(1, something(tryparse(Int, v), 12)) end

        frames = collect(start:step:stop)
        if pingpong_chk.checked[] && length(frames) ≥ 2
            frames = vcat(frames, reverse(frames[2:end-1]))
        end

        base = get_box_str(fname_box)
        base = isempty(base) ? fname : base
        outfile = joinpath(save_dir, "$(base)_axis$(a)_$(first(frames))to$(last(frames))_fps$(fps)_$(String(img_scale_mode[])).gif")

        try
            record(fig, outfile, frames; framerate = fps) do fidx
                idx[] = fidx
                ii, jj, kk = uv_to_ijk(u_idx[], v_idx[], axis[], idx[])
                i_idx[] = clamp(ii, 1, siz[1]); j_idx[] = clamp(jj, 1, siz[2]); k_idx[] = clamp(kk, 1, siz[3])
                refresh_labels!(); refresh_spectrum!()
            end
            @info "Animation saved: $outfile"
        catch e
            @error "Failed to export animation $outfile: $e"
        end
    end

    # ---------- Init ----------
    refresh_all!()
    display(fig)
    return fig
end

end # module