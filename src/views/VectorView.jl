# path: src/views/VectorView.jl
#
# 1D viewer for `VectorDataset` (and any `AbstractVector{<:Real}` routed through
# `load_dataset`). Body: one main `lines` plot, a stats card, a controls card
# (x/y scale menus, selection min/max textboxes, export buttons) and a status
# bar. Selection narrows stats + CSV export to the chosen interval.
#
# Headless contract: same as the cube/HEALPix views. `activate_gl=false` skips
# GLMakie activation; `display_fig=false` skips `display(fig)` so the function
# can run under CI without an OpenGL context.

# --------------------------------------------------------------------------
# Small parsers — Float64 variants so we don't lose precision for WCS-mapped
# X values (e.g. spectral axes in Hz). Kept private; the public API mirrors
# `_parse_axis_limits` in `helpers/UIBits.jl` but returns Float64 tuples.
# --------------------------------------------------------------------------

function _parse_vector_range(
    min_txt::AbstractString,
    max_txt::AbstractString;
    fallback::Tuple{Float64,Float64} = (0.0, 1.0),
    axis_name::AbstractString = "selection",
)
    smin = strip(String(min_txt))
    smax = strip(String(max_txt))
    if isempty(smin) && isempty(smax)
        return (true, false, fallback, "$(axis_name) cleared.")
    end
    if isempty(smin) ⊻ isempty(smax)
        return (false, false, fallback,
                "Fill both $(axis_name) min and max, or clear both.")
    end
    a = tryparse(Float64, smin)
    b = tryparse(Float64, smax)
    if a === nothing || b === nothing
        return (false, false, fallback, "$(axis_name) bounds must be numbers.")
    end
    lo = Float64(a); hi = Float64(b)
    if !(isfinite(lo) && isfinite(hi))
        return (false, false, fallback, "$(axis_name) bounds must be finite.")
    end
    if lo > hi
        lo, hi = hi, lo
        return (true, true, (lo, hi),
                "$(axis_name) bounds were swapped because min > max.")
    end
    if lo == hi
        lo = prevfloat(lo); hi = nextfloat(hi)
        return (true, true, (lo, hi),
                "Expanded equal $(axis_name) bounds to avoid zero width.")
    end
    return (true, true, (lo, hi), "$(axis_name) applied.")
end

# --------------------------------------------------------------------------
# Selection / stats helpers (pure, no Makie state). Tested directly via the
# headless test set.
# --------------------------------------------------------------------------

"""
    _vector_selection_indices(xs, sel) -> UnitRange{Int}

Return the index range of `xs` that falls within `sel = (lo, hi)`. When `sel`
is `nothing`, returns the full range. The X axis is assumed to be monotonic;
when it is not, we fall back to a linear scan.
"""
function _vector_selection_indices(xs::AbstractVector{<:Real}, sel)
    n = length(xs)
    n == 0 && return 1:0
    sel === nothing && return 1:n
    lo, hi = Float64(first(sel)), Float64(last(sel))
    # Fast path: monotonic non-decreasing X (the common case for index/WCS).
    if issorted(xs)
        # `searchsortedfirst` returns `n+1` when all entries are < lo, and
        # `searchsortedlast` returns `0` when all entries are > hi. We detect
        # these out-of-range cases BEFORE clamping; otherwise clamping would
        # collapse the empty range into a 1-element range at the boundary.
        i1 = searchsortedfirst(xs, lo)
        i2 = searchsortedlast(xs, hi)
        (i1 > n || i2 < 1 || i1 > i2) && return 1:0
        return clamp(i1, 1, n):clamp(i2, 1, n)
    end
    # Slow path: pick the contiguous run of matching indices. We linearly scan
    # and return the first/last matching index. This is fine: vectors big
    # enough to matter here are still tiny vs. cubes.
    first_idx = findfirst(x -> lo <= Float64(x) <= hi, xs)
    last_idx  = findlast(x -> lo <= Float64(x) <= hi, xs)
    first_idx === nothing && return 1:0
    return first_idx:last_idx
end

"""
    _vector_stats(ys) -> NamedTuple

Compute n / n_finite / n_nan / min / max / mean / std / median over a vector
view. NaNs are filtered out for the finite-only stats. Returns `NaN` for
stats that need at least one (or two) finite samples when those preconditions
are not met.
"""
function _vector_stats(ys::AbstractVector{<:Real})
    n = length(ys)
    n_nan = 0
    fin = Float64[]
    sizehint!(fin, n)
    @inbounds for v in ys
        x = Float64(v)
        if isfinite(x)
            push!(fin, x)
        else
            n_nan += 1
        end
    end
    n_fin = length(fin)
    if n_fin == 0
        return (n = n, n_finite = 0, n_nan = n_nan,
                min = NaN, max = NaN, mean = NaN, std = NaN, median = NaN)
    end
    mn  = minimum(fin)
    mx  = maximum(fin)
    mu  = mean(fin)
    md  = median(fin)
    sd  = n_fin >= 2 ? std(fin) : NaN
    return (n = n, n_finite = n_fin, n_nan = n_nan,
            min = mn, max = mx, mean = mu, std = sd, median = md)
end

# --------------------------------------------------------------------------
# X axis construction from dataset (WCS-aware).
# --------------------------------------------------------------------------

"""
    _vector_x_axis(ds) -> (xs::Vector{Float64}, xlabel::LaTeXString)

Build the X axis from a `VectorDataset`:
  * if `ds.wcs` is a usable `SimpleWCSAxis`, X = world_coord(wcs, 1, k) for
    k = 1..N, and the label comes from `wcs_axis_label`,
  * otherwise X = 1..N and the label is the dataset's `axis_label`.
"""
function _vector_x_axis(ds::VectorDataset)
    n = length(ds.data)
    if ds.wcs !== nothing && ds.wcs.available
        # `world_coord` lives on a vector-of-axes; wrap in a length-1 vector.
        wvec = SimpleWCSAxis[ds.wcs]
        xs = Float64[world_coord(wvec, 1, k) for k in 1:n]
        return xs, wcs_axis_label(wvec, 1; fallback = ds.axis_label)
    end
    xs = Float64[Float64(k) for k in 1:n]
    return xs, latexstring("\\text{", latex_safe(ds.axis_label), "}")
end

# --------------------------------------------------------------------------
# CSV export (no extra dependency — mirrors the pattern used by CubeView.jl
# for the 1D power-spectrum CSV).
# --------------------------------------------------------------------------

"""
    _write_vector_csv(out, xs, ys, idx, meta) -> nothing

Write `xs[idx]` / `ys[idx]` to `out` as a 2-column CSV. The first lines are
comment metadata so the file is self-describing without external context.
"""
function _write_vector_csv(out::AbstractString,
                           xs::AbstractVector{<:Real},
                           ys::AbstractVector{<:Real},
                           idx::AbstractRange,
                           meta::NamedTuple)
    open(String(out), "w") do io
        println(io, "# source=", meta.source_id)
        println(io, "# axis_label=", meta.axis_label)
        println(io, "# unit_label=", meta.unit_label)
        println(io, "# selection_indices=", first(idx), ":", last(idx))
        println(io, "# n_selected=", length(idx))
        println(io, "x,y")
        @inbounds for k in idx
            println(io, xs[k], ",", ys[k])
        end
    end
    return nothing
end

# --------------------------------------------------------------------------
# Public view function.
# --------------------------------------------------------------------------

"""
    _view_vector(ds::VectorDataset; kwargs...) -> Makie.Figure

Interactive 1D viewer. Layout:
  * main `lines` plot (x vs y, with the current selection shaded),
  * stats card (`n`, finite, NaN, min, max, mean, std, median over selection),
  * controls card (xscale / yscale menus, selection from/to textboxes,
    Apply / Reset selection buttons, Save PNG / Save PDF / Save CSV buttons),
  * status bar.

Selection narrows both the stats card and the CSV export to the chosen
interval. PNG / PDF exports always capture the full figure.
"""
function _view_vector(
    ds::VectorDataset;
    title::Union{Nothing,AbstractString} = nothing,
    xscale::Symbol = :lin,
    yscale::Symbol = :lin,
    xlimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
)
    # ---- raw arrays ----
    ys_raw = Float64[Float64(v) for v in ds.data]
    xs_raw, xlabel_tex = _vector_x_axis(ds)
    n = length(ys_raw)
    n >= 1 || throw(ArgumentError(
        "MANTA: cannot view an empty VectorDataset (source=$(ds.source_id))."))

    unit_label_tex = latexstring("\\text{", latex_safe(ds.unit_label), "}")
    title_text     = title === nothing ? ds.source_id : String(title)

    # ---- state ----
    xscale_obs = Observable(xscale)
    yscale_obs = Observable(yscale)

    # Selection: nothing => use full range. Stored as (Float64, Float64) in
    # world units so the user input is independent of `xscale`.
    selection = Observable{Union{Nothing,Tuple{Float64,Float64}}}(nothing)

    # Display-ready X / Y, with scales applied (NaN for ≤ 0 in log/ln modes).
    xs_disp = lift(xscale_obs) do mode
        out = apply_scale(Float32.(xs_raw), mode)
        Float32[isfinite(x) ? Float32(x) : Float32(NaN32) for x in out]
    end
    ys_disp = lift(yscale_obs) do mode
        out = apply_scale(Float32.(ys_raw), mode)
        Float32[isfinite(y) ? Float32(y) : Float32(NaN32) for y in out]
    end

    # Selection indices (computed in *world* X coordinates → robust to scale).
    sel_idx_obs = lift(selection) do sel
        _vector_selection_indices(xs_raw, sel)
    end

    # Stats computed on the selected window of the raw (untransformed) Y.
    stats_obs = lift(sel_idx_obs) do idx
        if isempty(idx)
            _vector_stats(Float64[])
        else
            _vector_stats(@view ys_raw[idx])
        end
    end

    # Selection boundaries materialized as two vertical lines (left/right).
    # We avoid `band!` here because it would force an explicit Y range and
    # interfere with the axis autoscaling. `vlines!` is what the
    # `apply_btn`-driven clims indicator in `manta(::AbstractMatrix)` uses, so
    # the visual idiom matches the rest of the project.
    sel_visible = lift(s -> s !== nothing, selection)
    sel_boundary_xs = lift(selection, xscale_obs) do sel, mode
        if sel === nothing
            return Float32[NaN32, NaN32]
        end
        lo, hi = sel
        # Convert world bounds to display bounds (apply current xscale).
        a = apply_scale(Float32[Float64(lo)], mode)[1]
        b = apply_scale(Float32[Float64(hi)], mode)[1]
        Float32[isfinite(a) ? a : Float32(NaN32),
                isfinite(b) ? b : Float32(NaN32)]
    end

    # ---- theme + figure ----
    ui_theme       = current_ui_theme()
    fig_bg         = ui_theme.background
    ui_text        = ui_theme.text
    ui_text_muted  = ui_theme.text_muted
    ui_accent      = ui_theme.accent
    ui_accent_dim  = ui_theme.accent_dim
    ui_selection   = ui_theme.selection

    pick_backend!(activate_gl)
    fig = Figure(size = _pick_fig_size(figsize), backgroundcolor = fig_bg)
    grid = fig[1, 1] = GridLayout()
    colgap!(grid, 16); rowgap!(grid, 14)

    # ---- main plot ----
    plot_grid = grid[1, 1] = GridLayout()
    ax = Axis(
        plot_grid[1, 1];
        title = make_main_title(title_text),
        xlabel = xlabel_tex,
        ylabel = unit_label_tex,
        xtickformat = latex_tick_formatter,
        ytickformat = latex_tick_formatter,
    )

    lines!(ax, xs_disp, ys_disp; color = ui_accent, linewidth = 1.6)
    # Selection boundaries — drawn after the line so they sit on top.
    vlines!(ax, sel_boundary_xs;
            color = ui_selection, linewidth = 1.6, linestyle = :dash,
            visible = sel_visible)
    # Optional dots when N is small enough to make them readable.
    if n <= 256
        scatter!(ax, xs_disp, ys_disp;
                 color = ui_accent_dim, markersize = 5, strokewidth = 0)
    end

    # User-provided x/y limits override autolimits.
    if xlimits !== nothing
        xlims!(ax, Float32(first(xlimits)), Float32(last(xlimits)))
    end
    if ylimits !== nothing
        ylims!(ax, Float32(first(ylimits)), Float32(last(ylimits)))
    end

    # ---- stats card ----
    stats_label_obs = lift(stats_obs) do s
        function _fmt(x)
            (x isa AbstractFloat && isnan(x)) ? "NaN" :
                string(round(Float64(x); digits = 6))
        end
        latexstring(
            "\\mathrm{N}=\\mathbf{", s.n, "}",
            "\\quad\\mathrm{finite}=\\mathbf{", s.n_finite, "}",
            "\\quad\\mathrm{NaN}=\\mathbf{", s.n_nan, "}",
            "\\quad\\mathrm{min}=\\mathbf{", latex_safe(_fmt(s.min)), "}",
            "\\quad\\mathrm{max}=\\mathbf{", latex_safe(_fmt(s.max)), "}",
            "\\quad\\mathrm{mean}=\\mathbf{", latex_safe(_fmt(s.mean)), "}",
            "\\quad\\mathrm{std}=\\mathbf{", latex_safe(_fmt(s.std)), "}",
            "\\quad\\mathrm{median}=\\mathbf{", latex_safe(_fmt(s.median)), "}",
        )
    end
    Label(grid[2, 1]; text = stats_label_obs,
          halign = :left, tellwidth = false, fontsize = 13, color = ui_text)

    # ---- controls card ----
    ctrl = grid[3, 1] = GridLayout(; alignmode = Outside())
    Label(ctrl[1, 1], text = "X scale", halign = :left, tellwidth = false,
          fontsize = 14, color = ui_text_muted)
    xscale_menu = Menu(ctrl[1, 2]; options = ["lin", "log10", "ln"],
                      prompt = String(xscale_obs[]), width = 96)
    Label(ctrl[1, 3], text = "Y scale", halign = :left, tellwidth = false,
          fontsize = 14, color = ui_text_muted)
    yscale_menu = Menu(ctrl[1, 4]; options = ["lin", "log10", "ln"],
                      prompt = String(yscale_obs[]), width = 96)

    Label(ctrl[2, 1], text = "Selection", halign = :left, tellwidth = false,
          fontsize = 14, color = ui_text_muted)
    sel_min_box = Textbox(ctrl[2, 2]; placeholder = "x min", width = 110, height = 32)
    sel_max_box = Textbox(ctrl[2, 3]; placeholder = "x max", width = 110, height = 32)
    sel_apply_btn = Button(ctrl[2, 4]; label = "Apply", width = 82, height = 32)
    sel_reset_btn = Button(ctrl[2, 5]; label = "Reset", width = 82, height = 32)

    Label(ctrl[3, 1], text = "Export", halign = :left, tellwidth = false,
          fontsize = 14, color = ui_text_muted)
    save_png_btn = Button(ctrl[3, 2]; label = "$(MANTA_ICONS.export_icon) PNG", width = 90, height = 32)
    save_pdf_btn = Button(ctrl[3, 3]; label = "$(MANTA_ICONS.export_icon) PDF", width = 90, height = 32)
    save_csv_btn = Button(ctrl[3, 4]; label = "$(MANTA_ICONS.export_icon) CSV", width = 90, height = 32)

    # Style consistent avec les autres vues MANTA.
    foreach(w -> manta_style_button_primary!(w, ui_theme),
            (sel_apply_btn, save_png_btn, save_pdf_btn, save_csv_btn))
    manta_style_button_ghost!(sel_reset_btn, ui_theme)
    foreach(w -> manta_style_menu!(w, ui_theme),
            (xscale_menu, yscale_menu))
    foreach(w -> manta_style_textbox!(w, ui_theme),
            (sel_min_box, sel_max_box))

    # Initial menu selection so the prompt matches the actual state.
    xscale_menu.selection[] = String(xscale_obs[])
    yscale_menu.selection[] = String(yscale_obs[])

    # ---- status bar ----
    ui_status = Observable(" ")
    Label(grid[4, 1]; text = ui_status, halign = :left,
          tellwidth = false, fontsize = 12, color = ui_text_muted)
    set_status!(msg::AbstractString) = (ui_status[] = String(msg); nothing)

    # Small textbox helper (same pattern as in MANTA.jl::manta(::AbstractMatrix)).
    set_box_text!(tb, s::AbstractString) = begin
        str = String(s)
        tb.displayed_string[] = str
        tb.stored_string[]    = str
        nothing
    end

    # ---- callbacks ----
    on(xscale_menu.selection) do sel
        sel === nothing && return
        xscale_obs[] = Symbol(sel)
        set_status!("X scale set to $(sel).")
    end
    on(yscale_menu.selection) do sel
        sel === nothing && return
        yscale_obs[] = Symbol(sel)
        set_status!("Y scale set to $(sel).")
    end

    on(sel_apply_btn.clicks) do _
        ok, manual, bounds, msg = _parse_vector_range(
            get_box_str(sel_min_box),
            get_box_str(sel_max_box);
            fallback = (first(xs_raw), last(xs_raw)),
            axis_name = "selection",
        )
        set_status!(msg)
        ok || return
        if manual
            selection[] = bounds
            set_box_text!(sel_min_box, string(first(bounds)))
            set_box_text!(sel_max_box, string(last(bounds)))
        else
            selection[] = nothing
        end
    end
    on(sel_reset_btn.clicks) do _
        selection[] = nothing
        set_box_text!(sel_min_box, "")
        set_box_text!(sel_max_box, "")
        set_status!("Selection cleared.")
    end

    # ---- export plumbing ----
    save_root = if save_dir === nothing
        d = joinpath(homedir(), "Desktop")
        isdir(d) ? d : pwd()
    else
        path = String(save_dir)
        isdir(path) || mkpath(path)
        path
    end
    safe_stem = replace(String(ds.source_id), r"[^A-Za-z0-9._-]" => "_")
    safe_stem = isempty(safe_stem) ? "vector" : safe_stem

    on(save_png_btn.clicks) do _
        out = joinpath(save_root, "$(safe_stem)_vector.png")
        try
            CairoMakie.save(String(out), fig; backend = CairoMakie)
            set_status!("Saved PNG to $(out).")
        catch e
            msg = "Failed to save PNG: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
        end
    end
    on(save_pdf_btn.clicks) do _
        out = joinpath(save_root, "$(safe_stem)_vector.pdf")
        try
            CairoMakie.save(String(out), fig; backend = CairoMakie)
            set_status!("Saved PDF to $(out).")
        catch e
            msg = "Failed to save PDF: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
        end
    end
    on(save_csv_btn.clicks) do _
        idx = sel_idx_obs[]
        if isempty(idx)
            set_status!("Selection is empty: nothing to save.")
            return
        end
        out = joinpath(save_root, "$(safe_stem)_vector.csv")
        meta = (
            source_id  = ds.source_id,
            axis_label = ds.axis_label,
            unit_label = ds.unit_label,
        )
        try
            _write_vector_csv(out, xs_raw, ys_raw, idx, meta)
            set_status!("Saved CSV to $(out).")
        catch e
            msg = "Failed to save CSV: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
        end
    end

    register_window_close!(fig)  # anchor in GC root — see MANTA._KEEP_ALIVE block comment
    enable_file_drop!(fig; activate_gl = activate_gl, display_fig = display_fig)
    display_fig && display(fig)
    return fig
end
