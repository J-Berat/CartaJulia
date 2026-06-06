# path: src/helpers/PowerSpectrum.jl
#
# 2D / radial power spectrum primitives reused by the cube viewer's embedded
# and pop-out PS layouts: window functions (`_ps_window1d`,
# `_ps_apodize_mask`), the 2D spectrum (`power_spectrum_2d`), the radial
# profile (`power_spectrum_1d_radial`) and a log-log slope fit
# (`fit_loglog_slope`). Extracted from helpers/Helpers.jl.


############################
# Power spectrum
############################

"""
    _ps_window1d(kind, n) -> Vector{Float64}

1D apodization window of length `n`. `kind ∈ (:hann, :hamming, :none)`.
"""
function _ps_window1d(kind::Symbol, n::Integer)
    n <= 1 && return ones(Float64, max(n, 0))
    if kind === :hann
        return Float64[0.5 - 0.5 * cos(2π * (i - 1) / (n - 1)) for i in 1:n]
    elseif kind === :hamming
        return Float64[0.54 - 0.46 * cos(2π * (i - 1) / (n - 1)) for i in 1:n]
    else
        return ones(Float64, n)
    end
end

"""
    _ps_apodize_mask(mask, taper) -> Matrix{Float64}

Cosine taper of a binary validity mask. The taper width `taper` (in pixels)
determines how far inside the valid region the apodization runs to zero at
the boundary with invalid pixels. Uses an L∞ (Chebyshev) two-pass distance
transform — only the relative magnitude matters for the cosine taper, so the
cheap chamfer is sufficient.
"""
function _ps_apodize_mask(mask::AbstractMatrix{Bool}, taper::Integer)
    ny, nx = size(mask)
    big = float(ny + nx + 1)
    d = fill(big, ny, nx)
    @inbounds for i in eachindex(mask)
        if !mask[i]
            d[i] = 0.0
        end
    end
    @inbounds for j in 1:nx, i in 1:ny
        di = d[i, j]
        if i > 1
            di = min(di, d[i - 1, j] + 1.0)
            if j > 1;  di = min(di, d[i - 1, j - 1] + 1.0); end
            if j < nx; di = min(di, d[i - 1, j + 1] + 1.0); end
        end
        if j > 1; di = min(di, d[i, j - 1] + 1.0); end
        d[i, j] = di
    end
    @inbounds for j in nx:-1:1, i in ny:-1:1
        di = d[i, j]
        if i < ny
            di = min(di, d[i + 1, j] + 1.0)
            if j < nx; di = min(di, d[i + 1, j + 1] + 1.0); end
            if j > 1;  di = min(di, d[i + 1, j - 1] + 1.0); end
        end
        if j < nx; di = min(di, d[i, j + 1] + 1.0); end
        d[i, j] = di
    end
    out = Matrix{Float64}(undef, ny, nx)
    t = float(max(taper, 1))
    @inbounds for i in eachindex(mask)
        if !mask[i]
            out[i] = 0.0
        else
            di = d[i]
            out[i] = di >= t ? 1.0 : 0.5 * (1.0 - cos(π * di / t))
        end
    end
    return out
end

"""
    power_spectrum_2d(img;
                      window=:hann, demean=true,
                      pad_pow2=false,
                      apodize_nan=false, nan_taper=4)
        -> NamedTuple

Centered (`fftshift`) 2D power spectrum `|F(img)|² / ⟨W_eff²⟩`.

NamedTuple fields:
  - `P2d::Matrix{Float64}`           shifted power spectrum, size `(ny_eff, nx_eff)`
  - `ny_in, nx_in::Int`              size of the input image
  - `ny_eff, nx_eff::Int`            size after optional zero-padding
  - `padded::Bool`                   `true` iff `pad_pow2` actually grew the array
  - `window::Symbol`                 effective window kind
  - `apodized::Bool`                 `true` iff a NaN apodization mask was applied
  - `f_sky::Float64`                 fraction of finite input pixels
  - `w_norm::Float64`                `⟨(window × mask)²⟩`, the MASTER-light
                                     normalization that has already been
                                     divided out of `P2d`

NaN handling: non-finite pixels are first replaced with zero; the demean
operates over valid pixels only; with `apodize_nan=true` an L∞-distance
cosine taper is built around the NaN regions and combined with the spectral
window. The MASTER-light correction divides the raw `|F|²` by `⟨W_eff²⟩` so
that, for a stationary signal whose spectrum is locally flat over the window
support, the recovered amplitude is unbiased to first order.
"""
function power_spectrum_2d(img::AbstractMatrix;
                            window::Symbol = :hann,
                            demean::Bool = true,
                            pad_pow2::Bool = false,
                            apodize_nan::Bool = false,
                            nan_taper::Integer = 4)
    A = Float64.(img)
    ny0, nx0 = size(A)
    finite_mask = isfinite.(A)
    n_valid = count(finite_mask)
    f_sky = isempty(finite_mask) ? 0.0 : n_valid / length(finite_mask)
    @inbounds for i in eachindex(A)
        if !finite_mask[i]
            A[i] = 0.0
        end
    end
    if demean && n_valid > 0
        s = 0.0
        @inbounds for i in eachindex(A)
            if finite_mask[i]; s += A[i]; end
        end
        m = s / n_valid
        if isfinite(m) && m != 0.0
            @inbounds for i in eachindex(A)
                if finite_mask[i]; A[i] -= m; end
            end
        end
    end
    has_invalid = n_valid < length(finite_mask)
    M = if apodize_nan && has_invalid
        _ps_apodize_mask(finite_mask, max(Int(nan_taper), 1))
    elseif has_invalid
        Float64.(finite_mask)
    else
        ones(Float64, ny0, nx0)
    end
    wy = _ps_window1d(window, ny0)
    wx = _ps_window1d(window, nx0)
    Weff = Matrix{Float64}(undef, ny0, nx0)
    @inbounds for j in 1:nx0, i in 1:ny0
        Weff[i, j] = wy[i] * wx[j] * M[i, j]
        A[i, j] *= Weff[i, j]
    end
    ny_eff, nx_eff = ny0, nx0
    padded = false
    if pad_pow2
        ny_eff = nextpow(2, max(ny0, 1))
        nx_eff = nextpow(2, max(nx0, 1))
        if ny_eff != ny0 || nx_eff != nx0
            A_pad = zeros(Float64, ny_eff, nx_eff)
            @inbounds A_pad[1:ny0, 1:nx0] .= A
            A = A_pad
            padded = true
        end
    end
    F = fftshift(fft(A))
    P2d = abs2.(F)
    w_norm = isempty(Weff) ? 0.0 : mean(Weff .^ 2)
    if w_norm > 0
        P2d ./= w_norm
    end
    return (P2d = P2d,
            ny_in = ny0, nx_in = nx0,
            ny_eff = ny_eff, nx_eff = nx_eff,
            padded = padded,
            window = window,
            apodized = apodize_nan && has_invalid,
            f_sky = f_sky,
            w_norm = w_norm)
end

"""
    power_spectrum_1d_radial(P2d) -> (radii::Vector{Float32}, profile::Vector{Float32})

Radial average of a centered 2D power spectrum, binned by integer pixel
radius from the FFT-shifted DC center. The returned `radii` are pixel-radii
(0, 1, 2, …); convert to cycles/pixel by dividing by `min(ny, nx)`.
"""
function power_spectrum_1d_radial(P2d::AbstractMatrix)
    ny, nx = size(P2d)
    cy = (ny + 1) / 2
    cx = (nx + 1) / 2
    rmax = floor(Int, min(cy - 1, cx - 1))
    rmax < 1 && return (Float32[], Float32[])
    nb = rmax + 1
    counts = zeros(Float64, nb)
    sums = zeros(Float64, nb)
    @inbounds for j in 1:nx, i in 1:ny
        r = sqrt((i - cy)^2 + (j - cx)^2)
        b = round(Int, r) + 1
        if 1 <= b <= nb
            sums[b] += P2d[i, j]
            counts[b] += 1
        end
    end
    radii = Float32.(0:rmax)
    prof = Float32[counts[i] > 0 ? sums[i] / counts[i] : 0.0 for i in 1:nb]
    return radii, prof
end

"""
    fit_loglog_slope(k, p; kmin, kmax) -> (slope, intercept, n_used)

Least-squares fit of `log10(p) = slope·log10(k) + intercept` over the band
`kmin ≤ k ≤ kmax`. Non-positive `k` and `p` values are dropped. Returns
`(NaN, NaN, 0)` if fewer than 2 valid points fall in the band.
"""
function fit_loglog_slope(k::AbstractVector, p::AbstractVector;
                          kmin::Real = 0.0, kmax::Real = Inf)
    length(k) == length(p) || throw(ArgumentError("k and p must have the same length."))
    xs = Float64[]
    ys = Float64[]
    for i in eachindex(k)
        ki = Float64(k[i]); pi = Float64(p[i])
        if isfinite(ki) && isfinite(pi) && ki > 0 && pi > 0 && ki >= kmin && ki <= kmax
            push!(xs, log10(ki)); push!(ys, log10(pi))
        end
    end
    n = length(xs)
    n < 2 && return (NaN, NaN, n)
    mx = mean(xs); my = mean(ys)
    sxx = 0.0; sxy = 0.0
    @inbounds for i in 1:n
        dx = xs[i] - mx
        sxx += dx * dx
        sxy += dx * (ys[i] - my)
    end
    sxx == 0 && return (NaN, NaN, n)
    slope = sxy / sxx
    intercept = my - slope * mx
    return (slope, intercept, n)
end

"""
    healpix_power_spectrum_map(pixels; demean=true) -> (map, nside, n_valid, f_sky, fill_value)

Build a Float64 RING HEALPix map suitable for spherical-harmonic analysis.
Finite pixels different from `Healpix.UNSEEN` are kept; invalid pixels are
filled with zero after the optional finite-pixel mean subtraction.
"""
function healpix_power_spectrum_map(pixels::AbstractVector{<:Real};
                                    demean::Bool = true)
    npix = length(pixels)
    nside = valid_healpix_npix(npix)
    nside > 0 || throw(ArgumentError("HEALPix power spectrum: npix=$(npix) is not 12*nside^2."))

    vals = Vector{Float64}(undef, npix)
    valid = falses(npix)
    s = 0.0
    n_valid = 0
    unseen = Float64(Healpix.UNSEEN)
    @inbounds for i in 1:npix
        v = Float64(pixels[i])
        ok = isfinite(v) && v != unseen
        valid[i] = ok
        if ok
            vals[i] = v
            s += v
            n_valid += 1
        else
            vals[i] = 0.0
        end
    end
    n_valid > 0 || throw(ArgumentError("HEALPix power spectrum: no finite pixels to analyse."))

    fill_value = demean ? s / n_valid : 0.0
    if demean
        @inbounds for i in 1:npix
            vals[i] = valid[i] ? vals[i] - fill_value : 0.0
        end
    else
        @inbounds for i in 1:npix
            valid[i] || (vals[i] = fill_value)
        end
    end

    map = Healpix.HealpixMap{Float64,Healpix.RingOrder,Vector{Float64}}(vals)
    return (map = map,
            nside = nside,
            n_valid = n_valid,
            f_sky = n_valid / npix,
            fill_value = fill_value)
end

"""
    healpix_power_spectrum(pixels; lmax=nothing, mmax=nothing, niter=3,
                           demean=true) -> NamedTuple

Compute the spherical-harmonic HEALPix power spectrum `C_l` with
`Healpix.anafast`. This is the full-sky `a_lm` route, not a 2D FFT of a
Mollweide projection.
"""
function healpix_power_spectrum(pixels::AbstractVector{<:Real};
                                lmax = nothing,
                                mmax = nothing,
                                niter::Integer = 3,
                                demean::Bool = true)
    prep = healpix_power_spectrum_map(pixels; demean = demean)
    cl = Healpix.anafast(prep.map; lmax = lmax, mmax = mmax, niter = niter)
    ell = collect(0:(length(cl) - 1))
    dl = Healpix.cl2dl(cl, 0)
    return (ell = ell,
            cl = cl,
            dl = dl,
            nside = prep.nside,
            lmax = isempty(ell) ? -1 : last(ell),
            n_valid = prep.n_valid,
            f_sky = prep.f_sky,
            demean = demean,
            fill_value = prep.fill_value,
            niter = Int(niter))
end

function _healpix_ps_positive_points(ell, y)
    keep = Int[]
    @inbounds for i in eachindex(ell, y)
        yi = Float64(y[i])
        if ell[i] > 0 && isfinite(yi) && yi > 0
            push!(keep, i)
        end
    end
    return Float64[Float64(ell[i]) for i in keep],
           Float64[Float64(y[i]) for i in keep]
end

function _healpix_ps_nonnegative_points(ell, y)
    keep = Int[]
    @inbounds for i in eachindex(ell, y)
        yi = Float64(y[i])
        if ell[i] > 0 && isfinite(yi)
            push!(keep, i)
        end
    end
    isempty(keep) && return Float64[0.0], Float64[0.0]
    return Float64[Float64(ell[i]) for i in keep],
           Float64[max(Float64(y[i]), 0.0) for i in keep]
end

function _write_healpix_power_spectrum_csv(out::AbstractString, ps, meta::NamedTuple)
    open(String(out), "w") do io
        println(io, "# source=", meta.source_id)
        println(io, "# label=", meta.label)
        println(io, "# nside=", ps.nside)
        println(io, "# lmax=", ps.lmax)
        println(io, "# n_valid=", ps.n_valid)
        println(io, "# f_sky=", ps.f_sky)
        println(io, "# demean=", ps.demean)
        println(io, "# niter=", ps.niter)
        println(io, "# method=Healpix.anafast spherical harmonics")
        println(io, "ell,C_l,D_l")
        @inbounds for i in eachindex(ps.ell, ps.cl, ps.dl)
            println(io, ps.ell[i], ",", ps.cl[i], ",", ps.dl[i])
        end
    end
    return nothing
end

"""
    open_healpix_power_spectrum_window!(pixels; title, source_id, save_root, status!)

Open a pop-out HEALPix power-spectrum viewer using the spherical-harmonic
`C_l` from `Healpix.anafast`.
"""
function open_healpix_power_spectrum_window!(
    pixels::AbstractVector{<:Real};
    title::AbstractString,
    source_id::AbstractString,
    save_root::AbstractString,
    label::AbstractString = title,
    status!::Function = (_ -> nothing),
)
    theme = current_ui_theme()
    status!("Computing HEALPix C_l with spherical harmonics...")
    ps = healpix_power_spectrum(pixels)
    ell_cl, cl_plot = _healpix_ps_positive_points(ps.ell, ps.cl)
    ell_dl, dl_plot = _healpix_ps_positive_points(ps.ell, ps.dl)
    cl_log_scale = !isempty(ell_cl)
    dl_log_scale = !isempty(ell_dl)
    cl_log_scale || ((ell_cl, cl_plot) = _healpix_ps_nonnegative_points(ps.ell, ps.cl))
    dl_log_scale || ((ell_dl, dl_plot) = _healpix_ps_nonnegative_points(ps.ell, ps.dl))
    cl_x_log_scale = all(>(0), ell_cl)
    dl_x_log_scale = all(>(0), ell_dl)

    fig_ps = Figure(size = (980, 700); backgroundcolor = theme.background)
    Label(fig_ps[1, 1:2];
        text = "HEALPix power spectrum",
        halign = :left,
        tellwidth = false,
        fontsize = 20,
        color = theme.text,
        padding = (0, 0, 8, 4))

    ax_cl = Axis(fig_ps[2, 1];
        title = latexstring("\\text{", latex_safe(title), " } C_l"),
        xlabel = L"\ell",
        ylabel = L"C_\ell",
        xscale = cl_x_log_scale ? log10 : identity,
        yscale = cl_log_scale ? log10 : identity,
        backgroundcolor = theme.panel,
        xgridcolor = (theme.border, 0.35),
        ygridcolor = (theme.border, 0.35),
        xtickformat = latex_tick_formatter,
        ytickformat = latex_tick_formatter,
        titlecolor = theme.text,
        xlabelcolor = theme.text,
        ylabelcolor = theme.text,
        xticklabelcolor = theme.text_muted,
        yticklabelcolor = theme.text_muted)
    lines!(ax_cl, ell_cl, cl_plot; color = theme.accent, linewidth = 2.2)
    scatter!(ax_cl, ell_cl, cl_plot; color = theme.accent, markersize = 4)
    guard_log_zoom!(ax_cl)   # safe scroll-zoom when the axis scale is log10

    ax_dl = Axis(fig_ps[2, 2];
        title = latexstring("\\text{", latex_safe(title), " } D_l"),
        xlabel = L"\ell",
        ylabel = L"D_\ell = \ell(\ell+1)C_\ell/(2\pi)",
        xscale = dl_x_log_scale ? log10 : identity,
        yscale = dl_log_scale ? log10 : identity,
        backgroundcolor = theme.panel,
        xgridcolor = (theme.border, 0.35),
        ygridcolor = (theme.border, 0.35),
        xtickformat = latex_tick_formatter,
        ytickformat = latex_tick_formatter,
        titlecolor = theme.text,
        xlabelcolor = theme.text,
        ylabelcolor = theme.text,
        xticklabelcolor = theme.text_muted,
        yticklabelcolor = theme.text_muted)
    lines!(ax_dl, ell_dl, dl_plot; color = theme.selection, linewidth = 2.2)
    scatter!(ax_dl, ell_dl, dl_plot; color = theme.selection, markersize = 4)
    guard_log_zoom!(ax_dl)   # safe scroll-zoom when the axis scale is log10

    controls = fig_ps[3, 1:2] = GridLayout(; alignmode = Outside(10, 10, 8, 8))
    colgap!(controls, 8)
    Box(controls[1:2, 1:7];
        color = theme.surface,
        strokecolor = theme.border,
        strokewidth = 1.0,
        cornerradius = 8,
        z = -5)
    save_png_btn = Button(controls[1, 1]; label = "$(MANTA_ICONS.export_icon) PNG", width = 94, height = 30)
    save_pdf_btn = Button(controls[1, 2]; label = "$(MANTA_ICONS.export_icon) PDF", width = 94, height = 30)
    save_csv_btn = Button(controls[1, 3]; label = "$(MANTA_ICONS.export_icon) CSV", width = 94, height = 30)
    meta_label = Label(controls[1, 4:7];
        text = "nside=$(ps.nside)  lmax=$(ps.lmax)  f_sky=$(round(ps.f_sky; digits=4))  method=anafast",
        halign = :left,
        tellwidth = false,
        fontsize = 13,
        color = theme.text_muted)
    status_label = Label(controls[2, 1:7];
        text = cl_log_scale ?
            "C_l from spherical harmonics; monopole removed before analysis." :
            "C_l from spherical harmonics; no positive power after monopole removal.",
        halign = :left,
        tellwidth = false,
        fontsize = 13,
        color = theme.text_muted)
    foreach(c -> colsize!(controls, c, Auto()), 1:7)
    foreach(btn -> manta_style_button_primary!(btn, theme), (save_png_btn, save_pdf_btn, save_csv_btn))
    rowsize!(controls, 1, Fixed(36))
    rowsize!(controls, 2, Fixed(30))
    colgap!(fig_ps.layout, 12)

    safe_base = replace(strip(source_id), r"[^A-Za-z0-9_.-]+" => "_")
    isempty(safe_base) && (safe_base = "healpix")
    ps_path(ext) = joinpath(save_root, "$(safe_base)_healpix_Cl.$(ext)")
    set_ps_status!(msg) = (status_label.text[] = msg; status!(msg); nothing)

    on(save_png_btn.clicks) do _
        try
            out = ps_path("png")
            CairoMakie.save(String(out), fig_ps; backend = CairoMakie)
            set_ps_status!("Saved HEALPix power spectrum to $(out).")
        catch e
            set_ps_status!("Failed to save PNG: $(sprint(showerror, e))")
        end
    end
    on(save_pdf_btn.clicks) do _
        try
            out = ps_path("pdf")
            CairoMakie.save(String(out), fig_ps; backend = CairoMakie)
            set_ps_status!("Saved HEALPix power spectrum to $(out).")
        catch e
            set_ps_status!("Failed to save PDF: $(sprint(showerror, e))")
        end
    end
    on(save_csv_btn.clicks) do _
        try
            out = ps_path("csv")
            _write_healpix_power_spectrum_csv(out, ps, (; source_id, label))
            set_ps_status!("Saved HEALPix power spectrum CSV to $(out).")
        catch e
            set_ps_status!("Failed to save CSV: $(sprint(showerror, e))")
        end
    end

    register_window_close!(fig_ps)
    display(fig_ps)
    status!("Opened HEALPix power spectrum C_l window.")
    return fig_ps
end
