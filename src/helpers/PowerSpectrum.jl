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
