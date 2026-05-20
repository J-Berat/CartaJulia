# path: src/views/cube/PowerSpectrumBundle.jl
#
# Pure helper for the 3D cube viewer: bundles the 2D power spectrum and its
# 1D radial profile (plus matching k-axes and metadata) for use by both the
# embedded and pop-out power-spectrum views in CubeView.jl.
#
# Extracted verbatim from src/views/CubeView.jl during a structural refactor
# (no behaviour change). Already top-level in the original file — no closure
# captures, so the move is purely organizational.

"""
    _cube_ps_bundle(sub; window, pad_pow2, apodize_nan, use_phys, dx, dy)
        -> NamedTuple

Compute the 2D and 1D radial power spectrum products needed by both the
embedded and pop-out power-spectrum views, plus their k-axes (cycles/pixel
or `1/physical_unit`). Returns:

| field          | meaning                                                  |
| -------------- | -------------------------------------------------------- |
| `P2d`          | raw 2D power spectrum from `power_spectrum_2d`           |
| `P2d_log10`    | `log10.(max.(P2d, floor))` — heatmap-ready               |
| `kx`, `ky`     | 1D coordinate vectors for the 2D heatmap                 |
| `radii`, `prof`| raw radial profile from `power_spectrum_1d_radial`       |
| `k`            | k-axis matching `prof` (cycles/pixel or physical)        |
| `prof_floored` | `prof` clamped above a small floor for log plotting      |
| `meta`         | structural metadata (size, window, padding, f_sky, …)    |

Keeping this in one place avoids drift between the two call sites that used
to inline this exact pipeline.
"""
function _cube_ps_bundle(sub::AbstractMatrix;
                         window::Symbol, pad_pow2::Bool, apodize_nan::Bool,
                         use_phys::Bool, dx::Real, dy::Real)
    res = power_spectrum_2d(sub; window = window, pad_pow2 = pad_pow2,
                            apodize_nan = apodize_nan)
    P2d = res.P2d
    ny, nx = res.ny_eff, res.nx_eff

    pmax_2d = maximum(P2d)
    floor_2d = max(eps(Float64), pmax_2d * 1e-12)
    P2d_log10 = log10.(max.(P2d, floor_2d))

    kx = collect(Float32, (-nx / 2):(nx / 2 - 1)) ./ Float32(nx)
    ky = collect(Float32, (-ny / 2):(ny / 2 - 1)) ./ Float32(ny)
    if use_phys
        kx ./= Float32(dx)
        ky ./= Float32(dy)
    end

    radii, prof = power_spectrum_1d_radial(P2d)
    k_cyc = radii ./ Float32(min(ny, nx))
    k = use_phys ? Float32.(k_cyc ./ Float32(sqrt(Float64(dx) * Float64(dy)))) : k_cyc

    pmax_1d = isempty(prof) ? 1.0f0 : maximum(prof)
    floor_1d = Float32(max(eps(Float32), pmax_1d * 1f-12))
    prof_floored = max.(prof, floor_1d)

    meta = (; ny_in = res.ny_in, nx_in = res.nx_in,
              ny_eff = ny, nx_eff = nx,
              padded = res.padded, window = res.window,
              apodized = res.apodized, f_sky = res.f_sky,
              w_norm = res.w_norm)

    return (; P2d, P2d_log10, kx, ky, radii, prof, k, prof_floored, meta)
end
