# path: examples/healpix_map.jl
#
# Display a HEALPix all-sky map in Mollweide projection.
#
# Run from the repository root:
#   julia --project=. examples/healpix_map.jl
#   julia --project=. examples/healpix_map.jl path/to/map.fits
#
# Headless smoke test:
#   MANTA_HEADLESS=1 julia --project=. examples/healpix_map.jl
#
# What it shows:
#   - building a synthetic Healpix.HealpixMap (dipole + a couple of hot spots);
#   - opening it directly with MANTA.manta_healpix (no file needed);
#   - the file path overload for a real HEALPix FITS map.

using MANTA
using Healpix

const HEADLESS = get(ENV, "MANTA_HEADLESS", "0") == "1"

"""
Synthetic ring-ordered HEALPix map at the given `nside`: a galactic-style
dipole plus two Gaussian hot spots. Returns a `Healpix.HealpixMap`.
"""
function synthetic_healpix_map(; nside::Int = 32)
    npix = 12 * nside^2                       # HEALPix invariant
    res = Healpix.Resolution(nside)
    spots = ((θ = π / 3, φ = π / 2), (θ = 2π / 3, φ = 4π / 3))  # (colat, lon)
    pixels = Vector{Float64}(undef, npix)
    for i in 1:npix
        θ, φ = Healpix.pix2angRing(res, i)   # colatitude, longitude (radians)
        dipole = cos(θ)                       # smooth large-scale gradient
        hot = 0.0
        for s in spots
            dθ = θ - s.θ
            dφ = φ - s.φ
            hot += 2.0 * exp(-(dθ^2 + dφ^2) / (2 * 0.12^2))
        end
        pixels[i] = dipole + hot
    end
    return Healpix.HealpixMap{Float64,Healpix.RingOrder,Vector{Float64}}(pixels)
end

fig = if !isempty(ARGS)
    # File path overload: MANTA validates that the FITS really holds a map.
    MANTA.manta_healpix(
        abspath(ARGS[1]);
        cmap        = :inferno,
        column      = 1,           # BinTable column to read
        activate_gl = !HEADLESS,
        display_fig = !HEADLESS,
    )
else
    m = synthetic_healpix_map()
    @info "Built synthetic HEALPix map" npix = length(m.pixels)
    MANTA.manta_healpix(
        m;
        cmap        = :inferno,
        scale       = :lin,
        activate_gl = !HEADLESS,
        display_fig = !HEADLESS,
    )
end

if HEADLESS
    @info "Headless run OK — Mollweide map built without a GL window."
else
    MANTA.wait_until_closed(fig)
end
