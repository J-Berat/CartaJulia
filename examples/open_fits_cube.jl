# path: examples/open_fits_cube.jl
#
# Open a 3D FITS cube in the interactive viewer.
#
# Run from the repository root:
#   julia --project=. examples/open_fits_cube.jl
#   julia --project=. examples/open_fits_cube.jl path/to/your_cube.fits
#
# Headless smoke test (no OpenGL window, builds the figure only):
#   MANTA_HEADLESS=1 julia --project=. examples/open_fits_cube.jl
#
# What it shows:
#   - building a synthetic FITS cube with a velocity (VRAD) WCS axis so MANTA
#     can label the spectrum axis on its own;
#   - opening it with MANTA.manta and a few common kwargs;
#   - keeping the process alive until the window is closed.

using MANTA
using FITSIO

# `MANTA_HEADLESS=1` exercises the same path the tests use: no GL context, no
# window. Otherwise we open a real interactive viewer.
const HEADLESS = get(ENV, "MANTA_HEADLESS", "0") == "1"

"Non-negative synthetic cube (kept ≥ 0 so :log10 / :ln scales never hit NaN)."
function synthetic_cube(; nx::Int = 80, ny::Int = 64, nz::Int = 40)
    data = Array{Float32}(undef, nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        # A drifting Gaussian blob across channels + a smooth ramp.
        x0 = nx * (0.3f0 + 0.4f0 * k / nz)
        blob = 800f0 * exp(-((i - x0)^2 + (j - ny / 2)^2) / (2f0 * 9f0^2))
        ramp = 0.5f0 * (i + 2j)
        data[i, j, k] = max(blob + ramp, 0f0)
    end
    return data
end

"Write `cube` to a FITS file, tagging axis 3 as a radio velocity axis (km/s)."
function write_cube_fits(path::AbstractString, cube::AbstractArray{<:Real,3})
    FITS(path, "w") do f
        hdr = FITSHeader(
            ["CTYPE3", "CRVAL3", "CRPIX3", "CDELT3", "CUNIT3", "BUNIT"],
            ["VRAD", -20.0, 1.0, 1.0, "km/s", "K"],
            ["", "", "", "", "", ""],
        )
        write(f, Array{Float32}(cube); header = hdr)
    end
    return path
end

# Use a file given on the command line, or generate one next to this script.
fits_path = if !isempty(ARGS)
    abspath(ARGS[1])
else
    p = joinpath(@__DIR__, "output", "example_cube.fits")
    isdir(dirname(p)) || mkpath(dirname(p))
    write_cube_fits(p, synthetic_cube())
    @info "Wrote synthetic cube" path = p
    p
end

fig = MANTA.manta(
    fits_path;
    cmap        = :magma,
    scale       = :log10,
    figsize     = (1600, 950),
    # Headless toggles: leave them at their defaults for an interactive window.
    activate_gl = !HEADLESS,
    display_fig = !HEADLESS,
)

if HEADLESS
    @info "Headless run OK — cube viewer built without a GL window."
else
    # Block until the user closes the window (no-op in a live REPL).
    MANTA.wait_until_closed(fig)
end
