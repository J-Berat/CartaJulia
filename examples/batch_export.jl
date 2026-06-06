# path: examples/batch_export.jl
#
# Render a list of FITS files to images headlessly with MANTA.manta_batch.
# No OpenGL context or window is created (activate_gl=false, display_fig=false
# are forced internally), so this runs fine on a server or in CI.
#
# Run from the repository root:
#   julia --project=. examples/batch_export.jl
#   julia --project=. examples/batch_export.jl a.fits b.fits c.fits
#
# What it shows:
#   - manta_batch over several files;
#   - choosing the output format, directory, and a filename prefix;
#   - forwarding ordinary `manta` kwargs (cmap, vmin, vmax, figsize) to every
#     render.

using MANTA
using FITSIO

"A small synthetic 2D field, deterministic per `seed` so the panels differ."
function synthetic_image(seed::Int; nx::Int = 200, ny::Int = 160)
    img = Array{Float32}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        cx = nx * (0.3 + 0.1 * seed)
        img[i, j] = Float32(500 * exp(-((i - cx)^2 + (j - ny / 2)^2) / (2 * 25.0^2)) + i + j)
    end
    return img
end

function write_image_fits(path::AbstractString, img::AbstractMatrix)
    FITS(path, "w") do f
        write(f, Array{Float32}(img))
    end
    return path
end

# Use FITS paths from the command line, or synthesize three of them.
outdir = joinpath(@__DIR__, "output")
isdir(outdir) || mkpath(outdir)

paths = if !isempty(ARGS)
    abspath.(ARGS)
else
    map(1:3) do k
        p = joinpath(outdir, "survey_field_$(k).fits")
        write_image_fits(p, synthetic_image(k))
    end
end
@info "Batch input files" paths

written = MANTA.manta_batch(
    paths;
    format   = :png,                       # :png (default) or :pdf
    save_dir = joinpath(outdir, "exports"),
    prefix   = "render_",                  # filenames become render_<stem>.png
    cmap     = :magma,
    vmin     = 0,
    vmax     = 600,
    figsize  = (1200, 800),
)

@info "Batch export complete" count = length(written)
for p in written
    println("  → ", p)
end
