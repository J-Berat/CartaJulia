# path: examples/open_hdf5.jl
#
# Open an HDF5 dataset in MANTA using the "file.h5:/group/dataset" syntax.
#
# Run from the repository root:
#   julia --project=. examples/open_hdf5.jl
#   julia --project=. examples/open_hdf5.jl path/to/file.h5:/group/dataset
#
# Headless smoke test:
#   MANTA_HEADLESS=1 julia --project=. examples/open_hdf5.jl
#
# What it shows:
#   - writing a 2D image into an HDF5 group with the metadata attributes MANTA
#     understands (units + a linear WCS via CTYPE/CRVAL/CRPIX/CDELT/CUNIT);
#   - addressing the internal dataset with the `path.h5:/group/dataset` spec;
#   - the same `lazy=true` flag the FITS loader exposes (on-demand hyperslabs).

using MANTA
using HDF5

const HEADLESS = get(ENV, "MANTA_HEADLESS", "0") == "1"

"A smooth 2D field so the contrast/colormap controls have something to chew on."
function synthetic_image(; nx::Int = 256, ny::Int = 256)
    img = Array{Float32}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        x = (i - nx / 2) / (nx / 6)
        y = (j - ny / 2) / (ny / 6)
        img[i, j] = Float32(exp(-(x^2 + y^2) / 2) + 0.1 * sin(0.2 * i) * cos(0.2 * j))
    end
    return img
end

"""
Write `img` to `path` under `/maps/intensity`, attaching the attributes MANTA
reads: a unit label and a linear (RA/Dec-like) WCS on both axes. Returns the
full MANTA address string `path:/maps/intensity`.
"""
function write_h5_image(path::AbstractString, img::AbstractMatrix)
    h5open(path, "w") do f
        g = create_group(f, "maps")
        g["intensity"] = Array{Float32}(img)   # write the dataset
        dset = g["intensity"]                  # re-open it to attach attributes
        # Unit label shown on the colorbar.
        attributes(dset)["units"] = "Jy/beam"
        # Linear WCS metadata, one CTYPE/CRVAL/CRPIX/CDELT/CUNIT per axis.
        attributes(dset)["CTYPE1"] = "RA---TAN"
        attributes(dset)["CRVAL1"] = 150.0
        attributes(dset)["CRPIX1"] = 128.0
        attributes(dset)["CDELT1"] = -0.001
        attributes(dset)["CUNIT1"] = "deg"
        attributes(dset)["CTYPE2"] = "DEC--TAN"
        attributes(dset)["CRVAL2"] = 2.0
        attributes(dset)["CRPIX2"] = 128.0
        attributes(dset)["CDELT2"] = 0.001
        attributes(dset)["CUNIT2"] = "deg"
    end
    return string(path, ":/maps/intensity")
end

# Use a spec from the command line, or build one next to this script.
spec = if !isempty(ARGS)
    ARGS[1]
else
    h5_path = joinpath(@__DIR__, "output", "example_image.h5")
    isdir(dirname(h5_path)) || mkpath(dirname(h5_path))
    s = write_h5_image(h5_path, synthetic_image())
    @info "Wrote HDF5 image" spec = s
    s
end

fig = MANTA.manta(
    spec;
    cmap        = :viridis,
    lazy        = true,          # on-demand reads; harmless for a small image
    activate_gl = !HEADLESS,
    display_fig = !HEADLESS,
)

if HEADLESS
    @info "Headless run OK — HDF5 image viewer built without a GL window."
else
    MANTA.wait_until_closed(fig)
end
