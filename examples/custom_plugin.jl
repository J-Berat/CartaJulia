# path: examples/custom_plugin.jl
#
# Extend MANTA without patching it, using the plugin API. Three surfaces exist:
#
#   :loader       (matcher, loader)            — teach MANTA a new file format
#   :dataset_view (DatasetType, name, fn)      — add a named view mode
#   :postprocess  (fig, ds, opts) -> nothing   — run after a view is built
#
# This example registers a loader for a toy ".xyz" text format and a
# postprocess hook, opens a file through the normal MANTA.manta entry point,
# then cleans up.
#
# Run from the repository root:
#   julia --project=. examples/custom_plugin.jl
#
# Headless smoke test:
#   MANTA_HEADLESS=1 julia --project=. examples/custom_plugin.jl

using MANTA

const HEADLESS = get(ENV, "MANTA_HEADLESS", "0") == "1"

# --- A toy on-disk format: ".xyz" -------------------------------------------
# Line 1: "nx ny"
# Rest  : nx*ny whitespace-separated Float32 values (column-major).
function write_xyz(path::AbstractString, img::AbstractMatrix)
    nx, ny = size(img)
    open(path, "w") do io
        println(io, nx, " ", ny)
        for v in img
            println(io, Float32(v))
        end
    end
    return path
end

"Loader: parse a .xyz file into an ImageDataset (an AbstractMANTADataset)."
function load_xyz(path::AbstractString; kwargs...)
    lines = readlines(path)
    nx, ny = parse.(Int, split(strip(lines[1])))
    vals = parse.(Float32, @view lines[2:(1 + nx * ny)])
    img = reshape(vals, nx, ny)
    return MANTA.ImageDataset(
        img;
        axis_labels = ["x", "y"],
        unit_label  = "counts",
        source_id   = MANTA.stable_source_id(abspath(path)),
        metadata    = Dict{Symbol,Any}(:format => "xyz"),
    )
end

# --- Register the extensions -------------------------------------------------
# A loader is a (matcher, loader) pair. register_plugin! returns the plugin so
# it can be removed later by identity.
xyz_loader = MANTA.register_plugin!(:loader, (
    path -> endswith(lowercase(path), ".xyz"),
    load_xyz,
))

# A postprocess hook is a `(fig, ds, opts) -> nothing` callback. It's the place
# to draw overlays or register custom shortcuts. MANTA dispatches the whole
# chain through `MANTA.run_postprocess!(fig, ds; opts...)`, which we call
# explicitly further down so the effect is visible.
xyz_post = MANTA.register_plugin!(:postprocess, function (fig, ds, opts)
    @info "postprocess fired" dataset = typeof(ds) source = ds.source_id opts = opts
    return nothing
end)

@info "Registered plugins" loaders = length(MANTA.list_plugins(:loader)) postprocess = length(MANTA.list_plugins(:postprocess))

# --- Use it through the normal entry point ----------------------------------
xyz_path = joinpath(@__DIR__, "output", "example.xyz")
isdir(dirname(xyz_path)) || mkpath(dirname(xyz_path))
write_xyz(xyz_path, Float32[i + 2j for i in 1:120, j in 1:90])

# MANTA.manta gives registered loaders first crack at the path, so our .xyz
# file is handled by load_xyz and rendered with the standard image viewer.
fig = MANTA.manta(
    xyz_path;
    cmap        = :cividis,
    activate_gl = !HEADLESS,
    display_fig = !HEADLESS,
)

# Run the registered :postprocess chain against this view. (Wire this into your
# own view builders to have it fire automatically.)
ds = MANTA.load_dataset(xyz_path)
MANTA.run_postprocess!(fig, ds)

if HEADLESS
    @info "Headless run OK — .xyz file loaded via the custom plugin."
else
    MANTA.wait_until_closed(fig)
end

# --- Clean up ----------------------------------------------------------------
# Remove just our plugins (by identity)…
MANTA.unregister_plugin!(:loader, xyz_loader)
MANTA.unregister_plugin!(:postprocess, xyz_post)
# …or drop everything at once with MANTA.clear_plugins!().
@info "After cleanup" loaders = length(MANTA.list_plugins(:loader)) postprocess = length(MANTA.list_plugins(:postprocess))
