# path: demo/run_demo.jl
# Run:
#   julia --project demo/run_demo.jl
#   NX=96 NY=72 NZ=48 FULLSCREEN=0 VMIN=5 VMAX=1500 julia --project demo/run_demo.jl 80 60 40

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

# Try to load deps; if any missing, run scripts/setup.jl once.
function _ensure_deps!()
    try
        @eval using GLMakie, CairoMakie, Makie, Observables, ImageFiltering, LaTeXStrings, FITSIO, GLFW
    catch
        @info "Installing dependencies via scripts/setup.jl"
        include(joinpath(@__DIR__, "..", "scripts", "setup.jl"))
        @eval using GLMakie, CairoMakie, Makie, Observables, ImageFiltering, LaTeXStrings, FITSIO, GLFW
    end
end
_ensure_deps!()

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using CartaViewer
using FITSIO

# Non-negative synthetic cube (avoids log-scale NaNs).
function create_synthetic_cube(; nx::Int=64, ny::Int=48, nz::Int=32)
    data = Array{Float32}(undef, nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        base = 0.8f0*(i + 8*j + 64*k)
        ripple = 40f0 * sin(2f0*π*i/nx) * cos(2f0*π*j/ny)
        data[i, j, k] = max(base + ripple, 0f0)  # why: keep ≥0 for log10/ln scales
    end
    data
end

function write_fits(path::AbstractString, data::AbstractArray{<:Real,3})
    FITS(path, "w") do f
        write(f, data)
    end
    path
end

# CLI/env overrides
function _parse_dims()
    if length(ARGS) == 3
        try
            return parse.(Int, ARGS)
        catch
        end
    end
    nx = parse(Int, get(ENV, "NX", "64"))
    ny = parse(Int, get(ENV, "NY", "48"))
    nz = parse(Int, get(ENV, "NZ", "32"))
    return (nx, ny, nz)
end

(nx, ny, nz) = _parse_dims()
fullscreen = get(ENV, "FULLSCREEN", "1") != "0"

vmin = let s = get(ENV, "VMIN", ""); isempty(s) ? nothing : parse(Float64, s) end
vmax = let s = get(ENV, "VMAX", ""); isempty(s) ? nothing : parse(Float64, s) end

# Prepare output FITS
outdir = joinpath(@__DIR__, "output"); isdir(outdir) || mkpath(outdir)
fits_path = joinpath(outdir, "synthetic_cube.fits")

cube = create_synthetic_cube(nx=nx, ny=ny, nz=nz)
write_fits(fits_path, cube)
@info "FITS ready" path=fits_path size=size(cube)

# Launch UI
fig = CartaViewer.carta(
    fits_path;
    cmap=:magma,
    vmin=vmin,
    vmax=vmax,
    invert=false,
    fullscreen=fullscreen,
)

display(fig)
