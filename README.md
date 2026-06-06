# MANTA

MANTA is an interactive astronomical data viewer written in Julia.
It is built on top of Makie/GLMakie and is designed for quick visual exploration
of maps, cubes, spectra, and HEALPix data with no boilerplate required.

The single entry point is `MANTA.manta(...)`. Pass it a file path or a Julia
array; MANTA dispatches automatically to the right viewer.

## What MANTA Can Open

- 2D FITS images;
- 3D FITS cubes, with slice navigation and a per-voxel or per-region spectrum;
- HEALPix FITS maps, displayed in Mollweide projection;
- HEALPix-PPV cubes (`npix x nchannels`), with one map per channel and a spectrum panel;
- HDF5 datasets, using the `"file.h5:/group/dataset"` syntax when an internal path is needed;
- in-memory Julia arrays: 1D vectors, 2D images, 3D cubes, RGB/RGBA data;
- `NamedTuple` or `Dict` collections of arrays for side-by-side multi-panel displays;
- `Healpix.HealpixMap` objects directly.

## Installation

Requirements:

- Julia 1.9 to 1.12;
- a graphical session with OpenGL support for GLMakie;
- `git`, if you clone the repository.

```bash
git clone https://github.com/J-Berat/MANTA.jl.git
cd MANTA.jl
julia --project=. scripts/setup.jl
```

The setup script installs the Julia dependencies listed in `Project.toml` and
precompiles the project environment, including the precompile workload that
reduces time-to-first-plot on subsequent launches.

## Quick Start

Run the built-in demo, which generates a synthetic FITS cube and opens the
viewer:

```bash
./manta
```

Open a specific FITS file:

```bash
./manta path/to/cube.fits
```

Control demo dimensions and contrast limits at startup:

```bash
NX=96 NY=72 NZ=48 VMIN=0 VMAX=1500 ./manta
```

## Julia Usage

Always launch Julia from the repository root with `--project=.` so the local
environment is active:

```bash
julia --project=.
```

Minimal example:

```julia
using MANTA

fig = MANTA.manta("path/to/cube.fits")
display(fig)
```

When running outside a REPL, keep the process alive until the user closes the
window:

```julia
fig = MANTA.manta("path/to/cube.fits")
display(fig)
MANTA.wait_until_closed(fig)
```

With common options:

```julia
fig = MANTA.manta(
    "path/to/cube.fits";
    cmap     = :magma,
    vmin     = 0.0,
    vmax     = 500.0,
    scale    = :log10,
    figsize  = (1600, 950),
    save_dir = "outputs",
    settings_path = "viewer_settings.toml",
)
display(fig)
```

Display arrays directly:

```julia
using MANTA

MANTA.manta(rand(128, 128); cmap=:magma)
MANTA.manta(rand(64, 64, 32); cmap=:viridis, vmin=0.0, vmax=1.0)
MANTA.manta(sin.(range(0.0, 4pi; length=512)); title="sin(t)")
```

## Examples

The `examples/` folder holds small, self-contained scripts that each open one
kind of input through the public API. They generate their own synthetic data
when no file is given, so they run out of the box:

```bash
julia --project=. examples/open_fits_cube.jl     # 3D FITS cube (slice + spectrum)
julia --project=. examples/open_hdf5.jl          # HDF5 image via "file.h5:/group/dataset"
julia --project=. examples/healpix_map.jl        # HEALPix map in Mollweide projection
julia --project=. examples/batch_export.jl       # headless render of several FITS to PNG
julia --project=. examples/custom_plugin.jl      # register a loader for a custom format
```

Each script also accepts a path to your own file, e.g.
`julia --project=. examples/open_fits_cube.jl path/to/cube.fits`.

Every example honours `MANTA_HEADLESS=1`, which builds the figure with
`activate_gl=false, display_fig=false` (no OpenGL window) — handy for a quick
smoke test on a server or in CI:

```bash
MANTA_HEADLESS=1 julia --project=. examples/open_fits_cube.jl
```

## Documentation

Long-form documentation lives in `docs/`:

- [Documentation index](docs/index.md): map of the long-form docs.
- [User guide](docs/user-guide.md): viewer capabilities, in-memory data, batch export, lazy loading, masks, power spectra, dark mode, and keyboard shortcuts.
- [Reference](docs/reference.md): complete `manta` keyword options, plugin API, and HDF5 conventions.
- [Development](docs/development.md): test commands, Docker notes, troubleshooting, and repository layout.

Docker-specific usage is also documented in [DOCKER.md](DOCKER.md).
