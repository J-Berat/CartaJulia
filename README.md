# MANTA

MANTA is an interactive astronomical data viewer written in Julia.
It is built with Makie/GLMakie and is meant for quick exploration of maps,
cubes, and HEALPix data.

The basic idea is simple: give a file or an array to `MANTA.manta(...)`, then
inspect it visually with controls for slices, spectra, contrast, contours, and
exports.

## What MANTA Can Open

- 2D FITS images;
- 3D FITS cubes, with slice navigation and a spectrum at the selected voxel;
- HEALPix FITS maps, displayed in Mollweide projection;
- HEALPix-PPV cubes, with one map per channel and an associated spectrum;
- HDF5 datasets, using `file.h5:/group/dataset` when an internal path is needed;
- in-memory Julia arrays: 2D images, 3D cubes, RGB/RGBA data;
- `NamedTuple` or `Dict` collections of arrays for multi-panel displays.

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

The setup script installs the Julia dependencies and precompiles the project
environment.

## Quick Start

Run the demo:

```bash
./manta
```

The demo creates a synthetic FITS cube in `demo/output/`, then opens the
interactive viewer.

Open a FITS file:

```bash
./manta path/to/cube.fits
```

Run the demo with custom dimensions:

```bash
NX=96 NY=72 NZ=48 ./manta
```

Set the demo contrast limits at startup:

```bash
VMIN=0 VMAX=100 ./manta
```

## Julia Usage

From the repository root:

```bash
julia --project=.
```

Then:

```julia
using MANTA

fig = MANTA.manta("path/to/cube.fits"; cmap=:magma)
display(fig)
```

With a few options:

```julia
fig = MANTA.manta(
    "path/to/cube.fits";
    cmap=:viridis,
    vmin=0,
    vmax=100,
    invert=false,
    figsize=(1400, 900),
    save_dir="outputs",
    settings_path="viewer_settings.toml",
)
display(fig)
```

## Useful Options

Common options:

- `cmap`: Makie colormap, for example `:viridis`, `:magma`, `:inferno`,
  `:plasma`, `:cividis`, or `:gray`;
- `vmin`, `vmax`: manual contrast limits;
- `invert`: reverse the colormap;
- `figsize`: window size, for example `(1400, 900)`;
- `save_dir`: export directory;
- `settings_path`: TOML file used to save and reload viewer state;
- `scale`: `:lin`, `:log10`, or `:ln` for views that support scaling;
- `hist_mode`: histogram mode, either `:bars` or `:kde`;
- `hist_bins`: number of histogram bins;
- `hist_xlimits`: manual x-axis limits for the histogram;
- `activate_gl=false`: useful for tests without an OpenGL context;
- `display_fig=false`: create the figure without displaying it automatically.

More specialized options:

- `rgb=true`: interpret a 3- or 4-channel stack as RGB/RGBA;
- `column`: column to read from a HEALPix FITS table;
- `nx`, `ny`: Mollweide projection resolution;
- `v0`, `dv`, `vunit`: spectral axis definition for HEALPix-PPV cubes.

## Viewer Possibilities

For a 3D cube, MANTA can:

- choose the slice axis and current slice index;
- click a voxel and display its spectrum;
- average spectra inside a box or circular selection;
- switch image and spectrum scales between linear, log10, and ln;
- adjust contrast manually, automatically, or with percentile presets;
- change or invert the colormap;
- smooth the displayed image;
- add automatic or manual contours;
- compare with a second cube of the same size;
- compute moment 0, 1, and 2 maps;
- export images, spectra, FITS products, and GIFs;
- save and reload viewer settings.

For HEALPix data, MANTA provides:

- Mollweide projection;
- right-click drag zoom;
- optional coordinate graticule;
- pixel or region selection;
- spectral channels for PPV cubes;
- contours, contrast control, colormap control, smoothing, and PNG/FITS exports
  depending on the view.

## HDF5

MANTA can read an HDF5 dataset directly:

```julia
fig = MANTA.manta("data/map.h5:/group/dataset")
display(fig)
```

If the path points to a group, MANTA tries to find a single child dataset, or
uses the `default_dataset` attribute when present.

Some HDF5 attributes are recognized when available:

- `units` or `bunit` for the unit label;
- `AXIS1NAME`, `AXIS2NAME`, etc. for axis names;
- `CTYPE`, `CRVAL`, `CRPIX`, `CDELT`, `CUNIT` for simple linear WCS metadata;
- `PIXTYPE = HEALPIX`, `ORDERING`, `NSIDE`, `COORDSYS` for HEALPix data;
- `v0`, `dv`, `vunit` for HEALPix-PPV cubes.

## In-Memory Data

MANTA can also display Julia data directly:

```julia
using MANTA

image = rand(128, 128)
MANTA.manta(image; title="image")

cube = rand(64, 64, 32)
MANTA.manta(cube; cmap=:magma)
```

1D vectors have a dedicated viewer with a `lines` plot, live statistics over an
optional selection range (n, finite, NaN, min, max, mean, std, median), `:lin /
:log10 / :ln` scale toggles for both axes, and PNG / PDF / CSV exports:

```julia
using MANTA

# A plain numeric vector goes through the AbstractVector bridge.
MANTA.manta(sin.(range(0.0, 4π; length = 512));
            title  = "sin(t)",
            yscale = :lin)

# A 1D FITS file (single image-HDU vector) opens the same viewer; if the
# header carries CTYPE1/CRVAL1/CDELT1/CUNIT1, the X axis is automatically
# WCS-mapped (e.g. frequency in Hz, velocity in km/s).
MANTA.manta("path/to/spectrum.fits")

# A VectorDataset can be built explicitly for custom labels / WCS.
ds = MANTA.VectorDataset(rand(Float32, 256);
        axis_label = "channel",
        unit_label = "K",
        source_id  = "my_spectrum")
MANTA.manta(ds; xlimits = (10, 200))
```

Keyword arguments for the 1D viewer: `title`, `xscale`, `yscale`, `xlimits`,
`ylimits`, `figsize`, `save_dir`, plus the usual `activate_gl` / `display_fig`
toggles for headless usage. The exported CSV contains the values currently
within the X-axis selection (or the full vector if no selection is set) and a
self-describing comment header.

Multiple panels:

```julia
MANTA.manta_panels(
    rand(128, 128),
    rand(128, 128);
    titles=("map A", "map B"),
    cmaps=(:viridis, :magma),
)
```

RGB image:

```julia
rgb = MANTA.rgb_image(U, V, W; normalize=:symmetric)
MANTA.manta(rgb; title="RGB")
```

## Development Commands

Install or update dependencies:

```bash
julia --project=. scripts/setup.jl
```

Run tests:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Check that the package loads:

```bash
julia --project=. -e 'using MANTA; println("MANTA OK")'
```

Show the Julia environment status:

```bash
julia --project=. -e 'using Pkg; Pkg.status()'
```

## Docker

Docker usage is documented in [DOCKER.md](DOCKER.md).

Short version:

```bash
docker build -t manta .
docker run --rm manta julia --project=. -e 'using Pkg; Pkg.test()'
```

Opening an interactive window from Docker requires graphical access, such as X11
on Linux or XQuartz on macOS.

## Troubleshooting

If no window opens, test GLMakie:

```bash
julia --project=. -e 'using GLMakie; display(GLMakie.Figure()); sleep(2)'
```

If this command fails, the issue is probably the graphical environment or
OpenGL.

If a dependency is missing:

```bash
julia --project=. scripts/setup.jl
```

If `log10` or `ln` scales produce empty regions, check that the displayed
values are strictly positive. Values less than or equal to zero are invalid for
those scales.

## Repository Layout

```text
.
|-- Project.toml
|-- README.md
|-- manta
|-- demo/
|   `-- run_demo.jl
|-- scripts/
|   `-- setup.jl
|-- src/
|   |-- MANTA.jl
|   |-- MANTAHealpix.jl
|   |-- datasets/
|   |-- loaders/
|   |-- views/
|   `-- helpers/
`-- test/
    `-- runtests.jl
```
