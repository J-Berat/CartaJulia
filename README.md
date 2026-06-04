# MANTA

MANTA is an interactive astronomical data viewer written in Julia.
It is built on top of Makie/GLMakie and is designed for quick visual exploration
of maps, cubes, spectra, and HEALPix data — with no boilerplate required.

The single entry point is `MANTA.manta(...)`. Pass it a file path or a Julia
array; MANTA dispatches automatically to the right viewer.

## What MANTA Can Open

- 2D FITS images;
- 3D FITS cubes, with slice navigation and a per-voxel or per-region spectrum;
- HEALPix FITS maps, displayed in Mollweide projection;
- HEALPix-PPV cubes (npix × nchannels), with one map per channel and a spectrum panel;
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
precompiles the project environment (including the precompile workload, which
dramatically reduces time-to-first-plot on subsequent launches).

## Quick Start

Run the built-in demo (generates a synthetic FITS cube and opens the viewer):

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

When running outside a REPL (e.g. from a script), keep the process alive until
the user closes the window with `wait_until_closed`, which is more reliable than
polling `isopen(fig.scene)`:

```julia
fig = MANTA.manta("path/to/cube.fits")
display(fig)
MANTA.wait_until_closed(fig)
```

With options:

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

## Viewer Options Reference

The table below covers all keyword arguments accepted by `manta`. Most apply to
every viewer; a few are specific to cubes or HEALPix data.

### Appearance

| kwarg | type | default | meaning |
|---|---|---|---|
| `cmap` | `Symbol` | `:viridis` | Makie colormap (`:magma`, `:inferno`, `:plasma`, `:cividis`, `:gray`, …) |
| `invert` | `Bool` | `false` | Reverse the colormap |
| `figsize` | `Tuple{Int,Int}` | auto | Window size in pixels, e.g. `(1400, 900)` |
| `scale` | `Symbol` | `:lin` | Image scale: `:lin`, `:log10`, `:ln`, `:asinh`, or `:sqrt` |
| `asinh_softening` | `Real` | `1.0` | Softening length `a` for the `:asinh` stretch (`asinh(x/a)`); clamped > 0. Small `a` ≈ logarithmic, large `a` ≈ linear |

### Contrast

| kwarg | type | default | meaning |
|---|---|---|---|
| `vmin`, `vmax` | `Real` | `nothing` | Manual contrast limits; auto if omitted |
| `hist_mode` | `Symbol` | `:bars` | Histogram display mode: `:bars` or `:kde` |
| `hist_bins` | `Int` | `64` | Number of histogram bins |
| `hist_xlimits` | `Tuple{Real,Real}` | `nothing` | Manual x-axis limits for the histogram |
| `hist_ylimits` | `Tuple{Real,Real}` | `nothing` | Manual y-axis limits for the histogram |
| `spec_ylimits` | `Tuple{Real,Real}` | `nothing` | Manual y-axis limits for the spectrum panel (cubes and HEALPix-PPV) |

### Moment maps (cubes and HEALPix-PPV)

| kwarg | type | default | meaning |
|---|---|---|---|
| `moment_threshold` | `Real` | `0.0` | Minimum value included in moment calculations |
| `moment_nsigma` | `Real` | `nothing` | If set, threshold is computed as `nsigma × σ` of the data |
| `moment_channels` | `AbstractVector{Int}` | `nothing` | Restrict moment calculations to these channel indices |

### FITS / HDF5-specific

| kwarg | type | default | meaning |
|---|---|---|---|
| `hdu` | `Integer` | `1` | HDU index to read (1 = primary; 0 = auto-pick first non-empty HDU). FITS only; ignored for HDF5 |
| `lazy` | `Bool` | `false` | Read slices on demand instead of loading the full file up-front. Supported for FITS cubes and for 2D/3D HDF5 datasets. Ignored for in-memory inputs. See [Lazy Loading](#lazy-loading). |

### Cube-specific

| kwarg | type | default | meaning |
|---|---|---|---|
| `settings_path` | `String` | `nothing` | Path to a TOML file used to save and reload viewer state |
| `compare` | `String` | `nothing` | Path to a second FITS cube for side-by-side comparison |
| `state` | any | `nothing` | Pre-load a viewer state snapshot (a `Dict` or `NamedTuple` as produced by "Copy code" or `save_viewer_settings`) |
| `rgb` | `Bool` | `false` | Interpret a 3- or 4-channel stack as RGB/RGBA instead of a cube |

### HEALPix-specific

| kwarg | type | default | meaning |
|---|---|---|---|
| `column` | `Int` | `1` | Column index in the BinTable HDU |
| `nx`, `ny` | `Int` | `1400`, `700` | Mollweide grid resolution |
| `v0`, `dv`, `vunit` | `Real`, `Real`, `String` | `0.0`, `1.0`, `"km/s"` | Spectral axis definition for HEALPix-PPV cubes |

### Headless / testing

| kwarg | type | default | meaning |
|---|---|---|---|
| `activate_gl` | `Bool` | `true` | Set to `false` to skip GLMakie activation (useful in tests and Docker) |
| `display_fig` | `Bool` | `true` | Set to `false` to build the figure without displaying it |

### Output

| kwarg | type | default | meaning |
|---|---|---|---|
| `save_dir` | `String` | `nothing` | Export directory; defaults to `~/Desktop` if it exists, otherwise the current directory |

## Dark Mode

MANTA ships with two UI themes: the default light theme and an anthracite dark
theme. In the cube viewer, click the `☾` / `☀` button in the bottom mode bar
or press `D` to toggle the active window.

You can also choose the default theme for new viewers globally for the current
Julia session with `set_dark_mode!`, called before opening a viewer:

```julia
using MANTA

MANTA.set_dark_mode!(true)    # dark background, bright widgets
MANTA.manta("path/to/cube.fits")

MANTA.set_dark_mode!(false)   # back to the standard light theme
```

The toggle updates both MANTA's own panel and widget colours and the Makie
global theme, so axes, tick labels, grid lines, and figure backgrounds follow
along. Query or fetch the active theme programmatically with `is_dark_mode()`,
`current_ui_theme()`, and `dark_ui_theme()`.

## Viewer Capabilities

### 3D Cube Viewer

Once a cube is open, the viewer lets you:

- navigate slices along any of the three axes with a slider or keyboard arrows;
- click a pixel to display its spectrum; draw a box or circular region to
  average spectra inside it;
- switch image and spectrum scales between `:lin`, `:log10`, `:ln`, `:asinh`, and `:sqrt`;
- adjust contrast manually, automatically, or with one-click p1-p99 / p5-p95 presets;
- change or invert the colormap; smooth the displayed image;
- add automatic or manual contours;
- compare with a second cube of the same dimensions (`compare` kwarg or the UI button);
- compute moment 0, 1, and 2 maps with configurable threshold and channel range;
- apply a persistent voxel mask (see [Masks](#masks));
- analyze the 2D power spectrum of any slice (see [Power Spectrum](#power-spectrum));
- undo and redo viewer-state changes (Ctrl-Z / Ctrl-Shift-Z; see [Undo / Redo](#undo--redo));
- export images (PNG/PDF), spectra (CSV), FITS products (slices, moments, region
  means, full cubes, mask), and animated GIFs;
- save and reload viewer settings to a TOML file; copy a Julia snippet that
  reproduces the current view exactly.

### HEALPix Viewer

For HEALPix data, MANTA provides:

- Mollweide projection with right-click drag zoom;
- an optional coordinate graticule;
- pixel or region selection;
- spectral channel navigation for PPV cubes;
- contours, contrast control, colormap control, smoothing;
- PNG and FITS exports.

### 2D Image Viewer

Single images get a lightweight viewer with a histogram, contrast controls,
colormap menu, scale toggle (:lin / :log10 / :ln), and PNG export.

### 1D Vector Viewer

Vectors have a dedicated viewer with a `lines` plot, live statistics over an
optional selection range (n, finite, NaN, min, max, mean, std, median),
`:lin / :log10 / :ln` scale toggles for both axes, and PNG / PDF / CSV exports:

```julia
using MANTA

# A plain numeric vector.
MANTA.manta(sin.(range(0.0, 4π; length=512)); title="sin(t)")

# A 1D FITS file whose header carries CTYPE1/CRVAL1/CDELT1/CUNIT1 maps the
# X axis automatically (e.g. frequency in Hz, velocity in km/s).
MANTA.manta("path/to/spectrum.fits")

# Build a VectorDataset explicitly for custom labels or WCS.
ds = MANTA.VectorDataset(rand(Float32, 256);
        axis_label = "channel",
        unit_label = "K",
        source_id  = "my_spectrum")
MANTA.manta(ds; xlimits=(10, 200))
```

Keyword arguments for the 1D viewer: `title`, `xscale`, `yscale`, `xlimits`,
`ylimits`, `figsize`, `save_dir`, plus the usual `activate_gl` / `display_fig`
toggles for headless usage.

## In-Memory Data

MANTA can display Julia arrays directly without any file on disk:

```julia
using MANTA

# 2D image
MANTA.manta(rand(128, 128); cmap=:magma)

# 3D cube — gets the full interactive slice + spectrum + moments viewer
MANTA.manta(rand(64, 64, 32); cmap=:viridis, vmin=0.0, vmax=1.0)

# RGB image from three velocity components
rgb = MANTA.rgb_image(U, V, W; normalize=:symmetric)
MANTA.manta(rgb; title="RGB")

# Multi-panel (NamedTuple or positional)
MANTA.manta_panels(
    rand(128, 128),
    rand(128, 128);
    titles = ("map A", "map B"),
    cmaps  = (:viridis, :magma),
)
```

## Batch Export

`manta_batch` renders a list of FITS files headlessly and writes one image per
file. No GL context or window is required.

```julia
using MANTA

out_paths = MANTA.manta_batch(
    ["obs1.fits", "obs2.fits", "obs3.fits"];
    format   = :png,          # :png (default) or :pdf
    save_dir = "exports",     # defaults to each file's own directory
    prefix   = "survey_",     # optional filename prefix
    cmap     = :magma,
    vmin     = 0,
    vmax     = 500,
    figsize  = (1200, 800),
)
# => ["exports/survey_obs1.png", "exports/survey_obs2.png", ...]
```

All `kwargs` are forwarded verbatim to `manta` with `activate_gl=false,
display_fig=false`, so no interactive state is created. Files that fail to
render are skipped with a `@warn` and do not abort the batch; the return value
contains only the paths that were actually written.

## Lazy Loading

By default, MANTA reads the full cube into memory at startup. For large files
this can be slow or exceed available RAM. Passing `lazy=true` enables on-demand
slice reading:

```julia
# FITS cube
fig = MANTA.manta("path/to/large_cube.fits"; lazy=true)

# HDF5 dataset (2D images and 3D cubes)
fig = MANTA.manta("path/to/large.h5:/group/dataset"; lazy=true)
```

With lazy loading, MANTA opens the file and reads only the currently displayed
slice (via memory-mapping for FITS and hyperslab reads for HDF5). FITS cubes
keep a small LRU cache of recently visited slices (7 by default), and an async
prefetch mechanism starts loading the next slice in the background as soon as
the current one is rendered, so interactive navigation stays smooth even when
scrolling back and forth. Memory usage stays proportional to a handful of
slices rather than the full cube.

Lazy HDF5 loading covers 2D and 3D numeric datasets only; HEALPix datasets and
non-numeric data are always read eagerly. The `lazy` flag is ignored for
in-memory array inputs.

> **Note:** lazy loading is a read-only view; exports that require the full
> cube (e.g. "Save cube FITS") will materialise all slices at export time.

## Masks

The 3D cube viewer carries an optional persistent voxel mask. Only the source
description is stored (not the full `BitArray`), so the mask can always be
regenerated from the data when a settings file is reloaded.

### Mask sources

| Source | Keeps |
|---|---|
| `NoMaskSource()` | Everything (mask disabled) |
| `FiniteSource()` | Voxels where `isfinite(data)` |
| `ThresholdSource(:ge, lo, _)` | `data >= lo` |
| `ThresholdSource(:le, _, hi)` | `data <= hi` |
| `ThresholdSource(:range, lo, hi)` | `lo <= data <= hi` |
| `ThresholdSource(:outside, lo, hi)` | `data < lo \|\| data > hi` |
| `RectangleSource(; i1, i2, j1, j2, k1, k2)` | Voxels in the closed index box; any bound can be `nothing` |

Non-finite voxels are always excluded by threshold sources, so NaNs never
leak into moment accumulators.

### Mask combinators

Sources can be composed arbitrarily:

```julia
src = AndSource(ThresholdSource(:ge, 0.5, 0.0), NotSource(RectangleSource(i2=32)))
```

`AndSource`, `OrSource`, and `NotSource` are all exported and round-trip
through `viewer_settings.toml` via `mask_source_to_toml` / `mask_source_from_toml`.

### Effect of the mask

An active mask propagates to:

- moment 0 / 1 / 2 maps (`moment_map` accepts an optional `mask` kwarg);
- the histogram of the displayed slice;
- per-voxel and per-region spectra (`mean_region_spectrum` likewise accepts `mask`);
- FITS exports — the `"mask"` export product writes the materialised `BitArray`
  as an `Int8` HDU.

### Programmatic usage

```julia
using MANTA

cube = rand(Float32, 64, 64, 32)

src  = ThresholdSource(:ge, 0.5, 0.0)      # keep voxels >= 0.5
mask = make_mask(src, cube)
@info "kept $(mask_count(mask)) of $(mask_total(mask)) voxels"

# Pass the materialised BitArray to moment_map:
m0 = moment_map(cube, 3, 1:32, 0; mask=mask.bits)

# Serialize / deserialize a source (e.g. for settings files):
d    = mask_source_to_toml(src)
src2 = mask_source_from_toml(d)
```

The mask source is persisted under the `"mask"` key of `viewer_settings.toml`.
Malformed or unknown entries silently degrade to `NoMaskSource()` so a
corrupted settings file never blocks a viewer launch.

## Power Spectrum

The 3D cube viewer has a built-in 2D power-spectrum panel. It computes:

- the 2D power spectrum of the currently displayed slice, shown as a log₁₀
  heatmap on a (kx, ky) grid;
- the 1D radially-averaged profile on a log-log plot.

Controls let you choose the apodization window (`:none`, `:hann`, `:tukey`, …),
toggle zero-padding to the next power of two, apply NaN apodization, and switch
the k-axis between cycles/pixel and physical units (when WCS is available). The
panel can be detached into a separate pop-out window.

No additional setup is required; the power-spectrum tab is always present in the
cube viewer UI.

## Undo / Redo

The 3D cube viewer records a snapshot history of the viewer state (contrast
limits, scale mode, axis, slice index, colormap, contour visibility, …). You
can step through this history with:

- **Ctrl-Z** — undo the last state change;
- **Ctrl-Shift-Z** — redo.

The history is bounded to 64 snapshots, so memory usage is predictable even
after thousands of UI interactions. Consecutive identical snapshots are
deduplicated, so dragging a slider quickly still produces a sensible history.

## Keyboard Shortcuts

All viewers expose a **Help** button (or **Shift+/**) that opens a floating
window listing every available shortcut for the current view.

Common shortcuts in the **3D cube viewer**:

| Key | Action |
|---|---|
| ←  / → | Previous / next slice |
| ↑  / ↓ | Previous / next slice (alternate) |
| `a` | Auto contrast |
| `1` | p1–p99 contrast preset |
| `5` | p5–p95 contrast preset |
| `i` | Invert colormap |
| `l` | Cycle scale (lin → log10 → ln → asinh → sqrt) |
| `c` | Toggle contours |
| `r` | Reset zoom |
| `+` / `-` | Zoom in / out (centered) |
| `s` | Save image |
| `Ctrl-Z` | Undo |
| `Ctrl-Shift-Z` | Redo |
| `Shift-/` | Open shortcut help window |

Common shortcuts in the **2D image viewer**:

| Key | Action |
|---|---|
| `a` | Auto contrast |
| `1` | p1–p99 contrast preset |
| `5` | p5–p95 contrast preset |
| `i` | Invert colormap |
| `l` | Cycle scale (lin → log10 → ln → asinh → sqrt) |
| `g` | Toggle WCS graticule (when the image carries a sky WCS) |
| `r` | Reset zoom |
| `+` / `-` | Zoom in / out (centered) |
| `s` | Save image (PNG) |
| `Shift-/` | Open shortcut help window |

The **HEALPix map and PPV-cube viewers** share the same conventions: `g`
toggles the coordinate graticule and `+` / `-` zoom the Mollweide view in and
out about its center (the numeric keypad `+` / `-` work too, in every viewer).

## Plugin System

MANTA exposes a lightweight extension API for adding new file-format loaders,
custom view modes, or post-processing overlays — without modifying MANTA itself.

Three extension points are available:

- **`:loader`** — register a `(matcher, loader)` pair. `matcher(path)::Bool`
  returns `true` when your loader can handle the file; `loader(path; kwargs...)`
  must return an `AbstractMANTADataset`.
- **`:dataset_view`** — register a `(DatasetType, name, fn)` triple to add a
  named view mode for an existing dataset type.
- **`:postprocess`** — register a `(fig, ds, opts) -> nothing` callback invoked
  after a view is fully built (useful for overlays or custom shortcuts).

```julia
using MANTA

# Add a loader for .myformat files.
my_loader = MANTA.register_plugin!(:loader, (
    path -> endswith(path, ".myformat"),
    (path; kwargs...) -> MyPackage.load_as_manta(path),
))

# Remove it later (by identity).
MANTA.unregister_plugin!(:loader, my_loader)

# List currently registered loaders.
MANTA.list_plugins(:loader)
```

Plugins are dispatched in registration order; the first matching loader wins.
`MANTA.clear_plugins!()` removes everything (used in tests).

## HDF5

MANTA reads an HDF5 dataset with the `"path.h5:/group/dataset"` syntax:

```julia
fig = MANTA.manta("data/map.h5:/group/dataset")
display(fig)
```

If the path points to an HDF5 group, MANTA tries to find a single child
dataset or uses the `default_dataset` attribute when present.

HDF5 attributes recognized when available:

- `units` or `bunit` for the unit label;
- `AXIS1NAME`, `AXIS2NAME`, … for axis names;
- `CTYPE`, `CRVAL`, `CRPIX`, `CDELT`, `CUNIT` for linear WCS metadata;
- `PIXTYPE = HEALPIX`, `ORDERING`, `NSIDE`, `COORDSYS` for HEALPix data;
- `v0`, `dv`, `vunit` for HEALPix-PPV spectral axes.

## Development Commands

Install or update dependencies:

```bash
julia --project=. scripts/setup.jl
```

Run the test suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Verify the package loads:

```bash
julia --project=. -e 'using MANTA; println("MANTA OK")'
```

Show the environment status:

```bash
julia --project=. -e 'using Pkg; Pkg.status()'
```

> Tests are designed to run without an OpenGL context (`activate_gl=false,
> display_fig=false`). Every new visual feature must expose a headless-compatible
> code path for CI and Docker.

## Docker

Docker usage is documented in [DOCKER.md](DOCKER.md).

Short version:

```bash
docker build -t manta .
docker run --rm manta julia --project=. -e 'using Pkg; Pkg.test()'
```

Opening an interactive window from Docker requires graphical access — X11
forwarding on Linux, or XQuartz on macOS.

## Troubleshooting

**No window opens:**

```bash
julia --project=. -e 'using GLMakie; display(GLMakie.Figure()); sleep(2)'
```

If this fails, the issue is the graphical environment or OpenGL, not MANTA.

**A dependency is missing:**

```bash
julia --project=. scripts/setup.jl
```

**`log10` / `ln` / `sqrt` scales show empty or blank regions:**
Values ≤ 0 are invalid for these scales; MANTA converts them to `NaN` before
display. Check that the data has strictly positive values in the region you are
viewing. For data with both positive and negative pixels (e.g. low-SNR maps),
prefer the `:asinh` stretch, which is defined on all of ℝ and keeps the sign;
tune its softening length with `asinh_softening` (`asinh(x / a)`).

**Lazy loading is slow on first slice:**
The async prefetch covers the *next* slice; the very first read always hits
disk. On networked filesystems, consider a local copy.

## Repository Layout

```text
.
├── Project.toml          Julia environment manifest
├── README.md
├── DOCKER.md
├── Dockerfile
├── docker-entrypoint.sh
├── manta                 Shell launcher
├── demo/
│   └── run_demo.jl       Synthetic demo cube + viewer
├── scripts/
│   └── setup.jl          Dependency installer & precompiler
├── src/
│   ├── MANTA.jl          Module entry point; public API dispatch
│   ├── MANTAHealpix.jl   HEALPix viewers (map, PPV cube, panels, RGB)
│   ├── datasets/
│   │   ├── Datasets.jl   Dataset types (CubeDataset, ImageDataset, …)
│   │   └── LoadDataset.jl  Dataset loader dispatch
│   ├── loaders/
│   │   ├── FITSLoader.jl
│   │   ├── HDF5Loader.jl
│   │   ├── InMemoryLoader.jl
│   │   ├── LazyFITS.jl   On-demand FITS slice reader with async prefetch
│   │   └── LazyHDF5.jl   On-demand HDF5 hyperslab reader with async prefetch
│   ├── masking/
│   │   └── Mask.jl       Declarative voxel mask system
│   ├── helpers/
│   │   ├── Helpers.jl    UI utilities (colormaps, LaTeX, contour parsing, …)
│   │   ├── UITheme.jl    Light/dark themes and widget style helpers
│   │   ├── UIConstants.jl  Shared UI sizing/spacing constants
│   │   ├── UIBits.jl     Reusable control-card / label builders
│   │   ├── Shortcuts.jl  Keyboard shortcut registration
│   │   ├── UndoRedo.jl   Bounded snapshot history (Ctrl-Z / Ctrl-Shift-Z)
│   │   ├── Plugins.jl    Extension point registry
│   │   ├── PowerSpectrum.jl  2D FFT + radial profile helpers
│   │   └── …             (Backend, Contours, Downsample, Errors, FITSHeaders,
│   │                      Images, Moments, Progress, Scaling, Slicing,
│   │                      Stats, WCS)
│   └── views/
│       ├── CubeView.jl   3D cube viewer (main orchestrator)
│       ├── HealpixMapView.jl   HEALPix map viewer
│       ├── HealpixProjection.jl  Mollweide projection helpers
│       ├── VectorView.jl       1D spectrum / vector viewer
│       └── cube/         Cube-viewer sub-bundles
│           ├── CubeViewState.jl
│           ├── CubeLayout.jl
│           ├── MaskBundle.jl
│           ├── CompareBundle.jl
│           ├── KeyboardBundle.jl
│           ├── ExportBundle.jl
│           ├── PowerSpectrumBundle.jl
│           ├── PSWindowBundle.jl
│           ├── SlicePipelineBundle.jl
│           ├── SpectrumBundle.jl
│           ├── UICallbacksBundle.jl
│           ├── SettingsBundle.jl
│           └── AnimationRequest.jl
└── test/
    ├── runtests.jl       Headless test entry point
    ├── cube_view.jl
    ├── healpix.jl
    ├── helpers.jl
    ├── lazy_fits.jl
    ├── lazy_hdf5.jl
    ├── loaders.jl
    └── masks.jl
```
