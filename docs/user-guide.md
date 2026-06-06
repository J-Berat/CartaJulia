# MANTA User Guide

This guide covers day-to-day viewer behavior and common workflows. For the full
keyword argument table, see [Reference](reference.md).

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
- apply a persistent voxel mask;
- analyze the 2D power spectrum of any slice;
- undo and redo viewer-state changes with Ctrl-Z / Ctrl-Shift-Z;
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
colormap menu, scale toggle (`:lin` / `:log10` / `:ln`), and PNG export.

### 1D Vector Viewer

Vectors have a dedicated viewer with a `lines` plot, live statistics over an
optional selection range (`n`, finite, NaN, min, max, mean, std, median),
`:lin` / `:log10` / `:ln` scale toggles for both axes, and PNG / PDF / CSV exports:

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

# 3D cube: gets the full interactive slice + spectrum + moments viewer
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

All `kwargs` are forwarded verbatim to `manta` with `activate_gl=false`,
`display_fig=false`, so no interactive state is created. Files that fail to
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

### Mask Sources

| Source | Keeps |
|---|---|
| `NoMaskSource()` | Everything (mask disabled) |
| `FiniteSource()` | Voxels where `isfinite(data)` |
| `ThresholdSource(:ge, lo, _)` | `data >= lo` |
| `ThresholdSource(:le, _, hi)` | `data <= hi` |
| `ThresholdSource(:range, lo, hi)` | `lo <= data <= hi` |
| `ThresholdSource(:outside, lo, hi)` | `data < lo \|\| data > hi` |
| `RectangleSource(; i1, i2, j1, j2, k1, k2)` | Voxels in the closed index box; any bound can be `nothing` |

Non-finite voxels are always excluded by threshold sources, so NaNs never leak
into moment accumulators.

### Mask Combinators

Sources can be composed arbitrarily:

```julia
src = AndSource(ThresholdSource(:ge, 0.5, 0.0), NotSource(RectangleSource(i2=32)))
```

`AndSource`, `OrSource`, and `NotSource` are all exported and round-trip through
`viewer_settings.toml` via `mask_source_to_toml` / `mask_source_from_toml`.

### Effect Of The Mask

An active mask propagates to:

- moment 0 / 1 / 2 maps (`moment_map` accepts an optional `mask` kwarg);
- the histogram of the displayed slice;
- per-voxel and per-region spectra (`mean_region_spectrum` likewise accepts `mask`);
- FITS exports: the `"mask"` export product writes the materialised `BitArray`
  as an `Int8` HDU.

### Programmatic Usage

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
Malformed or unknown entries silently degrade to `NoMaskSource()` so a corrupted
settings file never blocks a viewer launch.

## Power Spectrum

The 3D cube viewer has a built-in 2D power-spectrum panel. It computes:

- the 2D power spectrum of the currently displayed slice, shown as a `log10`
  heatmap on a `(kx, ky)` grid;
- the 1D radially-averaged profile on a log-log plot.

Controls let you choose the apodization window (`Hann`, `Hamming`, or `None`),
choose a dedicated colormap for the 2D PSD heatmap, toggle zero-padding to the
next power of two, apply NaN apodization, and switch the k-axis between
cycles/pixel and physical units when WCS is available. Set `k_min` / `k_max`,
then click `Fit` to fit a log-log slope over that x window; `Clear` removes the
overlay. The dashed fit line is drawn on the 1D PSD and the slope is shown in
the status text. The panel can be detached into a separate pop-out window.

No additional setup is required; the power-spectrum tab is always present in the
cube viewer UI.

## Undo / Redo

The 3D cube viewer records a snapshot history of the viewer state: contrast
limits, scale mode, axis, slice index, colormap, contour visibility, and other
display settings. You can step through this history with:

- **Ctrl-Z**: undo the last state change;
- **Ctrl-Shift-Z**: redo.

The history is bounded to 64 snapshots, so memory usage is predictable even
after thousands of UI interactions. Consecutive identical snapshots are
deduplicated, so dragging a slider quickly still produces a sensible history.

## Keyboard Shortcuts

All viewers expose a **Help** button (or **Shift+/**) that opens a floating
window listing every available shortcut for the current view.

Common shortcuts in the **3D cube viewer**:

| Key | Action |
|---|---|
| ← / → | Previous / next slice |
| ↑ / ↓ | Previous / next slice (alternate) |
| `a` | Auto contrast |
| `1` | p1-p99 contrast preset |
| `5` | p5-p95 contrast preset |
| `i` | Invert colormap |
| `l` | Cycle scale (`lin` → `log10` → `ln` → `asinh` → `sqrt`) |
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
| `1` | p1-p99 contrast preset |
| `5` | p5-p95 contrast preset |
| `i` | Invert colormap |
| `l` | Cycle scale (`lin` / `log10` / `ln`) |
| `g` | Toggle WCS graticule when the image carries a sky WCS |
| `r` | Reset zoom |
| `+` / `-` | Zoom in / out (centered) |
| `s` | Save image (PNG) |
| `Shift-/` | Open shortcut help window |

The **HEALPix map and PPV-cube viewers** share the same conventions: `g` toggles
the coordinate graticule and `+` / `-` zoom the Mollweide view in and out about
its center. The numeric keypad `+` / `-` work too, in every viewer.
