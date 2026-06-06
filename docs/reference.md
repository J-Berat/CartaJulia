# MANTA Reference

This page collects the complete user-facing reference for `MANTA.manta(...)`,
the extension API, and HDF5 conventions.

## Viewer Options Reference

The table below covers all keyword arguments accepted by `manta`. Most apply to
every viewer; a few are specific to cubes or HEALPix data.

### Appearance

| kwarg | type | default | meaning |
|---|---|---|---|
| `cmap` | `Symbol` | `:viridis` | Makie colormap (`:magma`, `:inferno`, `:plasma`, `:cividis`, `:gray`, ...) |
| `invert` | `Bool` | `false` | Reverse the colormap |
| `figsize` | `Tuple{Int,Int}` | auto | Window size in pixels, e.g. `(1400, 900)` |
| `scale` | `Symbol` | `:lin` | Image scale: `:lin`, `:log10`, `:ln`, `:asinh`, or `:sqrt` |
| `asinh_softening` | `Real` | `1.0` | Softening length `a` for the `:asinh` stretch (`asinh(x/a)`); clamped > 0. Small `a` is approximately logarithmic, large `a` is approximately linear |

### Contrast

| kwarg | type | default | meaning |
|---|---|---|---|
| `vmin`, `vmax` | `Real` | `nothing` | Manual contrast limits; auto if omitted |
| `hist_mode` | `Symbol` | `:bars` | Histogram display mode: `:bars` or `:kde` |
| `hist_bins` | `Int` | `64` | Number of histogram bins |
| `hist_xlimits` | `Tuple{Real,Real}` | `nothing` | Manual x-axis limits for the histogram |
| `hist_ylimits` | `Tuple{Real,Real}` | `nothing` | Manual y-axis limits for the histogram |
| `spec_ylimits` | `Tuple{Real,Real}` | `nothing` | Manual y-axis limits for the spectrum panel (cubes and HEALPix-PPV) |

### Moment Maps

These options apply to cubes and HEALPix-PPV data.

| kwarg | type | default | meaning |
|---|---|---|---|
| `moment_threshold` | `Real` | `0.0` | Minimum value included in moment calculations |
| `moment_nsigma` | `Real` | `nothing` | If set, threshold is computed as `nsigma x sigma` of the data |
| `moment_channels` | `AbstractVector{Int}` | `nothing` | Restrict moment calculations to these channel indices |

### FITS / HDF5

| kwarg | type | default | meaning |
|---|---|---|---|
| `hdu` | `Integer` | `1` | HDU index to read (`1` = primary; `0` = auto-pick first non-empty HDU). FITS only; ignored for HDF5 |
| `lazy` | `Bool` | `false` | Read slices on demand instead of loading the full file up-front. Supported for FITS cubes and for 2D/3D HDF5 datasets. Ignored for in-memory inputs |

### Cube-Specific

| kwarg | type | default | meaning |
|---|---|---|---|
| `settings_path` | `String` | `nothing` | Path to a TOML file used to save and reload viewer state |
| `compare` | `String` | `nothing` | Path to a second FITS cube for side-by-side comparison |
| `state` | any | `nothing` | Pre-load a viewer state snapshot (a `Dict` or `NamedTuple` as produced by "Copy code" or `save_viewer_settings`) |
| `rgb` | `Bool` | `false` | Interpret a 3- or 4-channel stack as RGB/RGBA instead of a cube |

### HEALPix-Specific

| kwarg | type | default | meaning |
|---|---|---|---|
| `column` | `Int` | `1` | Column index in the BinTable HDU |
| `nx`, `ny` | `Int` | `1400`, `700` | Mollweide grid resolution |
| `v0`, `dv`, `vunit` | `Real`, `Real`, `String` | `0.0`, `1.0`, `"km/s"` | Spectral axis definition for HEALPix-PPV cubes |

### Headless / Testing

| kwarg | type | default | meaning |
|---|---|---|---|
| `activate_gl` | `Bool` | `true` | Set to `false` to skip GLMakie activation (useful in tests and Docker) |
| `display_fig` | `Bool` | `true` | Set to `false` to build the figure without displaying it |

### Output

| kwarg | type | default | meaning |
|---|---|---|---|
| `save_dir` | `String` | `nothing` | Export directory; defaults to `~/Desktop` if it exists, otherwise the current directory |

## Plugin System

MANTA exposes a lightweight extension API for adding new file-format loaders,
custom view modes, or post-processing overlays without modifying MANTA itself.

Three extension points are available:

- **`:loader`**: register a `(matcher, loader)` pair. `matcher(path)::Bool`
  returns `true` when your loader can handle the file; `loader(path; kwargs...)`
  must return an `AbstractMANTADataset`.
- **`:dataset_view`**: register a `(DatasetType, name, fn)` triple to add a
  named view mode for an existing dataset type.
- **`:postprocess`**: register a `(fig, ds, opts) -> nothing` callback invoked
  after a view is fully built. This is useful for overlays or custom shortcuts.

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
`MANTA.clear_plugins!()` removes everything and is used in tests.

## HDF5

MANTA reads an HDF5 dataset with the `"path.h5:/group/dataset"` syntax:

```julia
fig = MANTA.manta("data/map.h5:/group/dataset")
display(fig)
```

If the path points to an HDF5 group, MANTA tries to find a single child dataset
or uses the `default_dataset` attribute when present.

HDF5 attributes recognized when available:

- `units` or `bunit` for the unit label;
- `AXIS1NAME`, `AXIS2NAME`, ... for axis names;
- `CTYPE`, `CRVAL`, `CRPIX`, `CDELT`, `CUNIT` for linear WCS metadata;
- `PIXTYPE = HEALPIX`, `ORDERING`, `NSIDE`, `COORDSYS` for HEALPix data;
- `v0`, `dv`, `vunit` for HEALPix-PPV spectral axes.
