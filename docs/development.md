# MANTA Development

This page collects development commands, Docker notes, troubleshooting, and the
repository map.

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

> Tests are designed to run without an OpenGL context (`activate_gl=false`,
> `display_fig=false`). Every new visual feature must expose a
> headless-compatible code path for CI and Docker.

## Docker

Docker usage is documented in [../DOCKER.md](../DOCKER.md).

Short version:

```bash
docker build -t manta .
docker run --rm manta julia --project=. -e 'using Pkg; Pkg.test()'
```

Opening an interactive window from Docker requires graphical access: X11
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

Values <= 0 are invalid for these scales; MANTA converts them to `NaN` before
display. Check that the data has strictly positive values in the region you are
viewing. For data with both positive and negative pixels, such as low-SNR maps,
prefer the `:asinh` stretch, which is defined on all real values and keeps the
sign. Tune its softening length with `asinh_softening` (`asinh(x / a)`).

**Lazy loading is slow on first slice:**

The async prefetch covers the next slice; the very first read always hits disk.
On networked filesystems, consider a local copy.

## Repository Layout

```text
.
├── Project.toml          Julia environment manifest
├── README.md
├── docs/
│   ├── user-guide.md
│   ├── reference.md
│   └── development.md
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
│   │   ├── Datasets.jl   Dataset types (CubeDataset, ImageDataset, ...)
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
│   │   ├── Helpers.jl    UI utilities (colormaps, LaTeX, contour parsing, ...)
│   │   ├── UITheme.jl    Light/dark themes and widget style helpers
│   │   ├── UIConstants.jl  Shared UI sizing/spacing constants
│   │   ├── UIBits.jl     Reusable control-card / label builders
│   │   ├── Shortcuts.jl  Keyboard shortcut registration
│   │   ├── UndoRedo.jl   Bounded snapshot history (Ctrl-Z / Ctrl-Shift-Z)
│   │   ├── Plugins.jl    Extension point registry
│   │   ├── PowerSpectrum.jl  2D FFT + radial profile helpers
│   │   └── ...           Backend, Contours, Downsample, Errors, FITSHeaders,
│   │                      Images, Moments, Progress, Scaling, Slicing,
│   │                      Stats, WCS
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
