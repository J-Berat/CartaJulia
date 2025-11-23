# CartaViewer

Interactive 3D FITS cube viewer built with **GLMakie**.  
View 2D slices along any axis, inspect the spectrum at the selected voxel, adjust scales/colormap, smooth with a Gaussian filter, and export PNG/PDF/GIF.

## Features
- Slice navigation along axes 1/2/3 with a movable crosshair.
- Per-pixel spectrum plot synced with the selected voxel.
- Image and spectrum scaling: `lin`, `log10`, `ln`.
- Colormap selection (by name) with invert toggle.
- Optional Gaussian smoothing with tunable sigma.
- Auto or manual colorbar limits (manual syncs spectrum Y-limits).
- Exports:
  - Full figure or separate slice/spectrum (PNG/PDF).
  - Animated GIF across slice indices (with optional ping-pong).

## Requirements
- Julia 1.9+ (tested up to 1.12).
- Working OpenGL context (GLMakie). macOS, Linux, and Windows are supported.

## Install
From the project root:
```bash
julia --project -e 'import Pkg; Pkg.instantiate()'
julia --project scripts/setup.jl     # ensures all deps are added
```

## Quick Start (Demo)
```bash
julia --project demo/run_demo.jl
```

## Usage (REPL)
```julia
julia --project
julia> push!(LOAD_PATH, "src"); using CartaViewer
julia> fig = CartaViewer.carta("path/to/your_cube.fits"; fullscreen=true)
```

## API
```julia
carta(filepath::String;
      cmap::Symbol = :viridis,
      vmin = nothing,                      # Float-like or nothing
      vmax = nothing,                      # Float-like or nothing
      invert::Bool = false,
      fullscreen::Bool = false,            # fit window to primary monitor
      size::Union{Nothing,Tuple{Int,Int}} = nothing)  # explicit (w,h), overrides fullscreen
```
Notes:
- If both `vmin` and `vmax` are provided, manual limits are enabled and the spectrum Y-limits are synced to `[vmin, vmax]`.
- `fullscreen=true` sizes the figure to your primary monitor resolution.
- Use `size=(w, h)` for a specific window size (takes precedence over `fullscreen`).

## UI Tips
- Arrow keys: move the crosshair (row/col) within the current slice.
- Mouse click on the image: pick a voxel.
- `i` key: invert the colormap.
- Image scale / Spectrum scale menus: choose `lin`, `log10`, or `ln`.
- Gaussian filter: enable the checkbox and tune σ with the slider.
- Colorbar limits: enter `min` and `max`, then click Apply.
- Save fig / Save slice+spec: writes to `~/Desktop` (or current directory if Desktop is not available).
- Export GIF: set `start/stop/step/fps`; Ping-pong repeats the sequence forward and backward.

## Testing
```bash
julia --project -e 'using Pkg; Pkg.test()'
```

## Troubleshooting
- GL window doesn’t appear: ensure you run with a GPU-capable OpenGL context (avoid headless SSH without proper display; on Linux, set `DISPLAY`).
- Text/LaTeX issues: the viewer uses inline LaTeX via `LaTeXStrings`; no LaTeX line breaks are used.
- Tiny UI on HiDPI (Retina): try `fullscreen=true` or launch with `size=(w, h)`; OS-level scaling also helps.
- GIF export fails headlessly: GIF recording needs an active OpenGL context.
