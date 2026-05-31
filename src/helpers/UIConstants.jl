# path: src/helpers/UIConstants.jl
#
# Central repository for every magic number / hard-wired UI constant in MANTA.
# All symbols are plain Julia `const` declarations at module scope so they are
# accessible from every file included into the MANTA module without any extra
# import statement.
#
# Organisation
# ─────────────
#   §1  Layout & breakpoints
#   §2  Undo / redo stack
#   §3  Histogram rendering
#   §4  Crosshair rendering
#   §5  Zoom-box interaction
#   §6  Region-selection rendering
#   §7  Contour rendering
#   §8  Scatter markers
#   §9  Mollweide / HEALPix projection
#
# Naming conventions
# ──────────────────
#   ALL_CAPS snake-case — consistent with Julia style for module-level consts.
#   Float32 literals (f0 suffix) for any value that ends up as a Makie attribute
#   (linewidth, alpha, markersize …) to avoid silent Float64 → Float32 casts.
#   Integer literals for pixel counts and bin counts.


############################
# §1  Layout & breakpoints
############################

"""
Width threshold (px) below which the compact layout is activated.

All viewers compare `fig_size[1] <= COMPACT_LAYOUT_W || fig_size[2] <= COMPACT_LAYOUT_H`
to decide between the normal and the compact control arrangement.
"""
const COMPACT_LAYOUT_W = 1500

"""Height threshold (px) — see `COMPACT_LAYOUT_W`."""
const COMPACT_LAYOUT_H = 950


############################
# §2  Undo / redo stack
############################

"""
Maximum number of snapshots kept in an `UndoRedoStack`.

64 entries gives ~50 meaningful user actions while keeping per-viewer memory
overhead well under 1 MB for typical state tuples.
"""
const UNDO_STACK_CAPACITY = 64


############################
# §3  Histogram rendering
############################

"""Default number of histogram bins when the user has not specified one."""
const HIST_BINS_DEFAULT = 64

"""Minimum accepted number of histogram bins (clamped on input)."""
const HIST_BINS_MIN = 4

"""Maximum accepted number of histogram bins (clamped on input)."""
const HIST_BINS_MAX = 512

"""Fill opacity for bar-plot histogram bars."""
const HIST_BAR_ALPHA = 0.44f0

"""Stroke linewidth on individual histogram bars (bar-plot mode)."""
const HIST_BAR_STROKE_LW = 0.3f0

"""Linewidth for the KDE / smooth-curve histogram mode."""
const HIST_KDE_LW = 1.8f0

"""
Linewidth for power-spectrum radial-profile curves in the PS panel.
Shares the same visual weight as `HIST_KDE_LW` by design.
"""
const PS_LINE_LW = 1.8f0

"""Linewidth for the comparison-dataset histogram curve."""
const HIST_COMPARE_LW = 1.6f0

"""
Linewidth for the vertical limit lines drawn at cmin / cmax in the histogram
(2D image viewer and cube viewer).
"""
const HIST_LIMITS_LW = 1.1f0

"""
Linewidth for the vertical limit lines drawn at cmin / cmax in the histogram
(HEALPix viewers — slightly lighter than `HIST_LIMITS_LW`).
"""
const HIST_LIMITS_LW_HP = 1.0f0

"""Opacity of the cmin / cmax vlines in all histogram panels."""
const HIST_LIMITS_ALPHA = 0.65f0


############################
# §4  Crosshair rendering
############################
#
# The crosshair uses a two-layer technique: a thick dark halo drawn first,
# then a thin bright line on top.  This gives good contrast on both bright
# and dark image regions without a fixed background assumption.

"""Linewidth of the dark (black) halo layer of the crosshair."""
const CROSSHAIR_LW_DARK = 3.2f0

"""Linewidth of the bright (white) line layer of the crosshair."""
const CROSSHAIR_LW_LIGHT = 1.2f0

"""Opacity of the dark halo layer of the crosshair."""
const CROSSHAIR_ALPHA_DARK = 0.52f0

"""Opacity of the bright line layer of the crosshair."""
const CROSSHAIR_ALPHA_LIGHT = 0.92f0


############################
# §5  Zoom-box interaction
############################

"""Linewidth of the dashed zoom-rectangle outline."""
const ZOOM_BOX_LW = 1.2f0

"""Opacity of the dashed zoom-rectangle outline."""
const ZOOM_BOX_ALPHA = 0.42f0

"""Linewidth of the solid L-shaped corner accents on the zoom rectangle."""
const ZOOM_CORNER_LW = 2.8f0

"""Opacity of the corner accents on the zoom rectangle."""
const ZOOM_CORNER_ALPHA = 0.95f0

"""
Fraction of the rectangle's width / height used as the length of each
L-shaped corner accent.  A value of 0.18 means each arm of the L covers
18 % of the corresponding dimension.
"""
const ZOOM_BEZIER_FACTOR = 0.18f0


############################
# §6  Region-selection rendering
############################

"""
Linewidth for the region-selection rectangle / ellipse overlay
(2D image viewer and cube viewer).
"""
const REGION_LW = 2.4f0

"""
Linewidth for the region-selection overlay in HEALPix viewers.
Slightly thinner than `REGION_LW` to match the smaller default figure scale.
"""
const REGION_LW_HP = 2.3f0

"""Opacity of all region-selection overlays."""
const REGION_ALPHA = 0.55f0


############################
# §7  Contour rendering
############################

"""Contour linewidth in the 2D image viewer and cube viewer."""
const CONTOUR_LW = 1.2f0

"""Contour linewidth in HEALPix viewers (slightly lighter)."""
const CONTOUR_LW_HP = 1.1f0

"""
Alpha for the automatic black contour colour (used on bright images).
The auto-colour logic computes the median brightness of the visible slice
and picks black (dark) vs. white (light) contours accordingly.
"""
const CONTOUR_AUTO_DARK_ALPHA = 0.72f0

"""Alpha for the automatic white contour colour (used on dark images)."""
const CONTOUR_AUTO_LIGHT_ALPHA = 0.75f0

"""
Normalised brightness pivot [0, 1] above which black contours are chosen.
Below this threshold white contours are used instead.
"""
const CONTOUR_AUTO_BRIGHTNESS_THRESHOLD = 0.5f0


############################
# §8  Scatter markers
############################

"""
Markersize for the pixel-click marker in the 2D and cube viewers
(scatter plot, units are screen pixels).
"""
const MARKER_SIZE = 10

"""Markersize for the pixel-click marker in HEALPix viewers."""
const MARKER_SIZE_HP = 12


############################
# §9  Mollweide / HEALPix projection
############################

"""
Bounding box of the full Mollweide projection in normalised coordinates:
(x_min, x_max, y_min, y_max).  The Mollweide ellipse fits exactly inside
[-2, 2] × [-1, 1] in these units.

Usage:
    set_mollweide_view!(ax, MOLLWEIDE_BOUNDS...)
    refresh_graticule_labels!(g, ax; bounds = MOLLWEIDE_BOUNDS)
"""
const MOLLWEIDE_BOUNDS = (-2.0, 2.0, -1.0, 1.0)
