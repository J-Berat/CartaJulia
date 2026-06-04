# path: src/helpers/Helpers.jl
#
# Umbrella file for the MANTA helper library. The implementation has been
# split into thematic sub-files under src/helpers/; this file only declares
# the shared module-level dependencies, the public export surface and the
# include order.
#
# Sub-files (all included into the parent MANTA module):
#   - Images.jl        : NaN-aware filtering + RGB helpers
#   - Scaling.jl       : log / lin display transforms
#   - Stats.jl         : finite extrema, percentile contrast, histograms
#   - Contours.jl      : automatic / manual contour spec parsing
#   - Slicing.jl       : cube slicing, region masks, dual-view product
#   - Moments.jl       : spectral moment integrals (M0 = Σ y·Δx)
#   - WCS.jl           : SimpleWCSAxis / WCSTransform + pixel↔world
#   - FITSHeaders.jl   : header construction for exported FITS products
#   - UIBits.jl        : LaTeX, paths, figure size, parsers, settings I/O
#   - PowerSpectrum.jl : 2D / radial PS primitives shared by the cube viewer
#
# API stable: apply_scale, clamped_extrema, percentile_clims, histogram_counts,
#             histogram_profile, histogram_ylabel,
#             is_rgb_like, as_rgb_image, as_rgb_pixels, rgb_image,
#             nan_gaussian_filter,
#             automatic_contour_levels, parse_contour_levels,
#             parse_contour_specs, format_contour_specs, contour_color_values,
#             ijk_to_uv, uv_to_ijk, get_slice,
#             region_uv_indices, mean_region_spectrum,
#             dual_view_product, moments, moments_map, moment_map,
#             moment_vectors,
#             filtered_cube_by_slice,
#             make_info_tex, to_cmap, get_box_str, _pick_fig_size,
#             _axis_render_height,
#             latex_safe, make_main_title, make_slice_title, make_spec_title,
#             latex_tick, latex_tick_formatter,
#             parse_textbox, parse_manual_clims, parse_gif_request,
#             SimpleWCSAxis, read_simple_wcs, has_wcs, world_coord,
#             wcs_axis_label, format_world_coord, data_unit_label,
#             WCSTransform, read_wcs_transform, pixel_scale,
#             sky_world_coords, spectral_quantity, spectral_quantity_word,
#             sky_dims, spectral_dim,
#             save_viewer_settings, load_viewer_settings

############################
# Exports
############################
export apply_scale, apply_scale_display, clamped_extrema, percentile_clims, histogram_counts
export ASINH_SOFTENING_DEFAULT, scale_label_tex, scale_menu_options, cycle_scale_mode
export histogram_profile, histogram_ylabel
export is_rgb_like, as_rgb_image, as_rgb_pixels, rgb_image
export nan_gaussian_filter
export automatic_contour_levels, parse_contour_levels
export parse_contour_specs, format_contour_specs, contour_color_values
export ijk_to_uv, uv_to_ijk, get_slice, get_slice_view, get_slice_copy
export as_float32, parse_path_spec
export region_uv_indices, mean_region_spectrum
export dual_view_product, resample_matrix_linear, resample_cube_linear, reproject_cube_linear
export moments, moments_map, moment_map, moment_vectors, filtered_cube_by_slice
export make_info_tex
export MANTA_ICONS
export MANTA_COLORMAP_OPTIONS, ui_colormap_options
export ColormapSelector, colormap_selector!
export to_cmap, get_box_str, _pick_fig_size, _axis_render_height
export latex_safe, make_main_title, make_slice_title, make_spec_title
export latex_tick, latex_tick_formatter
export parse_textbox
export parse_manual_clims, parse_histogram_bins, parse_histogram_xlimits
export parse_histogram_ylimits, parse_spectrum_ylimits, parse_gif_request
export manta_recipe, copy_text_to_clipboard
export SimpleWCSAxis, read_simple_wcs, has_wcs, world_coord
export wcs_axis_label, format_world_coord, format_world_value, data_unit_label
export format_cursor_readout, cursor_world_strings
export WCSTransform, read_wcs_transform, pixel_scale, sky_world_coords
export spectral_quantity, spectral_quantity_word, sky_dims, spectral_dim
export wcs_sky_grids, nice_graticule_step, wcs_graticule_levels, draw_wcs_graticule!, set_wcs_graticule_visible!
export fits_header_display_lines
export fits_header_for_slice, fits_header_for_moment
export fits_header_for_region_spectrum, fits_header_for_filtered_cube
export save_viewer_settings, load_viewer_settings
export manta_defaults_path, load_manta_defaults
export power_spectrum_2d, power_spectrum_1d_radial, fit_loglog_slope
export ShortcutBinding, register_shortcuts!, format_shortcut, format_shortcut_help, shortcut_help_message
export open_shortcut_help_window
export zoom_limits, zoom_axis!, zoom_shortcut_bindings
# Errors (actionable, structured)
export MANTAError, FileNotFoundError, UnsupportedFormatError,
       HDUSelectionError, InvalidArgumentError, DatasetShapeError
export require_file, rethrow_actionable, invalid_kwarg, invalid_hdu
# Progress + cancellation
export CancelToken, cancel!, is_cancelled
export ProgressTracker, tick!, set_progress!, finish!, with_progress
# Display downsampling
export downsample_factor, downsample_block_mean, downsample_subsample, auto_downsample
# Undo / redo (`clear!` and `current` are intentionally NOT re-exported —
# fully qualified `MANTA.clear!` / `MANTA.current` to avoid Base collisions)
export UndoRedoStack, register_state!, undo!, redo!, with_suppression
# Plugin extension surface
export register_plugin!, unregister_plugin!, list_plugins, clear_plugins!
export plugin_load, plugin_view, run_postprocess!
# Backend selection (headless robustness)
export is_headless_env, pick_backend!, with_export_backend
# Drag-and-drop file loading (reload the dropped file in the current window)
export enable_file_drop!, supported_drop_path

############################
# Deps
############################
using Makie
using LaTeXStrings
using TOML
using Statistics: quantile, mean, median, std
using ImageFiltering
using FFTW: fft, fftshift
import GLFW

############################
# Implementation (split by theme)
############################
include("Images.jl")
include("Scaling.jl")
include("Stats.jl")
include("Contours.jl")
include("Slicing.jl")
include("Moments.jl")
include("WCS.jl")           # provides header_has / header_get reused below
include("FITSHeaders.jl")
include("UIBits.jl")
include("PowerSpectrum.jl")
include("Shortcuts.jl")
include("Errors.jl")
include("Progress.jl")
include("Downsample.jl")
include("UndoRedo.jl")
include("Plugins.jl")
include("Backend.jl")
# DragDrop.jl references `manta` / `forget!` (defined in MANTA.jl) only inside
# function bodies, so they resolve at call time — include order is fine here.
include("DragDrop.jl")
