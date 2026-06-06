# path: src/views/CubeView.jl
#
# 3D cube interactive viewer (slice + per-voxel/region spectrum + comparison
# + moments + power spectrum + FITS products + GIF export + WCS-aware ticks).
#
# This is the full cube viewer body, extracted verbatim from the inline
# definition that used to live in `MANTA.jl::manta(::String)`. The only
# changes versus the original are in the prologue: instead of receiving a
# filepath and reading the FITS itself, the function now takes a
# `CubeDataset` and recovers `filepath` / `header` from `ds.metadata` when
# available (loaders set `:fits_path` and may set `:fits_header`).
#
# Public entry point: `view_cube(ds; kwargs...)`. The internal name
# `_view_cube` is preserved for backwards compatibility with earlier drafts.
#
# Pure helpers used by this file live in sibling files included by
# MANTA.jl before this one:
#   - src/views/cube/PowerSpectrumBundle.jl  -> `_cube_ps_bundle`

"""
    view_cube(ds::CubeDataset; kwargs...) -> Figure

Open the full interactive cube viewer for a `CubeDataset`. Supported kwargs:
`cmap`, `vmin`, `vmax`, `invert`, `scale`, `asinh_softening`, `figsize`,
`save_dir`, `activate_gl`, `display_fig`, `settings_path`.

The viewer offers slice navigation, per-voxel/region spectra, an optional
3-D cube view, comparison overlay, moment maps (0/1/2), 2-D and 1-D power
spectra, FITS product export, PNG/PDF/CSV exports and GIF recording. When
the dataset was loaded from a FITS file, the viewer reuses that file's
directory for resolving comparison datasets and for export defaults.

Lifecycle:
The returned `Figure` is pinned in `MANTA._KEEP_ALIVE` immediately after
construction so that Julia's GC cannot collect the figure (or any of the
`Observable`s in its reactive graph) while the window is open.  When the
user closes the window, GLMakie sets `fig.scene.events.window_open = false`
and the internal listener calls `forget!(fig)`, dropping the strong
reference.  You do not need to store the return value unless you want
programmatic access to the figure after construction.
"""

# ---------------------------------------------------------------------------
# Undo/redo snapshot type for the cube viewer.
#
# One *fixed* NamedTuple type so every snapshot shares a single type regardless
# of the concrete `MaskSource` in play. This matters because `UndoRedoStack{T}`
# pins `T` from the first snapshot and `register_state!` dispatches on `::T`:
# if `mask_source` were stored at its concrete type, switching from
# `NoMaskSource` to a `ThresholdSource` would change the tuple type and make
# `register_state!` fail to dispatch. Declaring the field as the abstract
# `MaskSource` keeps the type stable across the whole history.
#
# Covered state (audited): navigation + colormap + contrast, PLUS the mask
# source, the region selection (uvs + drag points + mode) and the moment
# product. `region_shape` is deliberately NOT stored — it is fully derived from
# `selection_mode` (`:circle ⇒ :circle`, otherwise `:box`) at its single
# assignment site in `UICallbacksBundle`, so storing it would only risk an
# inconsistent intermediate snapshot.
const _CubeUndoSnapshot = @NamedTuple{
    axis::Int,
    idx::Int,
    compare_idx::Int,
    cmap_name::Symbol,
    invert_cmap::Bool,
    img_scale_mode::Symbol,
    use_manual::Bool,
    clims_manual::Tuple{Float32,Float32},
    mask_source::MaskSource,
    region_uvs::Vector{Tuple{Int,Int}},
    region_p0::Tuple{Float32,Float32},
    region_p1::Tuple{Float32,Float32},
    selection_mode::Symbol,
    view_product::Symbol,
    moment_order::Int,
}

# NaN-safe projection of a drag-point into the snapshot. The empty selection
# stores `Point2f(NaN, NaN)`, and `NaN != NaN` would defeat the
# `register_state!` dedup (every register would look like a change, so a single
# slider drag would flood the history again). Map any non-finite coordinate to
# `0` so equal empty-selection snapshots compare equal.
_undo_clean_pt(p) = (isfinite(p[1]) ? Float32(p[1]) : 0f0,
                     isfinite(p[2]) ? Float32(p[2]) : 0f0)

_cube_overview_progress(coord::Integer, n::Integer) =
    n <= 1 ? 0.5f0 : Float32(clamp(coord, 1, n) - 1) / Float32(n - 1)

function _cube_overview_geometry(siz::NTuple{3,<:Integer}, axis::Integer,
                                 idx::Integer, i::Integer, j::Integer, k::Integer)
    y_by_axis = (3f0, 2f0, 1f0)
    a = clamp(Int(axis), 1, 3)
    coords = ntuple(d -> d == a ? idx : (d == 1 ? i : d == 2 ? j : k), 3)
    progress = ntuple(d -> _cube_overview_progress(coords[d], siz[d]), 3)
    y = y_by_axis[a]
    p = progress[a]
    return (;
        marker_points = Point2f[
            Point2f(progress[1], y_by_axis[1]),
            Point2f(progress[2], y_by_axis[2]),
            Point2f(progress[3], y_by_axis[3]),
        ],
        active_progress = Point2f[Point2f(0f0, y), Point2f(p, y)],
        active_marker = Point2f[Point2f(p, y - 0.38f0), Point2f(p, y + 0.38f0)],
        active_axis = a,
        active_progress_value = p,
    )
end

_cube3d_identity_rotation() = Float32[
    1 0 0
    0 1 0
    0 0 1
]

function _cube3d_axis_angle_rotation(axis::NTuple{3,<:Real}, angle_deg::Real)
    ax = Float32(axis[1])
    ay = Float32(axis[2])
    az = Float32(axis[3])
    norm_axis = sqrt(ax * ax + ay * ay + az * az)
    norm_axis > eps(Float32) || throw(ArgumentError("rotation axis must be non-zero"))
    x = ax / norm_axis
    y = ay / norm_axis
    z = az / norm_axis
    θ = Float32(angle_deg) * Float32(pi / 180)
    c = cos(θ)
    s = sin(θ)
    t = 1f0 - c
    return Float32[
        t*x*x + c      t*x*y - s*z    t*x*z + s*y
        t*x*y + s*z    t*y*y + c      t*y*z - s*x
        t*x*z - s*y    t*y*z + s*x    t*z*z + c
    ]
end

function _cube3d_compose_rotation(current::AbstractMatrix{<:Real},
                                  axis::NTuple{3,<:Real}, angle_deg::Real)
    size(current) == (3, 3) || throw(ArgumentError("current rotation must be 3x3"))
    return _cube3d_axis_angle_rotation(axis, angle_deg) * Float32.(current)
end

_cube3d_apply_rotation(R::AbstractMatrix{<:Real}, p::Point3f) = Point3f(
    R[1, 1] * p[1] + R[1, 2] * p[2] + R[1, 3] * p[3],
    R[2, 1] * p[1] + R[2, 2] * p[2] + R[2, 3] * p[3],
    R[3, 1] * p[1] + R[3, 2] * p[2] + R[3, 3] * p[3],
)

function _cube3d_coord(dim::Integer, coord::Real, siz::NTuple{3,<:Integer})
    n = max(Int(siz[dim]), 1)
    maxdim = Float32(maximum(siz))
    return Float32(coord - (n + 1) / 2) / maxdim
end

function _cube3d_corner(siz::NTuple{3,<:Integer}, ix::Integer, iy::Integer, iz::Integer)
    return Point3f(
        _cube3d_coord(1, ix == 1 ? 0.5 : siz[1] + 0.5, siz),
        _cube3d_coord(2, iy == 1 ? 0.5 : siz[2] + 0.5, siz),
        _cube3d_coord(3, iz == 1 ? 0.5 : siz[3] + 0.5, siz),
    )
end

function _cube3d_voxel_center(siz::NTuple{3,<:Integer}, i::Integer, j::Integer, k::Integer)
    return Point3f(
        _cube3d_coord(1, clamp(i, 1, siz[1]), siz),
        _cube3d_coord(2, clamp(j, 1, siz[2]), siz),
        _cube3d_coord(3, clamp(k, 1, siz[3]), siz),
    )
end

function _cube3d_view_geometry(siz::NTuple{3,<:Integer}, axis::Integer,
                               idx::Integer, i::Integer, j::Integer, k::Integer,
                               rotation::AbstractMatrix{<:Real})
    size(rotation) == (3, 3) || throw(ArgumentError("rotation must be 3x3"))
    a = clamp(Int(axis), 1, 3)
    id = clamp(Int(idx), 1, siz[a])
    corners = Dict{NTuple{3,Int},Point3f}()
    for x in 1:2, y in 1:2, z in 1:2
        p = _cube3d_corner(siz, x, y, z)
        corners[(x, y, z)] = _cube3d_apply_rotation(rotation, p)
    end

    edges = NTuple{2,NTuple{3,Int}}[
        ((1, 1, 1), (2, 1, 1)), ((1, 2, 1), (2, 2, 1)),
        ((1, 1, 2), (2, 1, 2)), ((1, 2, 2), (2, 2, 2)),
        ((1, 1, 1), (1, 2, 1)), ((2, 1, 1), (2, 2, 1)),
        ((1, 1, 2), (1, 2, 2)), ((2, 1, 2), (2, 2, 2)),
        ((1, 1, 1), (1, 1, 2)), ((2, 1, 1), (2, 1, 2)),
        ((1, 2, 1), (1, 2, 2)), ((2, 2, 1), (2, 2, 2)),
    ]
    box_segments = Point3f[]
    for (p0, p1) in edges
        push!(box_segments, corners[p0], corners[p1])
    end

    plane_center = _cube3d_coord(a, id, siz)
    lo = ntuple(d -> _cube3d_coord(d, 0.5, siz), 3)
    hi = ntuple(d -> _cube3d_coord(d, siz[d] + 0.5, siz), 3)
    if a == 1
        raw_loop = Point3f[
            Point3f(plane_center, lo[2], lo[3]),
            Point3f(plane_center, hi[2], lo[3]),
            Point3f(plane_center, hi[2], hi[3]),
            Point3f(plane_center, lo[2], hi[3]),
            Point3f(plane_center, lo[2], lo[3]),
        ]
    elseif a == 2
        raw_loop = Point3f[
            Point3f(lo[1], plane_center, lo[3]),
            Point3f(hi[1], plane_center, lo[3]),
            Point3f(hi[1], plane_center, hi[3]),
            Point3f(lo[1], plane_center, hi[3]),
            Point3f(lo[1], plane_center, lo[3]),
        ]
    else
        raw_loop = Point3f[
            Point3f(lo[1], lo[2], plane_center),
            Point3f(hi[1], lo[2], plane_center),
            Point3f(hi[1], hi[2], plane_center),
            Point3f(lo[1], hi[2], plane_center),
            Point3f(lo[1], lo[2], plane_center),
        ]
    end
    slice_loop = [_cube3d_apply_rotation(rotation, p) for p in raw_loop]
    selected_point = Point3f[
        _cube3d_apply_rotation(rotation, _cube3d_voxel_center(siz, i, j, k)),
    ]
    axis_segments = Point3f[
        _cube3d_apply_rotation(rotation, Point3f(-0.62, 0, 0)),
        _cube3d_apply_rotation(rotation, Point3f(0.62, 0, 0)),
        _cube3d_apply_rotation(rotation, Point3f(0, -0.62, 0)),
        _cube3d_apply_rotation(rotation, Point3f(0, 0.62, 0)),
        _cube3d_apply_rotation(rotation, Point3f(0, 0, -0.62)),
        _cube3d_apply_rotation(rotation, Point3f(0, 0, 0.62)),
    ]
    return (; box_segments, slice_loop, selected_point, axis_segments)
end

function _cube_rotated_projection(data, siz::NTuple{3,<:Integer}, projection_axis::Integer,
                                  rotation_axis::NTuple{3,<:Real}, angle_deg::Real,
                                  mode::Symbol)
    a = clamp(Int(projection_axis), 1, 3)
    mode in (:mean, :sum, :max) ||
        throw(ArgumentError("rotation projection mode must be :mean, :sum, or :max"))
    Rinv = transpose(_cube3d_axis_angle_rotation(rotation_axis, angle_deg))
    center = (
        Float32((siz[1] + 1) / 2),
        Float32((siz[2] + 1) / 2),
        Float32((siz[3] + 1) / 2),
    )
    u_dim, v_dim = a == 1 ? (2, 3) : a == 2 ? (1, 3) : (1, 2)
    nu, nv = siz[u_dim], siz[v_dim]
    out = Matrix{Float32}(undef, nu, nv)

    @inline function rotated_source_coord(q1::Float32, q2::Float32, q3::Float32)
        x = q1 - center[1]
        y = q2 - center[2]
        z = q3 - center[3]
        return (
            center[1] + Rinv[1, 1] * x + Rinv[1, 2] * y + Rinv[1, 3] * z,
            center[2] + Rinv[2, 1] * x + Rinv[2, 2] * y + Rinv[2, 3] * z,
            center[3] + Rinv[3, 1] * x + Rinv[3, 2] * y + Rinv[3, 3] * z,
        )
    end

    @inline function sample_trilinear(x::Float32, y::Float32, z::Float32)
        if !(1f0 <= x <= Float32(siz[1]) &&
             1f0 <= y <= Float32(siz[2]) &&
             1f0 <= z <= Float32(siz[3]))
            return NaN32
        end
        x0 = clamp(floor(Int, x), 1, siz[1])
        y0 = clamp(floor(Int, y), 1, siz[2])
        z0 = clamp(floor(Int, z), 1, siz[3])
        x1 = min(x0 + 1, siz[1])
        y1 = min(y0 + 1, siz[2])
        z1 = min(z0 + 1, siz[3])
        tx = x - Float32(x0)
        ty = y - Float32(y0)
        tz = z - Float32(z0)

        c000 = Float32(data[x0, y0, z0]); c100 = Float32(data[x1, y0, z0])
        c010 = Float32(data[x0, y1, z0]); c110 = Float32(data[x1, y1, z0])
        c001 = Float32(data[x0, y0, z1]); c101 = Float32(data[x1, y0, z1])
        c011 = Float32(data[x0, y1, z1]); c111 = Float32(data[x1, y1, z1])
        vals = (c000, c100, c010, c110, c001, c101, c011, c111)
        all(isfinite, vals) || return NaN32

        c00 = c000 * (1f0 - tx) + c100 * tx
        c10 = c010 * (1f0 - tx) + c110 * tx
        c01 = c001 * (1f0 - tx) + c101 * tx
        c11 = c011 * (1f0 - tx) + c111 * tx
        c0 = c00 * (1f0 - ty) + c10 * ty
        c1 = c01 * (1f0 - ty) + c11 * ty
        return c0 * (1f0 - tz) + c1 * tz
    end

    @inbounds for u in 1:nu, v in 1:nv
        acc = 0f0
        n = 0
        best = -Inf32
        for d in 1:siz[a]
            q = a == 1 ? (Float32(d), Float32(u), Float32(v)) :
                a == 2 ? (Float32(u), Float32(d), Float32(v)) :
                         (Float32(u), Float32(v), Float32(d))
            x, y, z = rotated_source_coord(q[1], q[2], q[3])
            val = sample_trilinear(x, y, z)
            isfinite(val) || continue
            if mode === :max
                best = max(best, val)
            else
                acc += val
            end
            n += 1
        end
        out[u, v] = n == 0 ? NaN32 :
                    mode === :mean ? acc / Float32(n) :
                    mode === :sum ? acc :
                    best
    end
    return out
end

# Build a `_CubeUndoSnapshot`. We construct via the positional type constructor
# (`_CubeUndoSnapshot((vals...))`) rather than a NamedTuple literal so the field
# types come from the type parameter, not from the runtime values. A literal
# `(; mask_source = NoMaskSource(), ...)` would have the *narrow* field type
# `NoMaskSource`; a return-type annotation would NOT widen it, because that
# narrow tuple is already a subtype of `_CubeUndoSnapshot` (Tuple covariance),
# so `convert` falls back to identity. The positional constructor instead yields
# the exact `_CubeUndoSnapshot` type, keeping `mask_source` at `MaskSource` and
# every snapshot's type identical — which `register_state!{T}` relies on.
# The value order below MUST match the field order in `_CubeUndoSnapshot`.
function _cube_undo_snapshot(;
    axis, idx, compare_idx, cmap_name, invert_cmap, img_scale_mode,
    use_manual, clims_manual, mask_source, region_uvs, region_p0, region_p1,
    selection_mode, view_product, moment_order,
)
    return _CubeUndoSnapshot((
        axis, idx, compare_idx, cmap_name, invert_cmap, img_scale_mode,
        use_manual, clims_manual, mask_source, region_uvs, region_p0, region_p1,
        selection_mode, view_product, moment_order,
    ))
end

function _view_cube(
    ds::CubeDataset;
    cmap::Symbol = :viridis,
    vmin = nothing,
    vmax = nothing,
    invert::Bool = false,
    scale::Symbol = :lin,
    asinh_softening::Real = ASINH_SOFTENING_DEFAULT,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    settings_path::Union{Nothing,AbstractString} = nothing,
    hist_mode::Symbol = :bars,
    hist_bins::Int = HIST_BINS_DEFAULT,
    hist_xlimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    hist_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    spec_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    moment_threshold::Real = 0.0,
    moment_nsigma::Union{Nothing,Real} = nothing,
    moment_channels::Union{Nothing,AbstractVector{<:Integer}} = nothing,
    compare::Union{Nothing,AbstractString} = nothing,
    state = nothing,
)
    data = as_float32(ds.data)
    siz  = size(data)
    wcs  = ds.wcs
    # Full linear+CD transform when the loader supplied one (FITS); enables
    # exact celestial cursor coordinates, with graceful fallback to per-axis
    # linear world_coord when absent (in-memory cubes, HDF5).
    _wcs_xform = get(ds.metadata, :wcs_transform, nothing)
    wcs_xform  = _wcs_xform isa WCSTransform ? _wcs_xform : nothing
    unit_label = ds.unit_label
    unit_label_tex = latexstring("\\text{", latex_safe(unit_label), "}")

    slice_dims(axis::Integer) = if axis == 1
        (siz[2], siz[3])  # (y, z)
    elseif axis == 2
        (siz[1], siz[3])  # (x, z)
    else
        (siz[1], siz[2])  # (x, y)
    end

    slice_axis_dims(axis::Integer) = if axis == 1
        (2, 3)  # u=y, v=z
    elseif axis == 2
        (1, 3)  # u=x, v=z
    else
        (1, 2)  # u=x, v=y
    end

    pixel_axis_name(dim::Integer) = dim == 1 ? "pixel x" : dim == 2 ? "pixel y" : "pixel z"
    dataset_pixel_axis_label(dim::Integer) = begin
        label = 1 <= dim <= length(ds.axis_labels) ? strip(String(ds.axis_labels[dim])) : ""
        isempty(label) && (label = pixel_axis_name(dim))
        occursin(r"\[[^\]]+\]\s*$", label) ? label : "$(label) [pix]"
    end

    slice_axis_labels(axis::Integer) = begin
        u_dim, v_dim = slice_axis_dims(axis)
        (
            wcs_axis_label(wcs, v_dim; fallback = dataset_pixel_axis_label(v_dim)),
            wcs_axis_label(wcs, u_dim; fallback = dataset_pixel_axis_label(u_dim)),
        )
    end

    has_named_world_axis(dim::Integer) =
        has_wcs(wcs, dim) && !isempty(strip(String(wcs[dim].ctype_base)))

    pixel_world_tick_formatter(dim::Integer) = vals -> [
        has_named_world_axis(dim) ? latex_tick(world_coord(wcs, dim, v)) : latex_tick(v)
        for v in vals
    ]

    spectral_coords(dim::Integer) = Float32[
        has_named_world_axis(dim) ? Float32(world_coord(wcs, dim, chan)) : Float32(chan)
        for chan in 1:siz[dim]
    ]

    # Source identification. `filepath` is "" for in-memory cubes; callers
    # that loaded the cube from disk get the original FITS path back through
    # `ds.metadata[:fits_path]`. The same is true for `:fits_header`.
    fits_path  = get(ds.metadata, :fits_path, nothing)
    filepath   = fits_path isa AbstractString ? String(fits_path) : ""
    fname_full = filepath != "" ? basename(filepath) : String(ds.source_id)
    fname      = String(replace(fname_full, r"\.fits(\.gz)?$"i => ""))
    header     = get(ds.metadata, :fits_header, nothing)
    compare_header_ref = Ref{Any}(nothing)

    @info "Cube ready" source=ds.source_id size=siz

    # ---------- State ----------
    st = build_cube_view_state(;
        cmap, invert, vmin, vmax, scale,
        hist_mode, hist_bins, hist_xlimits, hist_ylimits, spec_ylimits,
    )
    # Aliases — every existing reference in the function body keeps working as-is.
    axis                      = st.axis
    idx                       = st.idx
    compare_idx               = st.compare_idx
    i_idx                     = st.i_idx
    j_idx                     = st.j_idx
    k_idx                     = st.k_idx
    u_idx                     = st.u_idx
    v_idx                     = st.v_idx
    cmap_name                 = st.cmap_name
    invert_cmap               = st.invert_cmap
    cm_obs = lift(cmap_name, invert_cmap) do name, inv
        base = to_cmap(name); inv ? reverse(base) : base
    end
    ps_cmap_name = Observable(cmap_name[])
    ps_cm_obs = lift(ps_cmap_name) do name
        to_cmap(name)
    end
    img_scale_mode            = st.img_scale_mode
    spec_scale_mode           = st.spec_scale_mode
    compare_data              = st.compare_data
    compare_visible           = st.compare_visible
    compare_name              = st.compare_name
    compare_path_current      = st.compare_path_current
    compare_mode              = st.compare_mode
    view_product              = st.view_product
    moment_order              = st.moment_order
    rotation_axis_obs         = st.rotation_axis
    rotation_angle_obs        = st.rotation_angle
    rotation_projection_mode  = st.rotation_projection_mode
    layout_mode               = st.layout_mode
    control_mode              = st.control_mode
    focus_image               = Observable(false)
    anim_playing              = st.anim_playing
    gauss_on                  = st.gauss_on
    sigma                     = st.sigma
    show_crosshair            = st.show_crosshair
    show_marker               = st.show_marker
    show_grid                 = st.show_grid
    show_contours             = st.show_contours
    contour_use_manual        = st.contour_use_manual
    contour_manual_levels     = st.contour_manual_levels
    contour_manual_colors     = st.contour_manual_colors
    selection_mode            = st.selection_mode
    region_shape              = st.region_shape
    region_uvs                = st.region_uvs
    region_start              = st.region_start
    region_end                = st.region_end
    region_drag_active        = st.region_drag_active
    zoom_drag_active          = st.zoom_drag_active
    zoom_drag_start           = st.zoom_drag_start
    zoom_drag_end             = st.zoom_drag_end
    use_manual                = st.use_manual
    clims_manual              = st.clims_manual
    hist_mode_obs             = st.hist_mode_obs
    hist_bins_obs             = st.hist_bins_obs
    hist_xlimits_manual       = st.hist_xlimits_manual
    hist_xlimits_manual_value = st.hist_xlimits_manual_value
    hist_ylimits_manual       = st.hist_ylimits_manual
    hist_ylimits_manual_value = st.hist_ylimits_manual_value
    spec_ylimits_value        = st.spec_ylimits_value
    spec_ylimits_source       = st.spec_ylimits_source
    ui_status                 = st.ui_status
    ps_layout_status          = st.ps_layout_status
    # ---------- Mask state ----------
    mask_source_obs           = st.mask_source_obs
    mask_bits_obs             = st.mask_bits_obs
    mask_status_obs           = st.mask_status_obs

    ui_theme = current_ui_theme()
    ui_theme_ref = Ref(ui_theme)
    ui_accent = ui_theme.accent
    ui_accent_strong = ui_theme.accent_strong
    ui_surface = ui_theme.surface
    ui_panel = ui_theme.panel
    ui_panel_header = ui_theme.panel_header
    ui_border = ui_theme.border
    ui_text = ui_theme.text
    ui_text_muted = ui_theme.text_muted
    ui_selection = ui_theme.selection
    ui_compare = ui_theme.compare
    ui_success = ui_theme.success
    ui_mask = ui_theme.mask
    ui_error = ui_theme.error
    fig_bg = ui_theme.background

    style_checkbox!(chk) = manta_style_checkbox!(chk, ui_theme_ref[]; compact = compact_layout)
    style_slider!(sl) = manta_style_slider!(sl, ui_theme_ref[]; compact = compact_layout)
    style_button!(btn)         = manta_style_button!(btn, ui_theme_ref[]; compact = compact_layout)
    style_button_primary!(btn) = manta_style_button_primary!(btn, ui_theme_ref[]; compact = compact_layout)
    style_button_ghost!(btn)   = manta_style_button_ghost!(btn, ui_theme_ref[]; compact = compact_layout)
    style_segmented_button!(btn; active::Bool = false) = manta_style_segmented_button!(btn, ui_theme_ref[]; compact = compact_layout, active = active)
    style_menu!(menu) = manta_style_menu!(menu, ui_theme_ref[]; compact = compact_layout)
    style_textbox!(tb) = manta_style_textbox!(tb, ui_theme_ref[]; compact = compact_layout)

    # ---------- Slice + moment data pipeline ----------
    # See src/views/cube/SlicePipelineBundle.jl
    (; slice_raw, compare_slice_raw, base_slice_proc, compare_slice_proc,
       compare_product_proc, view_raw, slice_proc, slice_disp, compare_slice_disp,
       plot_stride, plot_x, plot_y, slice_plot, compare_slice_plot,
       mask_slice, moment_raw, _moment_cache, _get_gauss_kernel, _gauss_kernel_cache) =
        _cube_slice_pipeline_bundle(st;
            data, siz, wcs, asinh_softening,
            moment_threshold, moment_nsigma, moment_channels,
        )

    # ---------- Async slice prefetch (lazy cubes only) ----------
    # why: warm read_slice!'s cache for the neighbouring channel in the current
    # scroll direction, so navigating a memory-mapped cube doesn't pay the full
    # disk-read latency on every slice change. The slice_raw lift above is
    # registered on `idx` first, so by the time this callback runs the *current*
    # slice has already been read into the cache; we only speculate on the next.
    # This is a no-op for in-memory cubes — `prefetch_adjacent!` only dispatches
    # on lazy FITS/HDF5 cubes — so the eager path keeps zero overhead.
    if data isa Union{LazyFITSCube,LazyHDF5Cube}
        _prefetch_last_idx = Ref(idx[])
        on(idx) do id
            _prefetch_last_idx[] = prefetch_adjacent!(data, axis[], id, _prefetch_last_idx[])
        end
        on(axis) do _
            # Direction tracking across a change of slicing axis is meaningless;
            # just resync the tracker to the current index (no prefetch issued —
            # the first read on the new axis is necessarily a cold read).
            _prefetch_last_idx[] = idx[]
        end
    end

    # Histogram input: when a mask is active, NaN-out the rejected voxels so
    # `histogram_profile` (which is binning a single 2-D matrix) excludes
    # them naturally — its internal `histogram_counts` already ignores
    # non-finite samples.
    hist_slice_obs = lift(slice_disp, mask_slice) do s, ms
        ms === nothing && return s
        out = Matrix{Float32}(undef, size(s))
        @inbounds for i in eachindex(s, out, ms)
            out[i] = ms[i] ? Float32(s[i]) : Float32(NaN)
        end
        out
    end

    clims_auto = lift(slice_disp) do s
        clamped_extrema(s)
    end

    contour_auto_levels = lift(slice_disp) do s
        automatic_contour_levels(s; n = CONTOUR_N_LEVELS_DEFAULT)
    end

    contour_levels_obs = lift(contour_use_manual, contour_manual_levels, contour_auto_levels) do use_man, manual, auto
        use_man && !isempty(manual) ? manual : auto
    end
    # Auto-contrasted contour colour: white on dark images, black on bright ones.
    contour_default_color_obs = lift(slice_disp) do img
        auto_contour_color(img; fallback = :light)
    end
    contour_colors_obs = lift(contour_levels_obs, contour_use_manual, contour_manual_colors,
                              contour_default_color_obs) do levels, use_man, colors, def_color
        contour_color_values(use_man ? colors : String[], length(levels), def_color)
    end

    # ---------- Undo / redo history ----------
    # Snapshot type: a fixed `_CubeUndoSnapshot` (see top of this file). The
    # stack is bounded (capacity=64) and deduplicates identical consecutive
    # snapshots, so rapid slider drags don't flood the history. Beyond the
    # navigation/contrast bag, the snapshot also captures the mask source, the
    # region selection and the moment product so those edits are undoable too.
    _nav_snapshot() = _cube_undo_snapshot(;
        axis           = axis[],
        idx            = idx[],
        compare_idx    = compare_idx[],
        cmap_name      = cmap_name[],
        invert_cmap    = invert_cmap[],
        img_scale_mode = img_scale_mode[],
        use_manual     = use_manual[],
        clims_manual   = clims_manual[],
        mask_source    = mask_source_obs[],
        region_uvs     = copy(region_uvs[]),
        region_p0      = _undo_clean_pt(region_start[]),
        region_p1      = _undo_clean_pt(region_end[]),
        selection_mode = selection_mode[],
        view_product   = view_product[],
        moment_order   = moment_order[],
    )
    _undo_stack = UndoRedoStack(_nav_snapshot(); capacity = UNDO_STACK_CAPACITY)

    clims_obs = lift(use_manual, clims_auto, clims_manual) do um, ca, cm
        um ? cm : ca
    end

    # safe clims for plotting/layout
    clims_safe = lift(clims_obs) do (cmin, cmax)
        if !(isfinite(cmin) && isfinite(cmax)) || isnan(cmin) || isnan(cmax) || cmin == cmax
            (0f0, 1f0)
        else
            (cmin, cmax)
        end
    end

    hist_limits_obs = lift(hist_xlimits_manual, hist_xlimits_manual_value, clims_safe) do manual, xlim, clim
        manual ? xlim : clim
    end

    _hist_cache = Dict{Tuple{UInt64,Int,Tuple{Float32,Float32},Symbol}, Any}()
    function _cached_histogram_profile(s, lim, bins, mode)
        key = (
            hash(s),
            Int(bins),
            (Float32(first(lim)), Float32(last(lim))),
            normalize_histogram_mode(mode),
        )
        get!(_hist_cache, key) do
            if length(_hist_cache) >= 64
                empty!(_hist_cache)
            end
            histogram_profile(s; bins = bins, limits = lim, mode = mode)
        end
    end

    hist_pair_obs = lift(hist_slice_obs, hist_limits_obs, hist_bins_obs, hist_mode_obs) do s, lim, bins, mode
        _cached_histogram_profile(s, lim, bins, mode)
    end
    hist_x_obs = lift(p -> p.x, hist_pair_obs)
    hist_y_obs = lift(p -> p.y, hist_pair_obs)
    hist_width_obs = lift(p -> p.width, hist_pair_obs)
    hist_bars_visible = lift(m -> m === :bars, hist_mode_obs)
    hist_kde_visible = lift(m -> m === :kde, hist_mode_obs)
    hist_ylabel_obs = lift(histogram_ylabel, hist_mode_obs)

    compare_hist_pair_obs = lift(compare_slice_proc, img_scale_mode, compare_visible, hist_limits_obs, hist_bins_obs, hist_mode_obs) do s, scale_mode, visible, lim, bins, hist_mode_
        visible || return (Float32[], Float32[])
        A = apply_scale(s, scale_mode; asinh_softening = asinh_softening)
        out = similar(A, Float32)
        @inbounds for i in eachindex(A)
            x = A[i]
            out[i] = isfinite(x) ? Float32(x) : 0f0
        end
        profile = _cached_histogram_profile(out, lim, bins, hist_mode_)
        return (profile.x, profile.y)
    end
    compare_hist_x_obs = lift(p -> p[1], compare_hist_pair_obs)
    compare_hist_y_obs = lift(p -> p[2], compare_hist_pair_obs)

    compare_clims_safe = lift(compare_slice_disp, compare_mode, clims_safe) do s, mode, lim
        if mode in (:A, :B)
            lim
        else
            clamped_extrema(s)
        end
    end

    # Lightweight image shown while the slice slider is being dragged.
    # The full reactive pipeline still owns `slice_plot`; this observable is
    # only a temporary front-buffer so navigation can feel immediate without
    # recomputing histogram/spectrum/contours/clims for every mouse move.
    slice_plot_preview = Observable{Union{Nothing,NamedTuple{(:x,:y,:z),Tuple{Vector{Float32},Vector{Float32},Matrix{Float32}}}}}(nothing)
    plot_x_live = lift(plot_x, slice_plot_preview) do full, preview
        preview === nothing ? full : preview.x
    end
    plot_y_live = lift(plot_y, slice_plot_preview) do full, preview
        preview === nothing ? full : preview.y
    end
    slice_plot_live = lift(slice_plot, slice_plot_preview) do full, preview
        preview === nothing ? full : preview.z
    end
    on(slice_plot) do _
        slice_plot_preview[] === nothing || (slice_plot_preview[] = nothing)
    end

    spec_x_axes = (collect(0:(siz[1] - 1)), collect(0:(siz[2] - 1)), collect(0:(siz[3] - 1)))
    spec_y_buf  = Vector{Float32}(undef, siz[3])
    @views copyto!(spec_y_buf, data[1, 1, :])
    spec_x_raw  = Observable(spec_x_axes[3])
    spec_y_raw  = Observable(spec_y_buf)
    spec_y_disp = lift(spec_y_raw, spec_scale_mode) do y, m
        apply_scale(y, m; asinh_softening = asinh_softening)
    end
    # Second-cube spectrum buffer / Observables — same x grid as cube A
    # (same region selection), filled by `refresh_spectrum!` only when a
    # comparison cube is loaded. NaN-init so the (hidden) line shows
    # nothing before the first refresh.
    spec_y_compare_buf = Vector{Float32}(undef, siz[3])
    fill!(spec_y_compare_buf, NaN32)
    spec_y_compare_raw = Observable(spec_y_compare_buf)
    spec_y_compare_disp = lift(spec_y_compare_raw, spec_scale_mode) do y, m
        apply_scale(y, m; asinh_softening = asinh_softening)
    end
    # ---------- Figure & layout ----------
    pick_backend!(activate_gl)
    fig_size = _pick_fig_size(figsize)
    compact_layout = fig_size[1] <= COMPACT_LAYOUT_W || fig_size[2] <= COMPACT_LAYOUT_H
    roomy_compact_layout = compact_layout && fig_size[1] >= 1400 && fig_size[2] >= 840
    spec_axis_width = compact_layout ? (roomy_compact_layout ? 720 : 600) : 600
    spec_axis_height = compact_layout ? (roomy_compact_layout ? 230 : 185) : 320
    hist_axis_height = compact_layout ? (roomy_compact_layout ? 76 : 60) : 105
    ps_header_height = compact_layout ? 0 : 76
    ps_axis_size = compact_layout ? (roomy_compact_layout ? 420 : 320) : 620
    controls_row_heights = compact_layout ?
        (roomy_compact_layout ? (180, 130, 36) : (180, 164, 42)) :
        (188, 146, 46)
    controls_gap = compact_layout ? 8 : 16
    controls_height = sum(controls_row_heights) + 2 * controls_gap
    card_pad = compact_layout ? 9 : 12
    card_gap = compact_layout ? 7 : 10
    main_row_gap = compact_layout ? 8 : 14
    plot_row_height = compact_layout ? max(ps_axis_size + 80, fig_size[2] - controls_height - 2 * main_row_gap) : 0
    # Height reserved for row 1 when stacking the heatmap + PSD in :power_spectrum mode.
    ps_plot_row_height = max(360, fig_size[2] - controls_height - 2 * main_row_gap - 16)

    fig = Figure(size = fig_size, backgroundcolor = fig_bg)

    main_grid = fig[1, 1] = GridLayout()
    colgap!(main_grid, 18)
    rowgap!(main_grid, main_row_gap)
    # Image + contrast scale
    # valign=:center: vertically centres the image+colorbar block in its
    # main_grid cell. Without this, row 1 is as tall as spec_grid
    # (spectrum + histogram + info ≈ 550 px) and the dead space accumulates
    # above the image for non-square maps (DataAspect).
    # NOTE: tellwidth=false is intentionally NOT set here — CompareBundle
    # dynamically resizes column 2 of img_grid and must be able to propagate
    # the total width up to main_grid col 1.
    img_grid  = main_grid[1, 1] = GridLayout(; valign = :top)
    colgap!(img_grid, compact_layout ? 8 : 14)
    rowgap!(img_grid, compact_layout ? 6 : 8)
    img_a_grid = img_grid[1, 1] = GridLayout()
    img_cmp_grid = img_grid[1, 2] = GridLayout()
    colgap!(img_a_grid, -8)
    colgap!(img_cmp_grid, -8)
    rowgap!(img_a_grid, 3)
    rowgap!(img_cmp_grid, 3)
    rowsize!(img_a_grid, 1, Fixed(0))
    rowsize!(img_cmp_grid, 1, Fixed(0))

    xlab0, ylab0 = slice_axis_labels(axis[])
    main_title_obs = lift(view_product, moment_order, rotation_projection_mode, rotation_angle_obs, rotation_axis_obs) do product, order, rmode, rangle, raxis
        if product === :moment
            latexstring("\\text{", latex_safe(fname), " ", latex_safe(order == 0 ? "moment 0" : order == 1 ? "moment 1" : "moment 2"), "}")
        elseif product === :rotproj
            label = "rotated $(String(rmode)) projection, $(round(Float32(rangle); digits = 2))° around ($(round(raxis[1]; digits = 2)), $(round(raxis[2]; digits = 2)), $(round(raxis[3]; digits = 2)))"
            latexstring("\\text{", latex_safe(fname), " ", latex_safe(label), "}")
        else
            make_main_title(fname)
        end
    end
    # Moment captions follow the spectral quantity along the integrated axis:
    # FREQ → "mean frequency / frequency dispersion", WAVE → "wavelength",
    # VRAD/VOPT → "velocity". Falls back to "value" when no WCS classifies
    # the axis (channel-only cubes).
    spectral_word_for(a::Integer) =
        spectral_quantity_word(spectral_quantity(wcs, a))
    moment_unit_for(a::Integer) = begin
        has_wcs(wcs, a) && !isempty(wcs[a].cunit) ? " [" * wcs[a].cunit * "]" : ""
    end
    integrated_intensity_label(a::Integer) = begin
        u_ax = moment_unit_for(a)
        u = isempty(u_ax) ? unit_label :
            (unit_label == "value" ? strip(u_ax, [' ', '[', ']']) :
             unit_label * "·" * strip(u_ax, [' ', '[', ']']))
        "integrated intensity [" * String(u) * "]"
    end
    display_unit_label = lift(view_product, moment_order, axis, rotation_projection_mode) do product, order, a, rmode
        if product === :rotproj
            txt = rmode === :sum ? "projected sum [" * unit_label * "]" :
                  rmode === :max ? "projected max [" * unit_label * "]" :
                                   unit_label
            return latexstring("\\text{", latex_safe(txt), "}")
        end
        if product !== :moment
            return unit_label_tex
        end
        word = spectral_word_for(a)
        text = order == 0 ? integrated_intensity_label(a) :
               order == 1 ? "mean " * word * moment_unit_for(a) :
                            word * " dispersion" * moment_unit_for(a)
        latexstring("\\text{", latex_safe(text), "}")
    end
    ax_img = Axis(
        img_a_grid[2, 1];
        title     = main_title_obs,
        xlabel    = xlab0,
        ylabel    = ylab0,
        aspect    = DataAspect(),
        xtickformat = pixel_world_tick_formatter(slice_axis_dims(axis[])[2]),
        ytickformat = pixel_world_tick_formatter(slice_axis_dims(axis[])[1]),
    )
    compare_mode_label(mode::Symbol) = mode === :A ? "A" :
        mode === :B ? "B" :
        mode === :diff ? "A - B" :
        mode === :ratio ? "A / B" :
        "normalized residuals"
    compare_mode_chip_label(mode::Symbol) = mode === :A ? "A" :
        mode === :B ? "B" :
        mode === :diff ? "A-B" :
        mode === :ratio ? "A/B" :
        "resid z"
    compare_mode_chip_color(mode::Symbol) = mode === :A ? ui_accent :
        mode === :B ? ui_compare :
        mode === :diff ? ui_selection :
        mode === :ratio ? ui_success :
        ui_mask
    chip_text_color(c::RGBf) = (0.2126f0 * c.r + 0.7152f0 * c.g + 0.0722f0 * c.b) > 0.62f0 ?
        RGBf(0.08, 0.10, 0.14) : RGBf(0.98, 0.98, 0.96)
    compare_chip_text_obs = lift(compare_mode) do mode
        latexstring("\\textbf{", latex_safe(compare_mode_chip_label(mode)), "}")
    end
    compare_chip_color_obs = lift(compare_mode) do mode
        compare_mode_chip_color(mode)
    end
    compare_chip_text_color_obs = lift(compare_chip_color_obs) do color
        chip_text_color(color)
    end
    chip_height = compact_layout ? 20 : 22
    chip_width = compact_layout ? 54 : 62
    chip_fontsize = compact_layout ? 12 : 13
    chip_padding = compact_layout ? (6, 6, 2, 2) : (7, 7, 3, 3)
    chip_row_height = chip_height + 2
    chip_a_box = Box(
        img_a_grid[1, 1];
        color = ui_accent,
        strokecolor = ui_accent_strong,
        strokewidth = 0.8,
        cornerradius = 6,
        width = chip_width,
        height = chip_height,
        halign = :left,
        valign = :center,
        visible = compare_visible,
    )
    chip_a_label = Label(
        img_a_grid[1, 1];
        text = latexstring("\\textbf{A}"),
        color = chip_text_color(ui_accent),
        fontsize = chip_fontsize,
        padding = chip_padding,
        halign = :left,
        valign = :center,
        tellwidth = false,
        tellheight = false,
        visible = compare_visible,
    )
    chip_cmp_box = Box(
        img_cmp_grid[1, 1];
        color = compare_chip_color_obs,
        strokecolor = compare_chip_color_obs,
        strokewidth = 0.8,
        cornerradius = 6,
        width = chip_width,
        height = chip_height,
        halign = :left,
        valign = :center,
        visible = compare_visible,
    )
    chip_cmp_label = Label(
        img_cmp_grid[1, 1];
        text = compare_chip_text_obs,
        color = compare_chip_text_color_obs,
        fontsize = chip_fontsize,
        padding = chip_padding,
        halign = :left,
        valign = :center,
        tellwidth = false,
        tellheight = false,
        visible = compare_visible,
    )

    compare_title_obs = lift(compare_name, compare_mode) do name, mode
        label = compare_mode_label(mode)
        isempty(name) ? latexstring("\\text{", latex_safe(label), "}") :
            latexstring("\\text{", latex_safe(label), ": ", latex_safe(name), "}")
    end
    ax_cmp = Axis(
        img_cmp_grid[2, 1];
        title     = compare_title_obs,
        xlabel    = xlab0,
        ylabel    = ylab0,
        aspect    = DataAspect(),
        xtickformat = pixel_world_tick_formatter(slice_axis_dims(axis[])[2]),
        ytickformat = pixel_world_tick_formatter(slice_axis_dims(axis[])[1]),
    )
    if compact_layout
        ax_img.width[] = ps_axis_size
        ax_img.height[] = ps_axis_size
        ax_cmp.width[] = ps_axis_size
        ax_cmp.height[] = ps_axis_size
    end
    normal_axis_size = (;
        img_w = ax_img.width[],
        img_h = ax_img.height[],
        cmp_w = ax_cmp.width[],
        cmp_h = ax_cmp.height[],
    )
    colsize!(img_grid, 2, Fixed(0))

    uv_point = Observable(Point2f(1, 1))
    hm = heatmap!(ax_img, plot_x_live, plot_y_live, slice_plot_live; colormap = cm_obs, colorrange = clims_safe)
    heatmap!(ax_cmp, plot_x, plot_y, compare_slice_plot; colormap = cm_obs, colorrange = compare_clims_safe, visible = compare_visible)
    contour!(ax_img, plot_x, plot_y, slice_plot; levels = contour_levels_obs, color = contour_colors_obs, linewidth = CONTOUR_LW, visible = show_contours)
    compare_contours_visible = lift(show_contours, compare_visible) do contours, visible
        contours && visible
    end
    contour!(ax_cmp, plot_x, plot_y, compare_slice_plot; levels = contour_levels_obs, color = contour_colors_obs, linewidth = CONTOUR_LW, visible = compare_contours_visible)
    crosshair_segments = lift(axis, u_idx, v_idx, show_crosshair) do a, u, v, enabled
        enabled || return Point2f[]
        u_max, v_max = slice_dims(a)
        Point2f[
            Point2f(1, u), Point2f(v_max, u),
            Point2f(v, 1), Point2f(v, u_max),
        ]
    end
    zoom_box_segments = lift(zoom_drag_active, zoom_drag_start, zoom_drag_end) do active, p0, p1
        active || return Point2f[]
        if !(isfinite(p0[1]) && isfinite(p0[2]) && isfinite(p1[1]) && isfinite(p1[2]))
            return Point2f[]
        end
        x0, y0 = p0
        x1, y1 = p1
        Point2f[
            Point2f(x0, y0), Point2f(x1, y0),
            Point2f(x1, y0), Point2f(x1, y1),
            Point2f(x1, y1), Point2f(x0, y1),
            Point2f(x0, y1), Point2f(x0, y0),
        ]
    end
    # Corner accents: short L-shaped marks at each corner of the zoom rectangle.
    zoom_corner_segments = lift(zoom_drag_active, zoom_drag_start, zoom_drag_end) do active, p0, p1
        active || return Point2f[]
        if !(isfinite(p0[1]) && isfinite(p0[2]) && isfinite(p1[1]) && isfinite(p1[2]))
            return Point2f[]
        end
        x0, y0 = Float32(p0[1]), Float32(p0[2])
        x1, y1 = Float32(p1[1]), Float32(p1[2])
        cx = sign(x1 - x0) * abs(x1 - x0) * ZOOM_BEZIER_FACTOR
        cy = sign(y1 - y0) * abs(y1 - y0) * ZOOM_BEZIER_FACTOR
        Point2f[
            Point2f(x0, y0), Point2f(x0 + cx, y0),   # top-left  horizontal
            Point2f(x0, y0), Point2f(x0, y0 + cy),   # top-left  vertical
            Point2f(x1, y0), Point2f(x1 - cx, y0),   # top-right horizontal
            Point2f(x1, y0), Point2f(x1, y0 + cy),   # top-right vertical
            Point2f(x1, y1), Point2f(x1 - cx, y1),   # bot-right horizontal
            Point2f(x1, y1), Point2f(x1, y1 - cy),   # bot-right vertical
            Point2f(x0, y1), Point2f(x0 + cx, y1),   # bot-left  horizontal
            Point2f(x0, y1), Point2f(x0, y1 - cy),   # bot-left  vertical
        ]
    end
    region_segments_from_points(p0, p1, shape::Symbol) = begin
        if !(isfinite(p0[1]) && isfinite(p0[2]) && isfinite(p1[1]) && isfinite(p1[2]))
            return Point2f[]
        end
        x0, y0 = p0
        x1, y1 = p1
        if shape === :circle
            r = hypot(x1 - x0, y1 - y0)
            r < 0.5 && return Point2f[]
            pts = Point2f[]
            for t in LinRange(0, 2π, 97)
                push!(pts, Point2f(x0 + r * cos(t), y0 + r * sin(t)))
            end
            return pts
        else
            return Point2f[
                Point2f(x0, y0), Point2f(x1, y0),
                Point2f(x1, y1), Point2f(x0, y1),
                Point2f(x0, y0),
            ]
        end
    end
    region_segments = lift(region_start, region_end, region_shape, region_uvs, region_drag_active) do p0, p1, shape, uv, dragging
        (dragging || !isempty(uv)) ? region_segments_from_points(p0, p1, shape) : Point2f[]
    end
    # Crosshair: halo noir épais + trait blanc fin (rendu bi-couche).
    linesegments!(ax_img, crosshair_segments; color = (:black, CROSSHAIR_ALPHA_DARK),  linewidth = CROSSHAIR_LW_DARK,  linestyle = :solid)
    linesegments!(ax_img, crosshair_segments; color = (:white, CROSSHAIR_ALPHA_LIGHT), linewidth = CROSSHAIR_LW_LIGHT, linestyle = :solid)
    linesegments!(ax_cmp, crosshair_segments; color = (:black, CROSSHAIR_ALPHA_DARK),  linewidth = CROSSHAIR_LW_DARK,  linestyle = :solid, visible = compare_visible)
    linesegments!(ax_cmp, crosshair_segments; color = (:white, CROSSHAIR_ALPHA_LIGHT), linewidth = CROSSHAIR_LW_LIGHT, linestyle = :solid, visible = compare_visible)
    # Zoom rectangle: contour pointillé estompé + coins accentués solides.
    linesegments!(ax_img, zoom_box_segments;    color = (ui_selection, ZOOM_BOX_ALPHA),    linewidth = ZOOM_BOX_LW,    linestyle = :dash)
    linesegments!(ax_img, zoom_corner_segments; color = (ui_selection, ZOOM_CORNER_ALPHA), linewidth = ZOOM_CORNER_LW, linestyle = :solid)
    linesegments!(ax_cmp, zoom_box_segments;    color = (ui_selection, ZOOM_BOX_ALPHA),    linewidth = ZOOM_BOX_LW,    linestyle = :dash, visible = compare_visible)
    linesegments!(ax_cmp, zoom_corner_segments; color = (ui_selection, ZOOM_CORNER_ALPHA), linewidth = ZOOM_CORNER_LW, linestyle = :solid, visible = compare_visible)
    # Sélection région: amber semi-transparent.
    lines!(ax_img, region_segments; color = (ui_selection, REGION_ALPHA), linewidth = REGION_LW)
    lines!(ax_cmp, region_segments; color = (ui_selection, REGION_ALPHA), linewidth = REGION_LW, visible = compare_visible)
    marker_points = lift(uv_point, show_marker) do p, enabled
        enabled ? Point2f[p] : Point2f[]
    end
    scatter!(ax_img, marker_points; markersize = MARKER_SIZE)
    scatter!(ax_cmp, marker_points; markersize = MARKER_SIZE, visible = compare_visible)

    # Colorbar linked to plot; tellheight=false avoids layout feedback loops
    img_colorbar = Colorbar(
        img_a_grid[2, 2],
        hm;
        label = display_unit_label,
        width = 20,
        height = _axis_render_height(ax_img),
        tellheight = false,
        valign = :center,
    )
    # Dedicated colorbar for cube B — it lives inside the comparison panel so
    # compare mode keeps each heatmap paired with its own contrast scale.
    img_colorbar_cmp = Colorbar(
        img_cmp_grid[2, 2];
        colormap = cm_obs,
        colorrange = compare_clims_safe,
        label = display_unit_label,
        width = 20,
        height = _axis_render_height(ax_cmp),
        tellheight = false,
        valign = :center,
    )

    # Mini cube-position overview: three X/Y/Z rails below the heatmap.
    # It keeps the active slicing axis and slice index visible while the user
    # scrolls through a cube or flips between slicing directions.
    overview_grid = img_a_grid[3, 1:2] = GridLayout(;
        alignmode = Outside(compact_layout ? 4 : 6),
        tellwidth = false,
        tellheight = false,
    )
    Box(
        overview_grid[1, 1:3];
        color = ui_surface,
        strokecolor = ui_border,
        strokewidth = 0.9,
        cornerradius = 8,
        z = -5,
    )
    Label(
        overview_grid[1, 1];
        text = "X\nY\nZ",
        color = ui_text_muted,
        fontsize = compact_layout ? 10 : 11,
        lineheight = 1.15,
        halign = :right,
        valign = :center,
        tellwidth = false,
        padding = (8, 2, 0, 0),
    )
    ax_overview = Axis(
        overview_grid[1, 2];
        backgroundcolor = :transparent,
        limits = (-0.02f0, 1.02f0, 0.45f0, 3.55f0),
        width = compact_layout ? 190 : 250,
        height = compact_layout ? 30 : 36,
    )
    hidedecorations!(ax_overview)
    hidespines!(ax_overview)
    overview_axis_label(a::Integer) = a == 1 ? "X" : a == 2 ? "Y" : "Z"
    overview_status_obs = lift(axis, idx) do a, id
        "$(overview_axis_label(a)) $(clamp(id, 1, siz[a])) / $(siz[a])"
    end
    Label(
        overview_grid[1, 3];
        text = overview_status_obs,
        color = ui_text,
        fontsize = compact_layout ? 11 : 12,
        halign = :left,
        valign = :center,
        tellwidth = false,
        padding = (2, 10, 0, 0),
    )
    colsize!(overview_grid, 1, Fixed(compact_layout ? 22 : 26))
    colsize!(overview_grid, 2, Fixed(compact_layout ? 200 : 260))
    colsize!(overview_grid, 3, Fixed(compact_layout ? 76 : 92))
    rowsize!(overview_grid, 1, Fixed(compact_layout ? CUBE_OVERVIEW_HEIGHT_COMPACT : CUBE_OVERVIEW_HEIGHT))
    rowsize!(img_a_grid, 3, Fixed(compact_layout ? CUBE_OVERVIEW_HEIGHT_COMPACT : CUBE_OVERVIEW_HEIGHT))
    rowgap!(img_a_grid, compact_layout ? 4 : 7)

    # Panel B (img_cmp_grid) has no X/Y/Z overview rail of its own, but it must
    # reserve the *same* bottom row and row gaps as panel A so the two heatmaps
    # share identical cell geometry. Without this matching spacer the compare
    # axis gets a taller row-2 cell and ends up vertically offset / resized
    # relative to ax_img. The Box is transparent — it only realises the row.
    Box(img_cmp_grid[3, 1:2]; color = :transparent, strokewidth = 0)
    rowsize!(img_cmp_grid, 3, Fixed(compact_layout ? CUBE_OVERVIEW_HEIGHT_COMPACT : CUBE_OVERVIEW_HEIGHT))
    rowgap!(img_cmp_grid, compact_layout ? 4 : 7)

    overview_tracks = Point2f[
        Point2f(0f0, 3f0), Point2f(1f0, 3f0),
        Point2f(0f0, 2f0), Point2f(1f0, 2f0),
        Point2f(0f0, 1f0), Point2f(1f0, 1f0),
    ]
    overview_geom = lift(axis, idx, i_idx, j_idx, k_idx) do a, id, i, j, k
        _cube_overview_geometry(siz, a, id, i, j, k)
    end
    overview_points = lift(g -> g.marker_points, overview_geom)
    overview_active_progress = lift(g -> g.active_progress, overview_geom)
    overview_active_marker = lift(g -> g.active_marker, overview_geom)
    overview_active_point = lift(g -> Point2f[g.marker_points[g.active_axis]], overview_geom)
    linesegments!(
        ax_overview,
        overview_tracks;
        color = (ui_text_muted, 0.32),
        linewidth = CUBE_OVERVIEW_TRACK_LW,
        linestyle = :solid,
    )
    linesegments!(
        ax_overview,
        overview_active_progress;
        color = (ui_selection, 0.86),
        linewidth = CUBE_OVERVIEW_ACTIVE_LW,
        linestyle = :solid,
    )
    scatter!(
        ax_overview,
        overview_points;
        markersize = CUBE_OVERVIEW_DOT_SIZE,
        color = (ui_text, 0.75),
        strokecolor = ui_surface,
        strokewidth = 1.1,
    )
    scatter!(
        ax_overview,
        overview_active_point;
        markersize = CUBE_OVERVIEW_DOT_SIZE + 4,
        color = ui_selection,
        strokecolor = ui_text,
        strokewidth = 0.7,
    )
    linesegments!(
        ax_overview,
        overview_active_marker;
        color = ui_selection,
        linewidth = CUBE_OVERVIEW_MARKER_LW,
        linestyle = :solid,
    )

    # Info + spectrum
    spec_grid = main_grid[1, 2] = GridLayout()
    info_panel = spec_grid[1, 1] = GridLayout(; alignmode = Outside())
    info_box = Box(
        info_panel[1, 1];
        color = ui_surface,
        strokecolor = ui_border,
        strokewidth = 1.0,
        cornerradius = 12,
        z = -5,
    )
    lab_info = Label(
        info_panel[1, 1];
        text      = make_info_tex(1, 1, 1, 1, 1, 0f0),
        halign    = :left,
        valign    = :center,
        fontsize  = 16,
        color     = ui_text,
        padding   = (16, 16, 12, 12),
        lineheight = 1.2,
        tellwidth = false,
    )

    ax_spec = Axis(
        spec_grid[2, 1];
        title  = L"\text{Spectrum at selected pixel}",
        xlabel = L"\text{index along slice axis}",
        ylabel = unit_label_tex,
        width  = spec_axis_width,
        height = spec_axis_height,
        xtickformat = latex_tick_formatter,
        ytickformat = latex_tick_formatter,
    )
    lines!(ax_spec, spec_x_raw, spec_y_disp; label = "A")
    lines!(ax_spec, spec_x_raw, spec_y_compare_disp;
           color = ui_compare, linewidth = 1.6,
           visible = compare_visible, label = "B")
    # Legend gated by compare_visible — hidden when only cube A is loaded so
    # the legend chip does not waste space.
    spec_legend = axislegend(ax_spec; position = :rt,
                             framevisible = true,
                             backgroundcolor = (ui_panel, 0.85),
                             labelcolor = ui_text,
                             framecolor = ui_border,
                             padding = (8, 8, 4, 4))
    # Toggle whole Legend block across Makie versions: older Makie exposes
    # `.visible`, newer Makie only exposes `.scene.visible` / `.blockscene.visible`.
    _set_legend_visible! = (v::Bool) -> begin
        try; spec_legend.visible[] = v; catch; end
        try; spec_legend.scene.visible[] = v; catch; end
        try; spec_legend.blockscene.visible[] = v; catch; end
        nothing
    end
    _set_legend_visible!(compare_visible[])
    on(compare_visible) do v
        _set_legend_visible!(v)
    end
    ax_img.xgridvisible[] = show_grid[]
    ax_img.ygridvisible[] = show_grid[]
    ax_cmp.xgridvisible[] = show_grid[]
    ax_cmp.ygridvisible[] = show_grid[]
    ax_spec.xgridvisible[] = show_grid[]
    ax_spec.ygridvisible[] = show_grid[]

    ax_hist = Axis(
        spec_grid[3, 1];
        title = L"\text{Visible slice histogram}",
        xlabel = unit_label_tex,
        ylabel = hist_ylabel_obs,
        height = hist_axis_height,
        xtickformat = latex_tick_formatter,
        ytickformat = latex_tick_formatter,
    )
    barplot!(ax_hist, hist_x_obs, hist_y_obs; width = hist_width_obs, color = (ui_accent, HIST_BAR_ALPHA), strokecolor = ui_accent, strokewidth = HIST_BAR_STROKE_LW, visible = hist_bars_visible)
    lines!(ax_hist, hist_x_obs, hist_y_obs; color = ui_accent, linewidth = HIST_KDE_LW, visible = hist_kde_visible)
    lines!(ax_hist, compare_hist_x_obs, compare_hist_y_obs; color = ui_compare, linewidth = HIST_COMPARE_LW, visible = compare_visible)
    vlines!(ax_hist, lift(lim -> [first(lim), last(lim)], clims_safe); color = (ui_text_muted, HIST_LIMITS_ALPHA), linewidth = HIST_LIMITS_LW, linestyle = :dash)

    ps_layout = main_grid[1, 2] = GridLayout(;
        alignmode = Outside(compact_layout ? 4 : 8),
        halign = :center,
        valign = :top,
        tellwidth = false,
        tellheight = false,
    )
    ps_header = ps_layout[1, 1] = GridLayout(; tellwidth = false, tellheight = false)
    colgap!(ps_header, compact_layout ? 8 : 12)
    rowgap!(ps_header, compact_layout ? 6 : 8)
    ps_ui_blocks = Any[]
    track_ps!(block) = (push!(ps_ui_blocks, block); block)
    ps_settings_bar = ps_header[1, 1] = GridLayout(; alignmode = Outside(compact_layout ? 5 : 7), tellheight = false)
    track_ps!(Box(ps_settings_bar[1, 1]; color = ui_surface, strokecolor = ui_border,
                  strokewidth = 1.0, cornerradius = 8, z = -5))
    ps_settings = ps_settings_bar[1, 1] = GridLayout(; alignmode = Inside(), tellheight = false)
    colgap!(ps_settings, compact_layout ? 6 : 8)
    rowgap!(ps_settings, compact_layout ? 4 : 6)

    track_ps!(Label(ps_settings[1, 1]; text = "Source", halign = :right, fontsize = 12, color = ui_text_muted))
    ps_src_menu = track_ps!(Menu(ps_settings[1, 2]; options = ["zoom", "full"], prompt = "zoom", width = 72))
    track_ps!(Label(ps_settings[1, 3]; text = "Window", halign = :right, fontsize = 12, color = ui_text_muted))
    ps_win_menu = track_ps!(Menu(ps_settings[1, 4]; options = ["Hann", "Hamming", "None"], prompt = "Hann", width = 82))
    track_ps!(Label(ps_settings[1, 5]; text = "Units", halign = :right, fontsize = 12, color = ui_text_muted))
    ps_unit_menu = track_ps!(Menu(ps_settings[1, 6]; options = ["pixel", "physical"], prompt = "pixel", width = 82))
    track_ps!(Label(ps_settings[1, 7]; text = "PSD cmap", halign = :right, fontsize = 12, color = ui_text_muted))
    ps_cmap_menu = track_ps!(Menu(ps_settings[1, 8]; options = ui_colormap_options(), prompt = String(ps_cmap_name[]), width = 104))

    ps_pad_chk = track_ps!(Checkbox(ps_settings[2, 1]))
    track_ps!(Label(ps_settings[2, 2]; text = "Pad", halign = :left, fontsize = 12, color = ui_text))
    ps_nanapo_chk = track_ps!(Checkbox(ps_settings[2, 3]))
    track_ps!(Label(ps_settings[2, 4]; text = "NaN", halign = :left, fontsize = 12, color = ui_text))
    ps_kmin_box = track_ps!(Textbox(ps_settings[2, 5]; placeholder = "k_min", width = 66, height = 28))
    ps_kmax_box = track_ps!(Textbox(ps_settings[2, 6]; placeholder = "k_max", width = 66, height = 28))
    ps_fit_btn = track_ps!(Button(ps_settings[2, 7]; label = "$(MANTA_ICONS.fit) Fit", width = 66, height = 28))
    ps_clear_fit_btn = track_ps!(Button(ps_settings[2, 8]; label = "Clear", width = 66, height = 28))

    ps_actions = ps_header[1, 2] = GridLayout(; halign = :right, valign = :top, tellwidth = false, tellheight = false)
    colgap!(ps_actions, 6)
    ps_refresh_btn = track_ps!(Button(ps_actions[1, 1]; label = "Refresh", width = 76, height = 28))
    ps_popout_btn = track_ps!(Button(ps_actions[1, 2]; label = "$(MANTA_ICONS.fit) Window", width = 86, height = 28))
    colsize!(ps_header, 1, Auto())
    colsize!(ps_header, 2, Auto())

    ps_plot_grid = ps_layout[2, 1] = GridLayout(2, 4; halign = :center, valign = :top)
    colgap!(ps_plot_grid, -8)
    rowgap!(ps_plot_grid, compact_layout ? 6 : 12)
    rowsize!(ps_layout, 1, Fixed(ps_header_height))
    rowsize!(ps_layout, 2, Relative(1))
    colsize!(ps_plot_grid, 1, Auto())
    rowsize!(ps_plot_grid, 1, Auto())
    rowsize!(ps_plot_grid, 2, Auto())

    # Controls
    controls_grid = main_grid[2, 1:2] = GridLayout(; alignmode = Outside())
    colgap!(controls_grid, controls_gap)
    rowgap!(controls_grid, controls_gap)
    rowsize!(main_grid, 2, Fixed(controls_height))
    compact_layout && rowsize!(main_grid, 1, Fixed(plot_row_height))

    function control_card!(parent, row, col, title::AbstractString; rows::Int = 4, cols::Int = 4,
                           title_color = ui_text)
        card = parent[row, col] = GridLayout(;
            alignmode = Outside(card_pad),
            valign = :top,
            tellwidth = false,
            tellheight = false,
        )
        body_rows = rows + 1
        card_is_dark = ui_theme.background.r < 0.5
        card_border = _theme_rgba(ui_border, card_is_dark ? 0.58 : 0.82)
        header_divider = _theme_rgba(ui_border, card_is_dark ? 0.30 : 0.50)
        # Card body
        Box(card[1:body_rows, 1:cols];
            color = ui_panel, strokecolor = card_border,
            strokewidth = card_is_dark ? 0.8 : 0.9, cornerradius = 8, z = -6)
        # Header band (visually distinct title row)
        Box(card[1, 1:cols];
            color = ui_panel_header, strokecolor = header_divider,
            strokewidth = 0.8, cornerradius = 8, z = -5)
        Label(card[1, 1:cols];
            text = title,
            halign = :left, tellwidth = false,
            fontsize = 12,
            color = title_color,
            padding = (10, 10, 5, 5))
        Box(card[body_rows, 1:cols]; color = :transparent, strokewidth = 0, z = -7)
        rowsize!(card, 1, Fixed(compact_layout ? 28 : 32))
        rowsize!(card, body_rows, Fixed(compact_layout ? 10 : 12))
        rowgap!(card, compact_layout ? 7 : 9)
        colgap!(card, card_gap)
        return card
    end
    control_label!(layout, pos, txt) = Label(layout[pos...]; text = txt, halign = :left, tellwidth = false, fontsize = 13, color = ui_text_muted)

    mode_bar = controls_grid[3, 1:3] = GridLayout(; alignmode = Outside(0), halign = :center)
    colgap!(mode_bar, compact_layout ? 8 : 12)
    mode_segment = mode_bar[1, 1] = GridLayout(; alignmode = Outside(0))
    colgap!(mode_segment, 0)
    mode_nav_btn = Button(mode_segment[1, 1]; label = "$(MANTA_ICONS.nav) Navigation", width = 154, height = 32)
    mode_analysis_btn = Button(mode_segment[1, 2]; label = "$(MANTA_ICONS.analysis) Analysis", width = 140, height = 32)
    mode_export_btn = Button(mode_segment[1, 3]; label = "$(MANTA_ICONS.export_icon) Export", width = 118, height = 32)
    foreach(c -> colsize!(mode_segment, c, Auto()), 1:3)
    help_btn = Button(mode_bar[1, 2]; label = MANTA_ICONS.help, width = 46, height = 32)
    header_btn = Button(mode_bar[1, 3]; label = "HDR", width = 58, height = 32)
    theme_btn = Button(mode_bar[1, 4]; label = is_dark_mode() ? "☀" : "☾", width = 46, height = 32)
    focus_btn = Button(mode_bar[1, 5]; label = "Focus", width = 76, height = 32)
    # Undo / redo buttons — always visible regardless of active mode tab.
    btn_undo = Button(mode_bar[1, 6]; label = MANTA_ICONS.undo, width = 46, height = 32)
    btn_redo = Button(mode_bar[1, 7]; label = MANTA_ICONS.redo, width = 46, height = 32)
    foreach(c -> colsize!(mode_bar, c, Auto()), 1:7)
    view_card = control_card!(controls_grid, 1, 1, "View"; rows = 5, cols = 4)
    control_label!(view_card, (2, 1), "Image")
    img_scale_menu = Menu(view_card[2, 2]; options = scale_menu_options(), prompt = String(scale), width = 96)
    control_label!(view_card, (3, 1), "Spectrum")
    spec_scale_menu = Menu(view_card[3, 2]; options = scale_menu_options(), prompt = String(scale), width = 96)
    reset_zoom_btn = Button(view_card[2, 3:4]; label = "$(MANTA_ICONS.fit) Fit", width = 92, height = 32)
    cube3d_btn = Button(view_card[3, 3:4]; label = "3D cube", width = 92, height = 32)
    ps_btn = Button(view_card[4, 1:2]; label = "Power spectrum", width = 116, height = 32)
    base_layout_btn = Button(view_card[4, 3:4]; label = "Base layout", width = 116, height = 32)
    clim_fix_btn = Button(view_card[5, 1:2]; label = "Fix cbar", width = 116, height = 32)
    clim_auto_nav_btn = Button(view_card[5, 3:4]; label = "Auto cbar", width = 116, height = 32)
    foreach(c -> colsize!(view_card, c, Auto()), 1:4)

    slice_card = control_card!(controls_grid, 1, 2, "Slice"; rows = 5, cols = 5)
    axes_labels = ["dim1 (x)", "dim2 (y)", "dim3 (z)"]
    control_label!(slice_card, (2, 1), "Axis")
    axis_menu = Menu(slice_card[2, 2]; options = axes_labels, prompt = "dim3 (z)", width = 128)
    status_label = Label(slice_card[2, 3:5]; text = latexstring("\\text{axis } 3,\\, \\text{index } 1"), fontsize = 14, halign = :left, tellwidth = false, color = ui_text)
    control_label!(slice_card, (3, 1), "Index A")
    slice_slider = Slider(
        slice_card[3, 2:4];
        range = 1:siz[3],
        startvalue = 1,
        width = compact_layout ? 220 : 260,
        height = 26,
        halign = :left,
    )
    compare_slice_label = control_label!(slice_card, (4, 1), "Index B")
    compare_slice_slider = Slider(
        slice_card[4, 2:4];
        range = 1:siz[3],
        startvalue = 1,
        width = compact_layout ? 220 : 260,
        height = 26,
        halign = :left,
    )
    sigma_title_label = control_label!(slice_card, (5, 1), "Smoothing")
    sigma_label = Label(slice_card[5, 2]; text = latexstring("\\sigma = 1.5\\,\\text{px}"), fontsize = 14, halign = :left, tellwidth = false, color = ui_text)
    sigma_slider = Slider(
        slice_card[5, 3:4];
        range = LinRange(0, 10, 101),
        startvalue = 1.5,
        width = compact_layout ? 150 : 190,
        height = 26,
        halign = :left,
    )
    foreach(c -> colsize!(slice_card, c, Auto()), 1:5)

    contrast_card = control_card!(controls_grid, 1, 1, "$(MANTA_ICONS.contrast) Contrast"; rows = 4, cols = 5)
    clim_min_box   = Textbox(contrast_card[2, 1]; placeholder = "min", width = 120, height = 32)
    clim_max_box   = Textbox(contrast_card[2, 2]; placeholder = "max", width = 120, height = 32)
    clim_apply_btn = Button(contrast_card[2, 3]; label = "Apply", width = 86, height = 32)
    clim_auto_btn  = Button(contrast_card[2, 4]; label = "Auto", width = 78, height = 32)
    clim_p1_btn    = Button(contrast_card[3, 1]; label = "p1-p99", width = 92, height = 32)
    clim_p5_btn    = Button(contrast_card[3, 2]; label = "p5-p95", width = 92, height = 32)
    reset_zoom_analysis_btn = Button(contrast_card[3, 3:4]; label = "$(MANTA_ICONS.fit) Fit", width = 92, height = 32)
    foreach(c -> colsize!(contrast_card, c, Auto()), 1:5)

    # Export mode has only two cards; keep them grouped in the centre instead
    # of pinning them to the outer columns of the three-card control row.
    export_top = controls_grid[1, 1:3] = GridLayout(; alignmode = Outside(0))
    colgap!(export_top, controls_gap)

    export_left_spacer = Box(export_top[1, 1]; color = :transparent, strokewidth = 0)
    output_card = control_card!(export_top, 1, 2, "Output"; rows = 5, cols = 5)
    export_mid_spacer = Box(export_top[1, 3]; color = :transparent, strokewidth = 0)
    anim_card = control_card!(export_top, 1, 4, "Animation"; rows = 4, cols = 5)
    export_right_spacer = Box(export_top[1, 5]; color = :transparent, strokewidth = 0)
    colsize!(export_top, 1, Relative(1))
    colsize!(export_top, 2, Fixed(compact_layout ? 660 : 740))
    colsize!(export_top, 3, Fixed(compact_layout ? 14 : 22))
    colsize!(export_top, 4, Fixed(compact_layout ? 430 : 480))
    colsize!(export_top, 5, Relative(1))

    fmt_menu  = Menu(output_card[2, 1]; options = ["png", "pdf"], prompt = "png", width = 90)
    fname_box = Textbox(output_card[2, 2:4]; placeholder = "filename base", width = 220, height = 32)
    reset_zoom_export_btn = Button(output_card[2, 5]; label = "$(MANTA_ICONS.fit) Fit", width = 92, height = 32)
    btn_save_img  = Button(output_card[3, 1]; label = "$(MANTA_ICONS.export_icon) Image", width = 104, height = 32)
    btn_save_spec = Button(output_card[3, 2]; label = "$(MANTA_ICONS.export_icon) Spectrum", width = 126, height = 32)
    btn_save_state = Button(output_card[3, 3]; label = "$(MANTA_ICONS.export_icon) State", width = 104, height = 32)
    btn_load_state = Button(output_card[3, 4]; label = "Load state", width = 112, height = 32)
    btn_copy_code = Button(output_card[3, 5]; label = "Copy code", width = 112, height = 32)
    btn_show_compare = Button(output_card[4, 1]; label = "$(MANTA_ICONS.compare) Compare", width = 120, height = 32)
    compare_path_box = Textbox(output_card[4, 2:4]; placeholder = "", width = 0, height = 32)
    btn_load_compare = Button(output_card[4, 5]; label = "", width = 0, height = 32)
    compare_mode_menu = Menu(output_card[4, 2:3]; options = ["A", "B", "A - B", "A / B", "resid z"], prompt = "B", width = 0)
    compare_state_label = Label(output_card[5, 1:5]; text = "Comparison: no cube loaded", halign = :left, tellwidth = false, fontsize = 13, color = ui_text_muted)
    foreach(c -> colsize!(output_card, c, Auto()), 1:5)

    region_card = control_card!(controls_grid, 1, 2, "$(MANTA_ICONS.selection) Selection Spectrum";
                                rows = 4, cols = 4, title_color = ui_selection)
    region_mode_menu = Menu(region_card[2, 1]; options = ["point", "box", "circle"], prompt = "point", width = 112)
    region_clear_btn = Button(region_card[2, 2]; label = "Clear", width = 92, height = 32)
    region_count_label = Label(region_card[2, 3:4]; text = "0 px", halign = :left, tellwidth = false, fontsize = 14, color = ui_text_muted)
    spec_ymin_box = Textbox(region_card[3, 1]; placeholder = "y min", width = 92, height = 32)
    spec_ymax_box = Textbox(region_card[3, 2]; placeholder = "y max", width = 92, height = 32)
    spec_y_apply_btn = Button(region_card[3, 3]; label = "Apply y", width = 82, height = 32)
    spec_y_auto_btn = Button(region_card[3, 4]; label = "Auto y", width = 82, height = 32)
    foreach(c -> colsize!(region_card, c, Auto()), 1:4)

    # Histogram takes col 3 of row 1: three content rows match the height of
    # Contrast and Selection Spectrum, making the top strip visually balanced.
    hist_card = control_card!(controls_grid, 1, 3, "$(MANTA_ICONS.histogram) Histogram"; rows = 5, cols = 5)
    hist_mode_menu = Menu(hist_card[2, 1]; options = ["bars", "kde"], prompt = String(hist_mode_obs[]), width = 96)
    hist_bins_box = Textbox(hist_card[2, 2]; placeholder = "bins", width = 76, height = 32)
    hist_apply_btn = Button(hist_card[3, 3]; label = "Apply x", width = 82, height = 32)
    hist_auto_btn = Button(hist_card[3, 4]; label = "Auto x", width = 82, height = 32)
    hist_xmin_box = Textbox(hist_card[3, 1]; placeholder = "x min", width = 92, height = 32)
    hist_xmax_box = Textbox(hist_card[3, 2]; placeholder = "x max", width = 92, height = 32)
    hist_ymin_box = Textbox(hist_card[4, 1]; placeholder = "y min", width = 92, height = 32)
    hist_ymax_box = Textbox(hist_card[4, 2]; placeholder = "y max", width = 92, height = 32)
    hist_y_apply_btn = Button(hist_card[4, 3]; label = "Apply y", width = 82, height = 32)
    hist_y_auto_btn = Button(hist_card[4, 4]; label = "Auto y", width = 82, height = 32)
    foreach(c -> colsize!(hist_card, c, Auto()), 1:5)

    # Bottom row of Analysis mode: Mask | Products | Contours, centered in a sub-grid.
    # Contours moves here (one content row) so row 1 hosts only content-rich cards.
    analysis_bottom = controls_grid[2, 1:3] = GridLayout(; alignmode = Outside(0))
    colgap!(analysis_bottom, controls_gap)
    roomy_compact_controls = compact_layout && fig_size[1] >= 1400
    contour_levels_w = compact_layout ? (roomy_compact_controls ? 140 : 110) : 140
    contour_apply_w = compact_layout ? (roomy_compact_controls ? 82 : 74) : 82
    contour_card_w = compact_layout ? (roomy_compact_controls ? 340 : 310) : 370

    # Contours: single content row → rows = 2 (1 content + bottom spacer).
    # Textbox width trimmed to fit the narrower card in this row.
    contour_card = control_card!(analysis_bottom, 1, 6, "Contours"; rows = 2, cols = 5)
    contour_chk = Checkbox(contour_card[2, 1])
    Label(contour_card[2, 2]; text = "Show", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    contour_levels_box = Textbox(contour_card[2, 3:4]; placeholder = "auto or 1:red, 2:#00ffaa", width = contour_levels_w, height = 32)
    contour_apply_btn = Button(contour_card[2, 5]; label = "Apply", width = contour_apply_w, height = 32)
    foreach(c -> colsize!(contour_card, c, Auto()), 1:5)

    start_box = Textbox(anim_card[2, 1]; placeholder = "start", width = 72, height = 32)
    stop_box  = Textbox(anim_card[2, 2]; placeholder = "stop",  width = 72, height = 32)
    step_box  = Textbox(anim_card[2, 3]; placeholder = "step",  width = 72, height = 32)
    fps_box   = Textbox(anim_card[2, 4]; placeholder = "fps",   width = 72, height = 32)
    play_btn = Button(anim_card[3, 1]; label = "Play", width = 78, height = 32)
    anim_btn = Button(anim_card[3, 2:3]; label = "$(MANTA_ICONS.export_icon) GIF", width = 92, height = 32)
    loop_chk = Checkbox(anim_card[3, 4]); Label(anim_card[3, 5], text = "Loop", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    pingpong_chk = Checkbox(anim_card[4, 1]); Label(anim_card[4, 2:3], text = "Ping-pong", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    foreach(c -> colsize!(anim_card, c, Auto()), 1:5)

    display_card = control_card!(controls_grid, 1, 3, "Display"; rows = 5, cols = 4)
    Label(display_card[2, 1], text = "Colormap", halign = :left, tellwidth = false, fontsize = 14, color = ui_text_muted)
    cmap_menu = colormap_selector!(display_card[2, 2:4]; cmap = cmap, width = 156, compact = compact_layout, theme = ui_theme)
    invert_chk = Checkbox(display_card[3, 1]); invert_label = Label(display_card[3, 2], text = "Invert", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    gauss_chk = Checkbox(display_card[3, 3]); gauss_label = Label(display_card[3, 4], text = "Smoothing", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    crosshair_chk = Checkbox(display_card[4, 1]); Label(display_card[4, 2], text = "Crosshair", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    marker_chk = Checkbox(display_card[4, 3]); Label(display_card[4, 4], text = "Selection", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    grid_chk = Checkbox(display_card[5, 1]); Label(display_card[5, 2], text = "Grid", halign = :left, tellwidth = false, fontsize = 14, color = ui_text)
    foreach(c -> colsize!(display_card, c, Auto()), 1:4)

    rotation_card = control_card!(controls_grid, 2, 1:3, "Rotation projection"; rows = 3, cols = 10)
    Label(rotation_card[2, 1]; text = "Axis", halign = :right, tellwidth = false, fontsize = 13, color = ui_text_muted)
    rot_axis_x_box = Textbox(rotation_card[2, 2]; placeholder = "x", width = 58, height = 32)
    rot_axis_y_box = Textbox(rotation_card[2, 3]; placeholder = "y", width = 58, height = 32)
    rot_axis_z_box = Textbox(rotation_card[2, 4]; placeholder = "z", width = 58, height = 32)
    Label(rotation_card[2, 5]; text = "Angle", halign = :right, tellwidth = false, fontsize = 13, color = ui_text_muted)
    rot_angle_box = Textbox(rotation_card[2, 6]; placeholder = "deg", width = 74, height = 32)
    rot_mode_menu = Menu(rotation_card[2, 7]; options = ["mean", "sum", "max"], prompt = "mean", width = 86)
    rot_project_btn = Button(rotation_card[2, 8]; label = "Project", width = 86, height = 32)
    rot_slice_btn = Button(rotation_card[2, 9]; label = "Slice", width = 72, height = 32)
    rot_reset_btn = Button(rotation_card[2, 10]; label = "Reset", width = 74, height = 32)
    rot_status_label = Label(rotation_card[3, 1:10];
        text = "Projection follows the selected slice axis; the cube is rotated first.",
        halign = :left, tellwidth = false, fontsize = 13,
        color = ui_text_muted, padding = (10, 10, 0, 0))
    foreach(c -> colsize!(rotation_card, c, Auto()), 1:10)

    focus_bar = GridLayout(
        img_grid[2, 1:2];
        alignmode = Outside(0),
        halign = :center,
        valign = :top,
        tellwidth = false,
        tellheight = false,
    )
    colgap!(focus_bar, compact_layout ? 4 : 6)
    Box(focus_bar[1, 1:5];
        color = _theme_rgba(ui_panel, 0.94),
        strokecolor = _theme_rgba(ui_border, 0.75),
        strokewidth = 0.9,
        cornerradius = 8,
        z = -5)
    Label(focus_bar[1, 1];
        text = "Focus",
        halign = :left,
        tellwidth = false,
        fontsize = compact_layout ? 12 : 13,
        color = ui_text,
        padding = (10, 4, 5, 5))
    focus_exit_btn = Button(focus_bar[1, 2]; label = "Exit", width = 62, height = 30)
    focus_fit_btn = Button(focus_bar[1, 3]; label = MANTA_ICONS.fit, width = 42, height = 30)
    focus_auto_btn = Button(focus_bar[1, 4]; label = MANTA_ICONS.contrast, width = 42, height = 30)
    focus_help_btn = Button(focus_bar[1, 5]; label = MANTA_ICONS.help, width = 42, height = 30)
    foreach(c -> colsize!(focus_bar, c, Auto()), 1:5)

    moment_card = control_card!(analysis_bottom, 1, 4, "Products"; rows = 4, cols = 4)
    moment_menu = Menu(moment_card[2, 1:2]; options = ["M0 integrated", "M1 mean", "M2 dispersion"], prompt = "M0 integrated", width = compact_layout ? 156 : 174)
    btn_show_moment = Button(moment_card[2, 3]; label = "Show", width = compact_layout ? 68 : 76, height = 32)
    btn_show_slice = Button(moment_card[2, 4]; label = "Slice", width = compact_layout ? 68 : 76, height = 32)
    fits_product_menu = Menu(moment_card[3, 1:2]; options = ["slice", "region", "moment", "filtered cube", "mask"], prompt = "slice", width = compact_layout ? 136 : 150)
    btn_moment_png = Button(moment_card[3, 3]; label = "PNG", width = compact_layout ? 58 : 66, height = 32)
    btn_moment_fits = Button(moment_card[3, 4]; label = "FITS", width = compact_layout ? 58 : 66, height = 32)
    btn_save_fits = Button(moment_card[4, 1:2]; label = "$(MANTA_ICONS.export_icon) FITS", width = compact_layout ? 86 : 96, height = 32)
    foreach(c -> colsize!(moment_card, c, Auto()), 1:4)

    # ---- Mask card ----
    # Lives in analysis_bottom alongside Products/Contours so the user can
    # build masks while inspecting moments and contour levels. Widget layout:
    #   row 2 : Source menu | Op menu | lo | hi
    #   row 3 : i range box | j range | k range
    #   row 4 : Apply | Reset | Stats label
    mask_card = control_card!(analysis_bottom, 1, 2, "$(MANTA_ICONS.mask) Mask";
                              rows = 4, cols = 4, title_color = ui_mask)
    mask_source_menu = Menu(mask_card[2, 1]; options = ["none", "finite", "threshold", "rectangle"], prompt = "none", width = compact_layout ? 106 : 118)
    mask_op_menu = Menu(mask_card[2, 2]; options = ["≥", "≤", "range", "outside"], prompt = "≥", width = compact_layout ? 76 : 88)
    mask_lo_box = Textbox(mask_card[2, 3]; placeholder = "lo", width = compact_layout ? 58 : 66, height = 32)
    mask_hi_box = Textbox(mask_card[2, 4]; placeholder = "hi", width = compact_layout ? 58 : 66, height = 32)
    mask_i_box = Textbox(mask_card[3, 1:2]; placeholder = "i range a:b", width = compact_layout ? 122 : 142, height = 32)
    mask_j_box = Textbox(mask_card[3, 3]; placeholder = "j range a:b", width = compact_layout ? 84 : 98, height = 32)
    mask_k_box = Textbox(mask_card[3, 4]; placeholder = "k range a:b", width = compact_layout ? 84 : 98, height = 32)
    mask_apply_btn = Button(mask_card[4, 1]; label = "Apply", width = compact_layout ? 68 : 78, height = 32)
    mask_reset_btn = Button(mask_card[4, 2]; label = "Reset", width = compact_layout ? 68 : 78, height = 32)
    mask_status_label = Label(mask_card[4, 3:4]; text = mask_status_obs[], halign = :left, tellwidth = false, fontsize = 13, color = ui_text_muted)
    foreach(c -> colsize!(mask_card, c, Auto()), 1:4)

    # Finalise analysis_bottom: transparent Boxes force spacer columns to exist
    # so colsize! can address them. Cards live in cols 2 (Mask), 4 (Products), 6 (Contours).
    Box(analysis_bottom[1, 1]; color = :transparent, strokewidth = 0)
    Box(analysis_bottom[1, 3]; color = :transparent, strokewidth = 0)
    Box(analysis_bottom[1, 5]; color = :transparent, strokewidth = 0)
    Box(analysis_bottom[1, 7]; color = :transparent, strokewidth = 0)
    colsize!(analysis_bottom, 1, Relative(1))
    colsize!(analysis_bottom, 2, Fixed(compact_layout ? 360 : 440))   # Mask
    colsize!(analysis_bottom, 3, Fixed(compact_layout ? 14 : 22))
    colsize!(analysis_bottom, 4, Fixed(compact_layout ? 380 : 430))   # Products
    colsize!(analysis_bottom, 5, Fixed(compact_layout ? 14 : 22))
    colsize!(analysis_bottom, 6, Fixed(contour_card_w))               # Contours (compact: 1 row)
    colsize!(analysis_bottom, 7, Relative(1))

    foreach(c -> colsize!(controls_grid, c, Relative(1 / 3)), 1:3)
    rowsize!(controls_grid, 1, Fixed(controls_row_heights[1]))
    rowsize!(controls_grid, 2, Fixed(controls_row_heights[2]))
    rowsize!(controls_grid, 3, Fixed(controls_row_heights[3]))

    style_checkbox!(pingpong_chk)
    style_checkbox!(loop_chk)
    style_checkbox!(invert_chk)
    style_checkbox!(gauss_chk)
    style_checkbox!(crosshair_chk)
    style_checkbox!(marker_chk)
    style_checkbox!(grid_chk)
    style_menu!(img_scale_menu)
    style_menu!(spec_scale_menu)
    style_menu!(cmap_menu)
    style_menu!(ps_src_menu)
    style_menu!(ps_win_menu)
    style_menu!(ps_unit_menu)
    style_menu!(ps_cmap_menu)
    style_menu!(fmt_menu)
    style_menu!(compare_mode_menu)
    style_menu!(axis_menu)
    style_menu!(rot_mode_menu)
    style_menu!(moment_menu)
    style_menu!(fits_product_menu)
    style_textbox!(fname_box)
    style_textbox!(compare_path_box)
    style_textbox!(start_box)
    style_textbox!(stop_box)
    style_textbox!(step_box)
    style_textbox!(fps_box)
    style_textbox!(clim_min_box)
    style_textbox!(clim_max_box)
    style_textbox!(ps_kmin_box)
    style_textbox!(ps_kmax_box)
    style_textbox!(rot_axis_x_box)
    style_textbox!(rot_axis_y_box)
    style_textbox!(rot_axis_z_box)
    style_textbox!(rot_angle_box)
    # Mode navigation segmented control (state managed by set_mode_button_active!)
    style_segmented_button!(mode_nav_btn)
    style_segmented_button!(mode_analysis_btn)
    style_segmented_button!(mode_export_btn)
    # Unobtrusive utility
    style_button_ghost!(help_btn)
    style_button_ghost!(header_btn)
    style_button_ghost!(theme_btn)
    style_button_ghost!(focus_btn)
    style_button_ghost!(btn_undo)
    style_button_ghost!(btn_redo)
    style_button_ghost!(focus_exit_btn)
    style_button_ghost!(focus_fit_btn)
    style_button_ghost!(focus_auto_btn)
    style_button_ghost!(focus_help_btn)
    # Resets (ghost)
    style_button_ghost!(reset_zoom_btn)
    style_button_ghost!(cube3d_btn)
    style_button_primary!(rot_project_btn)
    style_button_ghost!(rot_slice_btn)
    style_button_ghost!(rot_reset_btn)
    style_button_ghost!(reset_zoom_analysis_btn)
    style_button_ghost!(reset_zoom_export_btn)
    style_button_ghost!(base_layout_btn)
    style_button_ghost!(clim_fix_btn)
    style_button_ghost!(clim_auto_nav_btn)
    # Secondaires neutres
    style_button_ghost!(ps_btn)
    style_button_ghost!(ps_refresh_btn)
    style_button_ghost!(ps_popout_btn)
    style_button_primary!(ps_fit_btn)
    style_button_ghost!(ps_clear_fit_btn)
    style_checkbox!(ps_pad_chk)
    style_checkbox!(ps_nanapo_chk)
    # Export — action principale + secondaires
    style_button_primary!(btn_save_img)
    style_button_primary!(btn_save_spec)
    style_button_primary!(btn_save_state)
    style_button_ghost!(btn_load_state)
    style_button_ghost!(btn_copy_code)
    style_button_ghost!(btn_show_compare)
    style_button_ghost!(btn_load_compare)
    # Animation
    style_button_ghost!(play_btn)
    style_button_primary!(anim_btn)
    # Contraste
    style_button_primary!(clim_apply_btn)
    style_button_ghost!(clim_auto_btn)
    style_button!(clim_p1_btn)
    style_button!(clim_p5_btn)
    style_menu!(region_mode_menu)
    style_menu!(hist_mode_menu)
    # Region — Clear is destructive → ghost
    style_button_ghost!(region_clear_btn)
    style_checkbox!(contour_chk)
    style_textbox!(contour_levels_box)
    style_button_primary!(contour_apply_btn)
    style_textbox!(spec_ymin_box)
    style_textbox!(spec_ymax_box)
    style_button_primary!(spec_y_apply_btn)
    style_button_ghost!(spec_y_auto_btn)
    style_textbox!(hist_bins_box)
    style_textbox!(hist_xmin_box)
    style_textbox!(hist_xmax_box)
    style_textbox!(hist_ymin_box)
    style_textbox!(hist_ymax_box)
    style_button_primary!(hist_apply_btn)
    style_button_ghost!(hist_auto_btn)
    style_button_primary!(hist_y_apply_btn)
    style_button_ghost!(hist_y_auto_btn)
    style_button!(btn_show_moment)
    style_button!(btn_show_slice)
    style_button_primary!(btn_moment_png)
    style_button_primary!(btn_moment_fits)
    style_button_primary!(btn_save_fits)
    style_menu!(mask_source_menu)
    style_menu!(mask_op_menu)
    style_textbox!(mask_lo_box)
    style_textbox!(mask_hi_box)
    style_textbox!(mask_i_box)
    style_textbox!(mask_j_box)
    style_textbox!(mask_k_box)
    style_button_primary!(mask_apply_btn)
    style_button_ghost!(mask_reset_btn)
    mask_apply_btn.buttoncolor[] = ui_mask
    mask_apply_btn.buttoncolor_hover[] = ui_theme.mask
    mask_apply_btn.buttoncolor_active[] = ui_theme.mask
    mask_apply_btn.strokecolor[] = ui_mask
    btn_show_compare.labelcolor_hover[] = ui_compare
    btn_show_compare.labelcolor_active[] = ui_compare
    btn_load_compare.labelcolor_hover[] = ui_compare
    btn_load_compare.labelcolor_active[] = ui_compare
    style_slider!(slice_slider)
    style_slider!(compare_slice_slider)
    style_slider!(sigma_slider)

    if compact_layout
        for btn in (help_btn, header_btn, theme_btn, focus_btn,
                    reset_zoom_btn, reset_zoom_analysis_btn, reset_zoom_export_btn,
                    cube3d_btn, rot_project_btn, rot_slice_btn, rot_reset_btn,
                    ps_btn, base_layout_btn, ps_refresh_btn, ps_popout_btn,
                    ps_fit_btn, ps_clear_fit_btn,
                    btn_undo, btn_redo,
                    focus_exit_btn, focus_fit_btn, focus_auto_btn, focus_help_btn,
                    btn_save_img, btn_save_spec, btn_save_state, btn_load_state, btn_copy_code,
                    btn_show_compare, btn_load_compare, play_btn, anim_btn, clim_apply_btn,
                    clim_auto_btn, clim_fix_btn, clim_auto_nav_btn,
                    clim_p1_btn, clim_p5_btn, region_clear_btn, contour_apply_btn,
                    spec_y_apply_btn, spec_y_auto_btn, hist_apply_btn, hist_auto_btn, hist_y_apply_btn, hist_y_auto_btn,
                    btn_show_moment, btn_show_slice, btn_moment_png, btn_moment_fits, btn_save_fits,
                    mask_apply_btn, mask_reset_btn)
            btn.height[] = 30
            btn.fontsize[] = 13
            btn.padding[] = (9, 9, 5, 5)
        end
        for menu in (img_scale_menu, spec_scale_menu, cmap_menu, ps_src_menu, ps_win_menu,
                     ps_unit_menu, ps_cmap_menu, fmt_menu, compare_mode_menu, axis_menu, region_mode_menu, hist_mode_menu,
                     rot_mode_menu, moment_menu, fits_product_menu, mask_source_menu, mask_op_menu)
            menu.height[] = 30
            menu.fontsize[] = 13
            menu.textpadding[] = (8, 8, 5, 5)
            menu.dropdown_arrow_size[] = 10
        end
        for tb in (ps_kmin_box, ps_kmax_box, clim_min_box, clim_max_box, fname_box,
                   compare_path_box, contour_levels_box, spec_ymin_box, spec_ymax_box,
                   hist_bins_box, hist_xmin_box, hist_xmax_box, hist_ymin_box, hist_ymax_box,
                   start_box, stop_box, step_box, fps_box,
                   rot_axis_x_box, rot_axis_y_box, rot_axis_z_box, rot_angle_box,
                   mask_lo_box, mask_hi_box, mask_i_box, mask_j_box, mask_k_box)
            tb.height[] = 30
            tb.fontsize[] = 13
            tb.textpadding[] = (8, 8, 5, 5)
        end
        for chk in (ps_pad_chk, ps_nanapo_chk, pingpong_chk, loop_chk, invert_chk,
                    gauss_chk, crosshair_chk, marker_chk, grid_chk, contour_chk)
            chk.size[] = 18
            chk.checkmarksize[] = 0.58
        end
        for sl in (slice_slider, compare_slice_slider, sigma_slider)
            sl.height[] = 20
            sl.linewidth[] = 8
        end
    end

    invert_chk.checked[] = invert_cmap[]
    cmap_menu.selection[] = String(cmap_name[])
    gauss_chk.checked[] = gauss_on[]
    crosshair_chk.checked[] = show_crosshair[]
    marker_chk.checked[] = show_marker[]
    grid_chk.checked[] = show_grid[]
    contour_chk.checked[] = show_contours[]
    loop_chk.checked[] = true
    hint_label = Label(
        main_grid[3, 2];
        text      = "arrows: move crosshair    left-click: pick / draw region    right-drag: zoom    i: invert colormap    d: dark mode",
        halign    = :right,
        fontsize  = 13,
        color     = ui_text_muted,
        tellwidth = false,
    )
    status_footer_label = Label(
        main_grid[4, 1:2];
        text = ui_status,
        color = ui_text_muted,
        halign = :left,
        tellwidth = false,
    )
    if compact_layout
        hint_label.visible[] = false
        status_footer_label.visible[] = false
    end
    # ---------- Undo/redo button enable state ----------
    on(_undo_stack.can_undo; update = true) do can
        t = ui_theme_ref[]
        btn_undo.labelcolor[]       = can ? t.text          : t.text_muted
        btn_undo.labelcolor_hover[] = can ? t.accent_strong : t.text_muted
    end
    on(_undo_stack.can_redo; update = true) do can
        t = ui_theme_ref[]
        btn_redo.labelcolor[]       = can ? t.text          : t.text_muted
        btn_redo.labelcolor_hover[] = can ? t.accent_strong : t.text_muted
    end
    # ---------- Helpers ----------
    function set_status!(msg::AbstractString)
        s = String(msg)
        ui_status[] = s
        status_footer_label.color[] = _cube_status_color(s, ui_theme_ref[])
        nothing
    end
    set_box_text!(tb, s::AbstractString) = begin
        str = String(s)
        tb.displayed_string[] = str
        tb.stored_string[] = str
        nothing
    end
    # Visual validity cue: red border on rejected (empty/invalid) input,
    # normal themed border when the field parses cleanly. See
    # `manta_flag_textbox!` in helpers/UITheme.jl.
    flag_box!(tb, ok::Bool) = manta_flag_textbox!(tb, ok, ui_theme_ref[])
    set_box_text!(rot_axis_x_box, "0")
    set_box_text!(rot_axis_y_box, "0")
    set_box_text!(rot_axis_z_box, "1")
    set_box_text!(rot_angle_box, "0")
    rot_mode_menu.selection[] = "mean"
    set_box_text!(hist_bins_box, string(hist_bins_obs[]))
    if hist_xlimits_manual[]
        lo, hi = hist_xlimits_manual_value[]
        set_box_text!(hist_xmin_box, string(lo))
        set_box_text!(hist_xmax_box, string(hi))
    end
    if hist_ylimits_manual[]
        lo, hi = hist_ylimits_manual_value[]
        set_box_text!(hist_ymin_box, string(lo))
        set_box_text!(hist_ymax_box, string(hi))
    end
    if spec_ylimits_source[] !== :auto
        lo, hi = spec_ylimits_value[]
        set_box_text!(spec_ymin_box, string(lo))
        set_box_text!(spec_ymax_box, string(hi))
    end

    set_block_visible!(block, visible::Bool) = begin
        try
            block.visible[] = visible
        catch
        end
        try
            block.scene.visible[] = visible
        catch
        end
        try
            block.blockscene.visible[] = visible
        catch
        end
        # visible=false alone does not remove a block from GLMakie's pick-buffer:
        # hidden blocks remain "pickable" and silently consume clicks, preventing
        # visible blocks stacked in the same grid cell from receiving events.
        # Translating the blockscene far off-screen when hidden solves this: the
        # block is physically outside the viewport so GLMakie never picks it.
        # Resetting to Vec3f(0) when visible restores the layout-computed position
        # (UI blocks are positioned via their viewport, not via the transform).
        try
            translate!(block.blockscene,
                       visible ? Vec3f(0) : Vec3f(-1_000_000f0, 0f0, 0f0))
        catch
        end
        nothing
    end

    function set_layout_contents_visible!(layout, visible::Bool)
        for block in try
            contents(layout)
        catch
            Any[]
        end
            set_block_visible!(block, visible)
            block isa GridLayout && set_layout_contents_visible!(block, visible)
        end
        nothing
    end

    nav_cards = (view_card, slice_card, display_card, rotation_card)
    analysis_cards = (contrast_card, region_card, contour_card, hist_card, moment_card, mask_card)
    export_cards = (output_card, anim_card)
    export_spacers = (export_left_spacer, export_mid_spacer, export_right_spacer)
    set_layout_contents_visible!(focus_bar, false)

    function set_mode_button_active!(btn, active::Bool)
        style_segmented_button!(btn; active = active)
        nothing
    end

    function refresh_control_mode!()
        if focus_image[]
            set_layout_contents_visible!(controls_grid, false)
            set_layout_contents_visible!(focus_bar, true)
            return nothing
        end
        mode = control_mode[]
        set_layout_contents_visible!(controls_grid, true)
        set_layout_contents_visible!(focus_bar, false)
        for card in nav_cards
            set_layout_contents_visible!(card, mode === :navigation)
        end
        for card in analysis_cards
            set_layout_contents_visible!(card, mode === :analysis)
        end
        for card in export_cards
            set_layout_contents_visible!(card, mode === :export)
        end
        for spacer in export_spacers
            set_block_visible!(spacer, mode === :export)
        end
        set_block_visible!(compare_slice_label, mode === :navigation && compare_visible[])
        set_block_visible!(compare_slice_slider, mode === :navigation && compare_visible[])
        set_mode_button_active!(mode_nav_btn, mode === :navigation)
        set_mode_button_active!(mode_analysis_btn, mode === :analysis)
        set_mode_button_active!(mode_export_btn, mode === :export)
        nothing
    end
    refresh_control_mode!()

    function _retint_theme_color(c, old_theme::MANTAUITheme, new_theme::MANTAUITheme)
        c == old_theme.panel         && return new_theme.panel
        c == old_theme.panel_header  && return new_theme.panel_header
        c == old_theme.accent        && return new_theme.accent
        c == old_theme.accent_dim    && return new_theme.accent_dim
        c == old_theme.accent_strong && return new_theme.accent_strong
        c == old_theme.track         && return new_theme.track
        c == old_theme.surface       && return new_theme.surface
        c == old_theme.surface_hover && return new_theme.surface_hover
        c == old_theme.surface_active && return new_theme.surface_active
        c == old_theme.border        && return new_theme.border
        c == old_theme.border_strong && return new_theme.border_strong
        c == old_theme.text          && return new_theme.text
        c == old_theme.text_muted    && return new_theme.text_muted
        c == old_theme.background    && return new_theme.background
        c == old_theme.selection     && return new_theme.selection
        c == old_theme.compare       && return new_theme.compare
        c == old_theme.success       && return new_theme.success
        c == old_theme.mask          && return new_theme.mask
        c == old_theme.error         && return new_theme.error
        return c
    end

    function _retint_prop!(block, prop::Symbol, old_theme::MANTAUITheme, new_theme::MANTAUITheme)
        hasproperty(block, prop) || return nothing
        obs = getproperty(block, prop)
        current = try
            obs[]
        catch
            return nothing
        end
        updated = _retint_theme_color(current, old_theme, new_theme)
        updated == current && return nothing
        try
            obs[] = updated
        catch
        end
        return nothing
    end

    function _retint_layout_contents!(layout, old_theme::MANTAUITheme, new_theme::MANTAUITheme)
        blocks = try
            contents(layout)
        catch
            Any[]
        end
        for block in blocks
            for prop in (:color, :strokecolor, :backgroundcolor, :textcolor,
                         :labelcolor, :labelcolor_hover, :labelcolor_active,
                         :bordercolor, :bordercolor_hover, :bordercolor_focused,
                         :boxcolor, :boxcolor_hover, :boxcolor_focused)
                _retint_prop!(block, prop, old_theme, new_theme)
            end
            block isa GridLayout && _retint_layout_contents!(block, old_theme, new_theme)
        end
        return nothing
    end

    function _style_axis_theme!(ax, theme::MANTAUITheme)
        grid_c = _theme_rgba(theme.border, 0.45)
        mgrid_c = _theme_rgba(theme.border, 0.20)
        for (prop, val) in (
            :backgroundcolor  => theme.panel,
            :xgridcolor       => grid_c,
            :ygridcolor       => grid_c,
            :xminorgridcolor  => mgrid_c,
            :yminorgridcolor  => mgrid_c,
            :xticklabelcolor  => theme.text_muted,
            :yticklabelcolor  => theme.text_muted,
            :xlabelcolor      => theme.text,
            :ylabelcolor      => theme.text,
            :titlecolor       => theme.text,
            :topspinecolor    => theme.border_strong,
            :bottomspinecolor => theme.border_strong,
            :leftspinecolor   => theme.border_strong,
            :rightspinecolor  => theme.border_strong,
            :xtickcolor       => theme.border_strong,
            :ytickcolor       => theme.border_strong,
        )
            _retint_prop!(ax, prop, ui_theme_ref[], theme)
            try
                getproperty(ax, prop)[] = val
            catch
            end
        end
        return nothing
    end

    function _style_colorbar_theme!(cb, theme::MANTAUITheme)
        for (prop, val) in (
            :labelcolor       => theme.text,
            :ticklabelcolor   => theme.text_muted,
            :topspinecolor    => theme.border_strong,
            :bottomspinecolor => theme.border_strong,
            :leftspinecolor   => theme.border_strong,
            :rightspinecolor  => theme.border_strong,
        )
            try
                getproperty(cb, prop)[] = val
            catch
            end
        end
        return nothing
    end

    function _restyle_cube_theme!(old_theme::MANTAUITheme, new_theme::MANTAUITheme)
        ui_theme_ref[] = new_theme
        try; fig.backgroundcolor[] = new_theme.background; catch; end
        try; fig.scene.backgroundcolor[] = new_theme.background; catch; end

        _retint_layout_contents!(fig.layout, old_theme, new_theme)
        for ax in (ax_img, ax_cmp, ax_spec, ax_hist)
            _style_axis_theme!(ax, new_theme)
        end
        for cb in (img_colorbar, img_colorbar_cmp)
            _style_colorbar_theme!(cb, new_theme)
        end
        try; spec_legend.backgroundcolor[] = (new_theme.panel, 0.85); catch; end
        try; spec_legend.labelcolor[] = new_theme.text; catch; end
        try; spec_legend.framecolor[] = new_theme.border; catch; end

        for chk in (pingpong_chk, loop_chk, invert_chk, gauss_chk, crosshair_chk,
                    marker_chk, grid_chk, contour_chk, ps_pad_chk, ps_nanapo_chk)
            style_checkbox!(chk)
        end
        for menu in (img_scale_menu, spec_scale_menu, cmap_menu, ps_src_menu, ps_win_menu,
                     ps_unit_menu, ps_cmap_menu, fmt_menu, compare_mode_menu, axis_menu, moment_menu,
                     fits_product_menu, region_mode_menu, hist_mode_menu, rot_mode_menu, mask_source_menu,
                     mask_op_menu)
            style_menu!(menu)
        end
        for tb in (fname_box, compare_path_box, start_box, stop_box, step_box, fps_box,
                   clim_min_box, clim_max_box, ps_kmin_box, ps_kmax_box, contour_levels_box,
                   spec_ymin_box, spec_ymax_box, hist_bins_box, hist_xmin_box, hist_xmax_box,
                   hist_ymin_box, hist_ymax_box,
                   rot_axis_x_box, rot_axis_y_box, rot_axis_z_box, rot_angle_box,
                   mask_lo_box, mask_hi_box, mask_i_box,
                   mask_j_box, mask_k_box)
            style_textbox!(tb)
        end
        for btn in (help_btn, header_btn, theme_btn, focus_btn, focus_exit_btn,
                    focus_fit_btn, focus_auto_btn, focus_help_btn,
                    reset_zoom_btn, reset_zoom_analysis_btn,
                    reset_zoom_export_btn, cube3d_btn, base_layout_btn, clim_fix_btn, clim_auto_nav_btn,
                    rot_slice_btn, rot_reset_btn,
                    ps_btn, ps_refresh_btn, ps_popout_btn, ps_clear_fit_btn,
                    btn_load_state, btn_copy_code,
                    btn_show_compare, btn_load_compare, play_btn, clim_auto_btn, region_clear_btn,
                    spec_y_auto_btn, hist_auto_btn, hist_y_auto_btn, mask_reset_btn)
            style_button_ghost!(btn)
        end
        for btn in (btn_save_img, btn_save_spec, btn_save_state, anim_btn, clim_apply_btn,
                    contour_apply_btn, spec_y_apply_btn, hist_apply_btn, hist_y_apply_btn,
                    rot_project_btn, ps_fit_btn, btn_moment_png, btn_moment_fits,
                    btn_save_fits, mask_apply_btn)
            style_button_primary!(btn)
        end
        for btn in (clim_p1_btn, clim_p5_btn, btn_show_moment, btn_show_slice)
            style_button!(btn)
        end
        style_slider!(slice_slider)
        style_slider!(compare_slice_slider)
        style_slider!(sigma_slider)

        mask_apply_btn.buttoncolor[] = new_theme.mask
        mask_apply_btn.buttoncolor_hover[] = new_theme.mask
        mask_apply_btn.buttoncolor_active[] = new_theme.mask
        mask_apply_btn.strokecolor[] = new_theme.mask
        btn_show_compare.labelcolor_hover[] = new_theme.compare
        btn_show_compare.labelcolor_active[] = new_theme.compare
        btn_load_compare.labelcolor_hover[] = new_theme.compare
        btn_load_compare.labelcolor_active[] = new_theme.compare
        theme_btn.label[] = is_dark_mode() ? "☀" : "☾"

        refresh_control_mode!()
        status_footer_label.color[] = _cube_status_color(ui_status[], new_theme)
        notify(_undo_stack.can_undo)
        notify(_undo_stack.can_redo)
        return nothing
    end

    function toggle_dark_mode!()
        old_theme = ui_theme_ref[]
        set_dark_mode!(!is_dark_mode())
        _restyle_cube_theme!(old_theme, current_ui_theme())
        set_status!(is_dark_mode() ? "Dark mode enabled." : "Light mode enabled.")
        return nothing
    end

    on(theme_btn.clicks) do _
        toggle_dark_mode!()
    end

    on(focus_help_btn.clicks) do _
        help_btn.clicks[] = help_btn.clicks[] + 1
    end

    # ---------- Mode-gated event helper ----------
    # Cards from different modes share the same grid cell.  set_block_visible!
    # now also translates hidden blocks off-screen (Vec3f(-1e6,0,0)) so that
    # GLMakie excludes them from its pick-buffer — this is the primary fix for
    # the "accidental click on another tab's button" bug.
    # `on_mode` is kept as a second line of defence: if GLMakie ever delivers a
    # spurious event to a hidden block (e.g. due to a picking race on the frame
    # the translation is applied), the callback still fires only when the active
    # mode matches, neutralising leak-through clicks.
    # `bypass_mode_gate` is a re-entrant escape hatch for routines that
    # programmatically poke widgets across modes (load state, etc.).
    bypass_mode_gate = Ref(false)
    on_mode(callback, obs, mode::Symbol) = on(obs) do v
        (bypass_mode_gate[] || control_mode[] === mode) || return
        callback(v)
    end

    on_mode(cube3d_btn.clicks, :navigation) do _
        try
            open_cube3d_window!()
        catch e
            msg = "Failed to open 3D cube: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception = (e, catch_backtrace())
        end
    end

    parse_rotation_box(tb, fallback::Float32) = begin
        txt = strip(get_box_str(tb))
        isempty(txt) && return fallback
        parsed = tryparse(Float32, txt)
        parsed === nothing ? fallback : parsed
    end

    function apply_rotation_projection_from_ui!()
        axv = (
            parse_rotation_box(rot_axis_x_box, rotation_axis_obs[][1]),
            parse_rotation_box(rot_axis_y_box, rotation_axis_obs[][2]),
            parse_rotation_box(rot_axis_z_box, rotation_axis_obs[][3]),
        )
        deg = parse_rotation_box(rot_angle_box, rotation_angle_obs[])
        norm_axis = sqrt(axv[1]^2 + axv[2]^2 + axv[3]^2)
        if !(isfinite(norm_axis) && norm_axis > eps(Float32) && isfinite(deg))
            flag_box!(rot_axis_x_box, false)
            flag_box!(rot_axis_y_box, false)
            flag_box!(rot_axis_z_box, false)
            flag_box!(rot_angle_box, isfinite(deg))
            rot_status_label.color[] = ui_error
            rot_status_label.text[] = "Rotation axis must be non-zero and angle must be finite."
            set_status!("Invalid rotation projection parameters.")
            return nothing
        end
        rotation_axis_obs[] = axv
        rotation_angle_obs[] = deg
        rotation_projection_mode[] = String(rot_mode_menu.selection[]) == "sum" ? :sum :
                                     String(rot_mode_menu.selection[]) == "max" ? :max :
                                     :mean
        view_product[] = :rotproj
        use_manual[] = false
        flag_box!(rot_axis_x_box, true)
        flag_box!(rot_axis_y_box, true)
        flag_box!(rot_axis_z_box, true)
        flag_box!(rot_angle_box, true)
        rot_status_label.color[] = ui_text_muted
        rot_status_label.text[] = "Showing $(String(rotation_projection_mode[])) projection after $(round(deg; digits = 3)) degrees around ($(axv[1]), $(axv[2]), $(axv[3]))."
        autolimits!(ax_img)
        refresh_labels!()
        refresh_hist_axes!()
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
        set_status!("Rotated 2D projection displayed in the main view.")
        return nothing
    end

    on_mode(rot_project_btn.clicks, :navigation) do _
        apply_rotation_projection_from_ui!()
    end

    on_mode(rot_slice_btn.clicks, :navigation) do _
        view_product[] = :slice
        use_manual[] = false
        rot_status_label.color[] = ui_text_muted
        rot_status_label.text[] = "Slice view restored; rotation parameters kept."
        autolimits!(ax_img)
        refresh_labels!()
        refresh_hist_axes!()
        set_status!("Slice view restored.")
    end

    on_mode(rot_reset_btn.clicks, :navigation) do _
        rotation_axis_obs[] = (0f0, 0f0, 1f0)
        rotation_angle_obs[] = 0f0
        rotation_projection_mode[] = :mean
        rot_mode_menu.selection[] = "mean"
        set_box_text!(rot_axis_x_box, "0")
        set_box_text!(rot_axis_y_box, "0")
        set_box_text!(rot_axis_z_box, "1")
        set_box_text!(rot_angle_box, "0")
        view_product[] = :slice
        use_manual[] = false
        rot_status_label.color[] = ui_text_muted
        rot_status_label.text[] = "Rotation reset and slice view restored."
        autolimits!(ax_img)
        refresh_labels!()
        refresh_hist_axes!()
        set_status!("Rotation projection reset.")
    end

    on_mode(rot_mode_menu.selection, :navigation) do sel
        sel === nothing && return
        rotation_projection_mode[] = String(sel) == "sum" ? :sum :
                                     String(sel) == "max" ? :max :
                                     :mean
        view_product[] === :rotproj && (refresh_labels!(); refresh_hist_axes!())
    end

    # Hit-test a clickable Label by its on-screen bounding box.
    #
    # `is_mouseinside(label.blockscene)` is NOT a containment test: a Block's
    # `blockscene` spans the *entire* figure, so it returns true for a click
    # anywhere in the window. Using it as a guard made every click toggle the
    # widget (Invert flipped on each click; the two smoothing labels both fired
    # and silently cancelled out). Test the cursor against the label's computed
    # layout bbox in figure pixel space instead — the same coordinate frame in
    # which `events.mouseposition` is reported.
    _click_in_label(label) =
        events(label.blockscene).mouseposition[] in label.layoutobservables.computedbbox[]

    function toggle_smoothing_from_label!(label, ev)
        ev.button == Mouse.left && ev.action == Mouse.press || return Consume(false)
        (bypass_mode_gate[] || control_mode[] === :navigation) || return Consume(false)
        _click_in_label(label) || return Consume(false)
        gauss_chk.checked[] = !gauss_chk.checked[]
        return Consume(true)
    end
    function toggle_invert_from_label!(ev)
        ev.button == Mouse.left && ev.action == Mouse.press || return Consume(false)
        (bypass_mode_gate[] || control_mode[] === :navigation) || return Consume(false)
        _click_in_label(invert_label) || return Consume(false)
        invert_chk.checked[] = !invert_chk.checked[]
        return Consume(true)
    end
    on(toggle_invert_from_label!, events(invert_label.blockscene).mousebutton)
    on(events(sigma_title_label.blockscene).mousebutton) do ev
        toggle_smoothing_from_label!(sigma_title_label, ev)
    end
    on(events(gauss_label.blockscene).mousebutton) do ev
        toggle_smoothing_from_label!(gauss_label, ev)
    end

    # ---------- Undo/redo: state registration observers ----------
    # Each key observable pushes a snapshot whenever it changes — unless we are
    # already inside a replay (`_undo_stack.suppress == true`).
    # `region_shape` is intentionally absent here: it always changes in lockstep
    # with `selection_mode` (and is derived from it on replay), so registering
    # `selection_mode` alone avoids an inconsistent intermediate snapshot. The
    # transient drag points (`region_start`/`region_end`) are likewise omitted —
    # only the committed `region_uvs` (set once at drag end) drives a snapshot.
    for _obs in (axis, idx, compare_idx, cmap_name, invert_cmap, img_scale_mode,
                 use_manual, clims_manual, mask_source_obs, region_uvs,
                 selection_mode, view_product, moment_order)
        on(_obs) do _
            _undo_stack.suppress && return
            register_state!(_undo_stack, _nav_snapshot())
        end
    end

    # Restore a snapshot produced by `_nav_snapshot()`. Runs inside
    # `with_suppression` so the above observers do not record spurious entries.
    function _apply_undo_snap!(snap)
        snap === nothing && return
        with_suppression(_undo_stack) do
            bypass_mode_gate[] = true
            try
                axis_menu.selection[]       = axes_labels[snap.axis]
                slice_slider.value[]        = snap.idx
                compare_slice_slider.value[] = get(snap, :compare_idx, snap.idx)
                img_scale_menu.selection[]  = String(snap.img_scale_mode)
                cmap_name[]                 = snap.cmap_name
                String(snap.cmap_name) in MANTA_COLORMAP_OPTIONS &&
                    (cmap_menu.selection[] = String(snap.cmap_name))
                invert_chk.checked[]        = snap.invert_cmap
                if snap.use_manual
                    clims_manual[] = snap.clims_manual
                    use_manual[]   = true
                    set_box_text!(clim_min_box, string(first(snap.clims_manual)))
                    set_box_text!(clim_max_box, string(last(snap.clims_manual)))
                else
                    use_manual[] = false
                    set_box_text!(clim_min_box, "")
                    set_box_text!(clim_max_box, "")
                end

                # Moment product. Setting the observables retriggers the slice
                # pipeline; sync the menu label so the visible widget matches.
                moment_order[] = snap.moment_order
                moment_menu.selection[] = snap.moment_order == 1 ? "M1 mean" :
                                          snap.moment_order == 2 ? "M2 dispersion" :
                                                                   "M0 integrated"
                view_product[] = snap.view_product

                # Mask: replay through `apply_mask_source!` so the BitArray is
                # rematerialised, the moment cache cleared and the status label
                # refreshed — exactly as a fresh mask application would do.
                apply_mask_source!(snap.mask_source)

                # Region selection. `region_shape` is derived from
                # `selection_mode`; the drag points restore the overlay.
                selection_mode[] = snap.selection_mode
                region_shape[]   = snap.selection_mode === :circle ? :circle : :box
                region_uvs[]     = snap.region_uvs
                region_start[]   = Point2f(snap.region_p0[1], snap.region_p0[2])
                region_end[]     = Point2f(snap.region_p1[1], snap.region_p1[2])
                region_count_label.text[] = "$(length(snap.region_uvs)) px"

                refresh_labels!()
                refresh_spectrum!()
            finally
                bypass_mode_gate[] = false
            end
        end
    end

    on(btn_undo.clicks) do _
        snap = undo!(_undo_stack)
        snap === nothing && (set_status!("Nothing to undo."); return)
        _apply_undo_snap!(snap)
        set_status!("Undo.")
    end
    on(btn_redo.clicks) do _
        snap = redo!(_undo_stack)
        snap === nothing && (set_status!("Nothing to redo."); return)
        _apply_undo_snap!(snap)
        set_status!("Redo.")
    end

    function set_compare_panel_visible!(visible::Bool)
        set_block_visible!(ax_cmp, visible)
        set_block_visible!(img_colorbar_cmp, visible)
        set_block_visible!(chip_a_box, visible)
        set_block_visible!(chip_a_label, visible)
        set_block_visible!(chip_cmp_box, visible)
        set_block_visible!(chip_cmp_label, visible)
        rowsize!(img_a_grid, 1, visible ? Fixed(chip_row_height) : Fixed(0))
        rowsize!(img_cmp_grid, 1, visible ? Fixed(chip_row_height) : Fixed(0))
        colsize!(img_grid, 2, visible ? Auto() : Fixed(0))
        nothing
    end

    set_compare_panel_visible!(false)

    # ---------- Spectrum helpers ----------
    # See src/views/cube/SpectrumBundle.jl
    (; refresh_spec_ylim!, refresh_spectrum!) = _cube_spectrum_bundle(;
        data, siz, axis, i_idx, j_idx, k_idx,
        region_uvs, mask_bits_obs, compare_data,
        spec_x_axes, spec_x_raw, spec_y_raw, spec_y_buf,
        spec_y_compare_raw, spec_y_compare_buf,
        spec_ylimits_source, spec_ylimits_value, ax_spec,
    )

    # ---------- Compare: loader UI + cube alignment ----------
    # See src/views/cube/CompareBundle.jl
    (; show_compare_loader!, hide_compare_loader!, resolve_compare_path, pick_compare_path, load_compare_cube!) =
        _cube_compare_bundle(;
            filepath, wcs, data, siz,
            compare_data, compare_header = compare_header_ref,
            compare_visible, compare_name, compare_path_current,
            btn_show_compare, compare_mode_menu, compare_path_box, btn_load_compare,
            compare_state_label, ax_cmp, img_colorbar_cmp, img_grid,
            show_grid, ui_text_muted, ui_success,
            refresh_spectrum!, set_status!, set_block_visible!, set_box_text!,
        )

    cube3d_fig_ref = Ref{Any}(nothing)
    cube3d_alive_ref = Ref(false)

    function open_cube3d_window!()
        if cube3d_alive_ref[] && cube3d_fig_ref[] !== nothing
            set_status!("3D cube window already open.")
            return cube3d_fig_ref[]
        end

        theme = ui_theme_ref[]
        rot_obs = Observable(_cube3d_identity_rotation())
        geom_obs = lift(axis, idx, i_idx, j_idx, k_idx, rot_obs) do a, id, i, j, k, R
            _cube3d_view_geometry(siz, a, id, i, j, k, R)
        end

        fig3 = Figure(size = (900, 720); backgroundcolor = theme.background)
        Label(fig3[1, 1];
              text = "3D cube orientation",
              halign = :center,
              tellwidth = false,
              fontsize = 20,
              color = theme.text,
              padding = (0, 0, 10, 4))

        ax3 = Axis3(
            fig3[2, 1];
            title = latexstring("\\text{", latex_safe(fname), " cube frame}"),
            xlabel = "x",
            ylabel = "y",
            zlabel = "z",
            aspect = :data,
            limits = (-0.72f0, 0.72f0, -0.72f0, 0.72f0, -0.72f0, 0.72f0),
            backgroundcolor = theme.panel,
            xgridcolor = (theme.border, 0.35),
            ygridcolor = (theme.border, 0.35),
            zgridcolor = (theme.border, 0.35),
            titlecolor = theme.text,
            xlabelcolor = theme.text,
            ylabelcolor = theme.text,
            zlabelcolor = theme.text,
            xticklabelcolor = theme.text_muted,
            yticklabelcolor = theme.text_muted,
            zticklabelcolor = theme.text_muted,
        )
        linesegments!(ax3, lift(g -> g.axis_segments, geom_obs);
                      color = (theme.text_muted, 0.32), linewidth = 1.2)
        linesegments!(ax3, lift(g -> g.box_segments, geom_obs);
                      color = theme.text, linewidth = 2.1)
        lines!(ax3, lift(g -> g.slice_loop, geom_obs);
               color = theme.selection, linewidth = 4)
        scatter!(ax3, lift(g -> g.selected_point, geom_obs);
                 color = theme.selection, markersize = 14,
                 strokecolor = theme.text, strokewidth = 1.0)

        controls = fig3[3, 1] = GridLayout(; alignmode = Outside(10, 10, 6, 6))
        colgap!(controls, 8)
        rowgap!(controls, 6)
        Box(controls[1:2, 1:10];
            color = theme.surface, strokecolor = theme.border,
            strokewidth = 1.0, cornerradius = 8, z = -5)
        Label(controls[1, 1]; text = "Axis", color = theme.text_muted,
              fontsize = 13, halign = :right)
        axis_x_box = Textbox(controls[1, 2]; placeholder = "x", width = 58, height = 30)
        axis_y_box = Textbox(controls[1, 3]; placeholder = "y", width = 58, height = 30)
        axis_z_box = Textbox(controls[1, 4]; placeholder = "z", width = 58, height = 30)
        Label(controls[1, 5]; text = "Angle", color = theme.text_muted,
              fontsize = 13, halign = :right)
        angle_box = Textbox(controls[1, 6]; placeholder = "deg", width = 74, height = 30)
        apply_btn = Button(controls[1, 7]; label = "Rotate", width = 82, height = 30)
        reset_btn = Button(controls[1, 8]; label = "Reset", width = 74, height = 30)
        fit_btn = Button(controls[1, 9]; label = MANTA_ICONS.fit, width = 42, height = 30)
        state_label = Label(controls[2, 1:10];
            text = "Drag in the 3D scene to orbit the camera; use axis + angle to rotate the cube frame.",
            color = theme.text_muted, fontsize = 13, halign = :left,
            tellwidth = false, padding = (12, 12, 0, 4))
        foreach(c -> colsize!(controls, c, Auto()), 1:10)
        rowsize!(controls, 1, Fixed(36))
        rowsize!(controls, 2, Fixed(32))

        for tb in (axis_x_box, axis_y_box, axis_z_box, angle_box)
            style_textbox!(tb)
        end
        style_button_primary!(apply_btn)
        style_button_ghost!(reset_btn)
        style_button_ghost!(fit_btn)
        set_box_text!(axis_x_box, "0")
        set_box_text!(axis_y_box, "0")
        set_box_text!(axis_z_box, "1")
        set_box_text!(angle_box, "15")

        parse_rot_value(tb, fallback::Float32) = begin
            txt = strip(get_box_str(tb))
            isempty(txt) && return fallback
            v = tryparse(Float32, txt)
            v === nothing ? fallback : v
        end

        on(apply_btn.clicks) do _
            axv = (
                parse_rot_value(axis_x_box, 0f0),
                parse_rot_value(axis_y_box, 0f0),
                parse_rot_value(axis_z_box, 1f0),
            )
            deg = parse_rot_value(angle_box, 0f0)
            try
                rot_obs[] = _cube3d_compose_rotation(rot_obs[], axv, deg)
                state_label.text[] = "Applied $(round(deg; digits = 3)) degrees around axis ($(axv[1]), $(axv[2]), $(axv[3]))."
                set_status!("3D cube rotated around arbitrary axis.")
            catch e
                msg = sprint(showerror, e)
                state_label.text[] = "Rotation failed: $(msg)"
                set_status!("Invalid 3D cube rotation axis.")
            end
        end

        on(reset_btn.clicks) do _
            rot_obs[] = _cube3d_identity_rotation()
            state_label.text[] = "Cube frame reset. Drag in the 3D scene to choose another camera angle."
            set_status!("3D cube rotation reset.")
        end

        on(fit_btn.clicks) do _
            limits!(ax3, -0.72f0, 0.72f0, -0.72f0, 0.72f0, -0.72f0, 0.72f0)
            set_status!("3D cube view fitted.")
        end

        rowsize!(fig3.layout, 1, Auto())
        rowsize!(fig3.layout, 2, Relative(1))
        rowsize!(fig3.layout, 3, Fixed(92))

        cube3d_fig_ref[] = fig3
        cube3d_alive_ref[] = true
        register_window_close!(fig3) do
            cube3d_alive_ref[] = false
            cube3d_fig_ref[] = nothing
        end
        display_fig && display(fig3)
        set_status!("3D cube window opened.")
        return fig3
    end

    header_fig_ref = Ref{Any}(nothing)
    header_alive_ref = Ref(false)

    function open_cube_header_window!()
        if header_alive_ref[] && header_fig_ref[] !== nothing
            set_status!("Header window already open.")
            return header_fig_ref[]
        end

        sources = [
            (label = "Cube A", name = fname_full, path = filepath, header = header),
        ]
        if compare_visible[]
            cmp_name = isempty(compare_name[]) ? "comparison cube" : compare_name[]
            push!(sources, (label = "Cube B", name = cmp_name,
                            path = compare_path_current[], header = compare_header_ref[]))
        end

        source_labels = [s.label for s in sources]
        lines_by_label = Dict{String,Vector{String}}(
            s.label => fits_header_display_lines(s.header) for s in sources)
        names_by_label = Dict{String,String}(
            s.label => (isempty(String(s.name)) ? s.label : String(s.name)) for s in sources)
        paths_by_label = Dict{String,String}(
            s.label => (isempty(String(s.path)) ? "(in-memory source)" : String(s.path)) for s in sources)

        page_lines = 36
        selected_label = Observable(first(source_labels))
        page_obs = Observable(1)
        backend_sym = pick_backend!(activate_gl)
        fig_hdr = Figure(size = (980, 760); backgroundcolor = ui_theme.background)

        Label(fig_hdr[1, 1]; text = "FITS header",
              halign = :center, tellwidth = false,
              fontsize = 20, color = ui_text,
              padding = (0, 0, 10, 6))

        controls = fig_hdr[2, 1] = GridLayout(; alignmode = Outside(8, 8, 4, 4))
        colgap!(controls, 8)
        Label(controls[1, 1]; text = "Source", halign = :right,
              fontsize = 13, color = ui_text_muted)
        source_menu = Menu(controls[1, 2]; options = source_labels,
                           prompt = first(source_labels), width = 120)
        prev_btn = Button(controls[1, 3]; label = "Prev", width = 74, height = 30)
        next_btn = Button(controls[1, 4]; label = "Next", width = 74, height = 30)
        page_label = Label(controls[1, 5]; text = "", halign = :left,
                           fontsize = 13, color = ui_text_muted, tellwidth = false)
        style_menu!(source_menu)
        style_button_ghost!(prev_btn)
        style_button_ghost!(next_btn)

        source_title = Label(fig_hdr[3, 1]; text = "", halign = :left,
                             tellwidth = false, fontsize = 14, font = :bold,
                             color = ui_text, padding = (14, 14, 6, 2))
        source_path = Label(fig_hdr[4, 1]; text = "", halign = :left,
                            tellwidth = false, fontsize = 12,
                            color = ui_text_muted, padding = (14, 14, 0, 6))

        body = fig_hdr[5, 1] = GridLayout(; alignmode = Outside(10, 10, 8, 8))
        Box(body[1, 1]; color = ui_surface, strokecolor = ui_border,
            strokewidth = 1.0, cornerradius = 8, z = -5)
        body_label = Label(body[1, 1]; text = "", halign = :left, valign = :top,
                           tellwidth = false, tellheight = false,
                           fontsize = 13, color = ui_text,
                           padding = (14, 14, 12, 12),
                           lineheight = 1.08)

        rowsize!(fig_hdr.layout, 1, Auto())
        rowsize!(fig_hdr.layout, 2, Auto())
        rowsize!(fig_hdr.layout, 3, Auto())
        rowsize!(fig_hdr.layout, 4, Auto())
        rowsize!(fig_hdr.layout, 5, Relative(1))

        function refresh_header_body!()
            label = selected_label[]
            lines = get(lines_by_label, label, ["(no FITS header available)"])
            n_pages = max(1, cld(length(lines), page_lines))
            pg = clamp(page_obs[], 1, n_pages)
            page_obs[] = pg
            first_line = (pg - 1) * page_lines + 1
            last_line = min(length(lines), first_line + page_lines - 1)
            body_label.text[] = join(lines[first_line:last_line], "\n")
            source_title.text[] = "$(label): $(get(names_by_label, label, label))"
            source_path.text[] = get(paths_by_label, label, "")
            page_label.text[] = "Page $(pg) / $(n_pages) · $(length(lines)) cards"
            nothing
        end

        on(source_menu.selection) do sel
            sel === nothing && return
            label = String(sel)
            haskey(lines_by_label, label) || return
            selected_label[] = label
            page_obs[] = 1
            refresh_header_body!()
        end
        on(prev_btn.clicks) do _
            page_obs[] = max(1, page_obs[] - 1)
            refresh_header_body!()
        end
        on(next_btn.clicks) do _
            label = selected_label[]
            n_pages = max(1, cld(length(lines_by_label[label]), page_lines))
            page_obs[] = min(n_pages, page_obs[] + 1)
            refresh_header_body!()
        end
        refresh_header_body!()

        header_fig_ref[] = fig_hdr
        header_alive_ref[] = true
        register_window_close!(fig_hdr) do
            header_alive_ref[] = false
            header_fig_ref[] = nothing
        end
        if display_fig
            if backend_sym === :GLMakie
                try
                    screen = GLMakie.Screen(; title = "MANTA FITS header",
                                             focus_on_show = true)
                    display(screen, fig_hdr)
                catch e
                    @warn "MANTA: failed to open header in dedicated GLMakie window, falling back to default display" exception = (e, catch_backtrace())
                    try
                        display(fig_hdr)
                    catch
                    end
                end
            else
                display(fig_hdr)
            end
        end
        set_status!("Header window opened.")
        return fig_hdr
    end

    on(header_btn.clicks) do _
        try
            open_cube_header_window!()
        catch e
            msg = "Failed to open FITS header: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception = (e, catch_backtrace())
        end
    end

    function world_info_string()
        any(has_wcs(wcs, dim) for dim in 1:3) || return ""
        coords = (
            format_world_coord(wcs, 1, i_idx[]),
            format_world_coord(wcs, 2, j_idx[]),
            format_world_coord(wcs, 3, k_idx[]),
        )
        return join(coords, ", ")
    end

    function selection_info_tex()
        if view_product[] === :rotproj
            u = clamp(u_idx[], 1, size(slice_proc[], 1))
            v = clamp(v_idx[], 1, size(slice_proc[], 2))
            val = slice_proc[][u, v]
            mode_label = String(rotation_projection_mode[])
            axis_label = axis[]
            return latexstring(
                "\\mathrm{rotated\\ projection}\\quad\\mathrm{axis}=\\mathbf{",
                axis_label,
                "}\\quad\\mathrm{mode}=\\mathbf{\\text{",
                latex_safe(mode_label),
                "}}\\quad\\mathrm{pixel}\\,(\\mathrm{row},\\mathrm{col})=\\mathbf{(",
                u,
                ",",
                v,
                ")}\\quad\\mathrm{value}=\\mathbf{",
                isnan(val) ? "NaN" : string(round(Float32(val); digits = 4)),
                "}",
            )
        end
        if isempty(region_uvs[])
            val = data[i_idx[], j_idx[], k_idx[]]
            # Inline reconstruction rather than re-concatenating an existing
            # LaTeXString wrapped in $...$: otherwise LaTeXStrings.latexstring
            # detects the leading/trailing $ and inserts the remainder outside
            # math mode, which crashes MathTeXEngine (\quad, \mathbf, \^{}
            # outside $$).
            info = string(
                "\\mathrm{pixel}\\,(i,j,k)=\\mathbf{(",
                i_idx[], ",", j_idx[], ",", k_idx[], ")}",
                "\\quad\\mathrm{slice}\\,(\\mathrm{row},\\mathrm{col})=\\mathbf{(",
                u_idx[], ",", v_idx[], ")}",
                "\\quad\\mathrm{intensity}=\\mathbf{",
                isnan(val) ? "NaN" : string(round(Float32(val); digits = 4)),
                "}",
            )
            winfo = world_info_string()
            isempty(winfo) && return latexstring(info)
            # Wrap the WCS block in \text{} so that text-mode escapes produced
            # by latex_safe (\^{}, \_, ...) are interpreted in the correct
            # context rather than being sent raw into math mode.
            return latexstring(info, "\\quad\\mathrm{WCS}\\,(\\text{", latex_safe(winfo), "})")
        else
            npx = length(region_uvs[])
            y = mean_region_spectrum(data, axis[], region_uvs[]; mask = mask_bits_obs[])
            chan = clamp(idx[], 1, length(y))
            val = y[chan]
            shape = region_shape[] === :circle ? "circle" : "box"
            return latexstring(
                "\\mathrm{region}\\,\\text{", shape, "}\\quad\\mathrm{pixels}=\\mathbf{",
                npx,
                "}\\quad\\mathrm{slice\\ mean}=\\mathbf{",
                isnan(val) ? "NaN" : string(round(Float32(val); digits = 4)),
                "}",
            )
        end
    end

    function clear_region!()
        region_uvs[] = Tuple{Int,Int}[]
        region_start[] = Point2f(NaN32, NaN32)
        region_end[] = Point2f(NaN32, NaN32)
        region_drag_active[] = false
        region_count_label.text[] = "0 px"
        nothing
    end

    function update_region_from_drag!(p0::Point2f, p1::Point2f)
        u_max, v_max = slice_dims(axis[])
        uv = region_uv_indices(u_max, v_max, p0[1], p0[2], p1[1], p1[2], region_shape[])
        region_uvs[] = uv
        region_count_label.text[] = "$(length(uv)) px"
        if isempty(uv)
            set_status!("Selection canceled: draw a larger $(String(region_shape[])).")
        else
            set_status!("Selection spectrum averaged over $(length(uv)) pixels.")
        end
        nothing
    end

    function apply_percentile_clims!(lo::Real, hi::Real; subsample::Integer = 1)
        parsed_clims = percentile_clims(slice_disp[], lo, hi; subsample = subsample)
        clims_manual[] = parsed_clims
        use_manual[] = true
        set_box_text!(clim_min_box, string(first(parsed_clims)))
        set_box_text!(clim_max_box, string(last(parsed_clims)))
        flag_box!(clim_min_box, true)
        flag_box!(clim_max_box, true)
        if spec_ylimits_source[] === :contrast
            spec_ylimits_value[] = parsed_clims
            set_box_text!(spec_ymin_box, string(first(parsed_clims)))
            set_box_text!(spec_ymax_box, string(last(parsed_clims)))
            flag_box!(spec_ymin_box, true)
            flag_box!(spec_ymax_box, true)
            refresh_spec_ylim!()
        end
        set_status!("Contrast set to p$(lo)-p$(hi).")
        nothing
    end

    function refresh_uv!()
        a = axis[]
        u_max, v_max = slice_dims(a)
        u, v = ijk_to_uv(i_idx[], j_idx[], k_idx[], a)
        u = clamp(u, 1, u_max)
        v = clamp(v, 1, v_max)
        u_idx[] = u; v_idx[] = v
        uv_point[] = Point2f(v, u)
    end

    function refresh_labels!()
        lab_info.text[] = selection_info_tex()
        status_label.text[] = latexstring("\\text{axis } $(axis[]),\\, \\text{index } $(idx[])")
    end

    function refresh_hist_axes!()
        xlo, xhi = hist_limits_obs[]
        if hist_ylimits_manual[]
            ylo, yhi = hist_ylimits_manual_value[]
            limits!(ax_hist, Float32(xlo), Float32(xhi), Float32(ylo), Float32(yhi))
        else
            autolimits!(ax_hist)
            xlims!(ax_hist, Float32(xlo), Float32(xhi))
        end
    end

    function refresh_axis_labels!()
        xlab, ylab = slice_axis_labels(axis[])
        ax_img.xlabel[] = xlab
        ax_img.ylabel[] = ylab
        ax_cmp.xlabel[] = xlab
        ax_cmp.ylabel[] = ylab
        u_dim, v_dim = slice_axis_dims(axis[])
        ax_img.xtickformat[] = pixel_world_tick_formatter(v_dim)
        ax_img.ytickformat[] = pixel_world_tick_formatter(u_dim)
        ax_cmp.xtickformat[] = pixel_world_tick_formatter(v_dim)
        ax_cmp.ytickformat[] = pixel_world_tick_formatter(u_dim)
    end

    refresh_all!() = (refresh_axis_labels!(); refresh_uv!(); refresh_labels!(); refresh_spectrum!())

    # ---------- Reactivity ----------
    on(clims_obs) do (cmin, cmax)
        if spec_ylimits_source[] === :contrast
            spec_ylimits_value[] = (Float32(cmin), Float32(cmax))
            refresh_spec_ylim!()
        end
    end

    on(spec_scale_mode) do _
        refresh_spec_ylim!()
    end

    reset_zoom!() = begin
        autolimits!(ax_img)
        compare_visible[] && autolimits!(ax_cmp)
        nothing
    end

    on_mode(reset_zoom_btn.clicks, :navigation) do _
        reset_zoom!()
    end
    on_mode(reset_zoom_analysis_btn.clicks, :analysis) do _
        reset_zoom!()
    end
    on_mode(reset_zoom_export_btn.clicks, :export) do _
        reset_zoom!()
    end
    on(focus_fit_btn.clicks) do _
        focus_image[] || return
        reset_zoom!()
        set_status!("Focus image: view fitted.")
    end

    # NOTE: UI callback registrations are in src/views/cube/UICallbacksBundle.jl
    # and are wired up below (after render_power_spectrum_layout! is defined).

    # ---------- Mask: parsing helpers + apply/reset ----------
    # See src/views/cube/MaskBundle.jl
    (; _parse_int_range, _set_mask_status!, apply_mask_source!, reset_mask!, build_mask_source_from_ui) =
        _cube_mask_bundle(;
            data, _moment_cache,
            mask_source_obs, mask_bits_obs, mask_status_obs,
            mask_status_label,
            mask_lo_box, mask_hi_box, mask_i_box, mask_j_box, mask_k_box,
            mask_source_menu, mask_op_menu,
            refresh_spectrum!, set_status!, set_box_text!,
            ui_mask, ui_error, ui_text_muted,
        )

    on_mode(mask_apply_btn.clicks, :analysis) do _
        src, err = build_mask_source_from_ui()
        if src === nothing
            mask_status_label.color[] = ui_error
            mask_status_label.text[] = err
            set_status!("Mask not applied: $(err)")
            return
        end
        try
            apply_mask_source!(src)
            if src isa NoMaskSource
                set_status!("Mask cleared (source = none).")
            else
                set_status!("Mask applied. " * mask_status_obs[])
            end
        catch e
            msg = "Failed to build mask: $(sprint(showerror, e))"
            mask_status_label.color[] = ui_error
            mask_status_label.text[] = msg
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
        end
    end

    on_mode(mask_reset_btn.clicks, :analysis) do _
        reset_mask!()
    end

    on_mode(moment_menu.selection, :analysis) do sel
        sel === nothing && return
        label = String(sel)
        moment_order[] = startswith(label, "M1") ? 1 : startswith(label, "M2") ? 2 : 0
        view_product[] === :moment && set_status!("Showing $(label) along axis $(axis[]).")
    end

    on_mode(btn_show_moment.clicks, :analysis) do _
        view_product[] = :moment
        use_manual[] = false
        autolimits!(ax_spec)
        xlims!(ax_spec, 0f0, Float32(max(0, length(spec_y_buf) - 1)))
        autolimits!(ax_img)
        set_status!("Moment map displayed along axis $(axis[]).")
    end

    on_mode(btn_show_slice.clicks, :analysis) do _
        view_product[] = :slice
        use_manual[] = false
        autolimits!(ax_img)
        set_status!("Slice view restored.")
    end
    on(focus_auto_btn.clicks) do _
        focus_image[] || return
        apply_percentile_clims!(1, 99)
    end

    # ---------- Embedded power-spectrum layout ----------
    ps_layout_src = Observable(:zoom)      # :zoom | :full
    ps_layout_window = Observable(:hann)   # :hann | :hamming | :none
    ps_layout_pad = Observable(false)
    ps_layout_nanapo = Observable(false)
    ps_layout_units = Observable(:pixel)   # :pixel | :physical
    ps_layout_fit = Observable(false)
    ps_layout_blocks = Any[]

    function set_embedded_ps_visible!(visible::Bool)
        for block in ps_ui_blocks
            set_block_visible!(block, visible && !compact_layout)
        end
        for block in ps_layout_blocks
            set_block_visible!(block, visible)
        end
        nothing
    end
    set_embedded_ps_visible!(false)
    if compact_layout
        rowsize!(spec_grid, 1, Fixed(0))
        set_block_visible!(info_box, false)
        set_block_visible!(lab_info, false)
    end

    ps_u_dim_now() = slice_axis_dims(axis[])[1]
    ps_v_dim_now() = slice_axis_dims(axis[])[2]
    ps_physical_available() = has_wcs(wcs, ps_u_dim_now()) && has_wcs(wcs, ps_v_dim_now())
    ps_pixel_scales() = begin
        if ps_physical_available()
            dy = abs(wcs[ps_u_dim_now()].cdelt)
            dx = abs(wcs[ps_v_dim_now()].cdelt)
            (dx, dy)
        else
            (1.0, 1.0)
        end
    end
    ps_physical_unit_label() = begin
        if ps_physical_available()
            u = wcs[ps_v_dim_now()].cunit
            isempty(u) ? "1" : u
        else
            ""
        end
    end

    function ps_layout_clear!()
        live_blocks = Any[]
        append!(live_blocks, ps_layout_blocks)
        try
            append!(live_blocks, contents(ps_plot_grid))
        catch
        end
        for b in unique(live_blocks)
            set_block_visible!(b, false)
            try
                Makie.delete!(b)
            catch
            end
        end
        empty!(ps_layout_blocks)
        nothing
    end

    function ps_layout_subimage(M::AbstractMatrix)
        if ps_layout_src[] === :full
            return M
        end
        fl = ax_img.finallimits[]
        x0 = Float64(fl.origin[1])
        y0 = Float64(fl.origin[2])
        x1 = x0 + Float64(fl.widths[1])
        y1 = y0 + Float64(fl.widths[2])
        i_lo = clamp(Int(floor(min(x0, x1))), 1, size(M, 1))
        i_hi = clamp(Int(ceil(max(x0, x1))),  1, size(M, 1))
        j_lo = clamp(Int(floor(min(y0, y1))), 1, size(M, 2))
        j_hi = clamp(Int(ceil(max(y0, y1))),  1, size(M, 2))
        (i_hi <= i_lo || j_hi <= j_lo) && return M
        return M[i_lo:i_hi, j_lo:j_hi]
    end
    ps_layout_subimage() = ps_layout_subimage(slice_proc[])

    function ps_layout_status_text(meta)
        io = IOBuffer()
        print(io, "size $(meta.ny_in)×$(meta.nx_in)")
        if meta.padded
            print(io, " (pad→$(meta.ny_eff)×$(meta.nx_eff))")
        end
        print(io, " • $(meta.src) • ")
        print(io, meta.window === :none ? "none" : titlecase(String(meta.window)))
        meta.apodized && print(io, " • NaN apod")
        meta.f_sky < 1.0 && print(io, " • f_sky=$(round(meta.f_sky; digits = 3))")
        print(io, " • k=", meta.k_phys ? "1/$(ps_physical_unit_label())" : "cycles/pixel")
        return String(take!(io))
    end

    function render_power_spectrum_layout!()
        ps_layout_clear!()
        sub = ps_layout_subimage()
        ny0, nx0 = size(sub)
        if ny0 < 4 || nx0 < 4
            lab = Label(ps_plot_grid[1, 1]; text = "Selection too small for FFT (need ≥ 4×4).", fontsize = 14, color = ui_text)
            push!(ps_layout_blocks, lab)
            ps_layout_status[] = " "
            return
        end

        src_label = ps_layout_src[] === :full ? "full" : "zoom"
        use_phys = ps_layout_units[] === :physical && ps_physical_available()
        dx, dy = ps_pixel_scales()
        k_unit_lbl = use_phys ? "1/$(ps_physical_unit_label())" : "cycles/pixel"
        products = [("A", sub, ui_accent)]
        if compare_visible[]
            push!(products, ("B", ps_layout_subimage(compare_slice_proc[]), ui_compare))
        end
        n_products = length(products)

        # ---- Dynamic sizes for the power-spectrum plots row ----
        single_ps_product = n_products == 1 && !compact_layout
        ps_inner_gap = compact_layout ? 6 : 12
        # Per-axis height overhead: title + x-label + x-tick-labels + top/bottom padding.
        # These decorations are NOT included in `height=` on an Axis, so they must be
        # subtracted from the row budget before allocating raw axis-content heights;
        # otherwise the two stacked axes overflow into the controls row below.
        axis_deco_h = compact_layout ? 52 : 64   # empirical: ~52–64 px per axis block
        ps_struct    = (compact_layout ? 4 : 8) * 2 +   # 2× alignmode Outside()
                       (compact_layout ? 0 : 8) +         # header-to-plot rowgap (0 in compact: ps_header_height = 0)
                       (single_ps_product ? axis_deco_h :
                                            2 * axis_deco_h + ps_inner_gap)
        ps_avail = max(compact_layout ? 140 : 160,
                       ps_plot_row_height - ps_header_height - ps_struct)
        ps_right_w = max(360, fig_size[1] ÷ 2 - (single_ps_product ? 40 : 80))
        if single_ps_product
            heatmap_box_h = clamp(ps_avail, 190, ps_axis_size)
            ps_box_w = heatmap_box_h
            psd_box_w = clamp(ps_right_w - ps_box_w - 54, 280, ps_axis_size)
            psd_box_h = heatmap_box_h
        else
            raw_heatmap_h = clamp(round(Int, ps_avail * 0.60), compact_layout ? 80 : 100, ps_axis_size)
            raw_box_w = n_products == 1 ? raw_heatmap_h :
                        floor(Int, (ps_right_w - 28 * n_products - 18 * (n_products - 1)) / n_products)
            ps_box_w = clamp(raw_box_w, compact_layout ? 145 : 170, ps_axis_size)
            heatmap_box_h = min(raw_heatmap_h, ps_box_w)
            psd_box_w = ps_box_w
            psd_box_h = max(compact_layout ? 60 : 80, ps_avail - heatmap_box_h - ps_inner_gap)
        end

        colgap!(ps_plot_grid, single_ps_product ? 14 : n_products == 1 ? -8 : (compact_layout ? 8 : 14))
        for c in 1:4
            colsize!(ps_plot_grid, c, Auto())
        end
        if single_ps_product
            colsize!(ps_plot_grid, 2, Fixed(26))
            colsize!(ps_plot_grid, 4, Fixed(0))
            rowsize!(ps_plot_grid, 2, Fixed(0))
        elseif n_products == 1
            colsize!(ps_plot_grid, 3, Fixed(0))
            colsize!(ps_plot_grid, 4, Fixed(0))
            rowsize!(ps_plot_grid, 2, Auto())
        else
            colsize!(ps_plot_grid, 2, Fixed(26))
            colsize!(ps_plot_grid, 4, Fixed(26))
            rowsize!(ps_plot_grid, 2, Auto())
        end

        first_meta = nothing
        fit_chunks = String[]
        for (ip, (label, img, line_color)) in enumerate(products)
            grid_col = 2ip - 1
            cb_col = 2ip

            # why: factored computation shared with the pop-out window's ps_render!
            bundle = _cube_ps_bundle(img;
                                     window = ps_layout_window[],
                                     pad_pow2 = ps_layout_pad[],
                                     apodize_nan = ps_layout_nanapo[],
                                     use_phys = use_phys, dx = dx, dy = dy)
            meta = (; bundle.meta..., k_phys = use_phys, src = src_label)
            first_meta === nothing && (first_meta = meta)

            # ---- 2D power spectrum heatmap (row 1) ----
            ax2d = Axis(
                ps_plot_grid[1, grid_col];
                title = latexstring("\\text{", latex_safe(label), " 2D power spectrum (log10) — ", latex_safe(src_label), "}"),
                xlabel = use_phys ?
                    latexstring("k_x\\;(", latex_safe(k_unit_lbl), ")") :
                    L"k_x\;\text{(cycles/pixel)}",
                ylabel = use_phys ?
                    latexstring("k_y\\;(", latex_safe(k_unit_lbl), ")") :
                    L"k_y\;\text{(cycles/pixel)}",
                aspect = DataAspect(),
                width = ps_box_w,
                height = heatmap_box_h,
                halign = :center,
                valign = :top,
                xtickformat = latex_tick_formatter,
                ytickformat = latex_tick_formatter,
            )
            hm_ps = heatmap!(ax2d, bundle.kx, bundle.ky, bundle.P2d_log10; colormap = ps_cm_obs[])
            cb = Colorbar(
                ps_plot_grid[1, cb_col],
                hm_ps;
                label = L"\log_{10}|F|^2",
                width = 18,
                height = _axis_render_height(ax2d),
                tellheight = false,
                valign = :top,
            )
            push!(ps_layout_blocks, ax2d)
            push!(ps_layout_blocks, cb)

            # ---- 1D radial PSD (log–log) just below the heatmap ----
            prof = bundle.prof
            k = bundle.k
            p_floored = bundle.prof_floored
            # log–log : on retire le DC (k = 0) et tout k ≤ 0 (cf. convention :log10).
            pos_mask = k .> 0
            k_pos = k[pos_mask]
            p_pos = p_floored[pos_mask]

            ax1d = Axis(
                ps_plot_grid[single_ps_product ? 1 : 2,
                             single_ps_product ? 3 : grid_col];
                title = latexstring("\\text{", latex_safe(label), " 1D radial power spectrum (log–log) — ", latex_safe(src_label), "}"),
                xlabel = use_phys ?
                    latexstring("k\\;(", latex_safe(k_unit_lbl), ")") :
                    L"k\;\text{(cycles/pixel)}",
                ylabel = L"\langle|F|^2\rangle",
                xscale = log10,
                yscale = log10,
                width = psd_box_w,
                height = psd_box_h,
                halign = :center,
                valign = :top,
                xtickformat = latex_tick_formatter,
            )
            isempty(k_pos) || lines!(ax1d, k_pos, p_pos; color = line_color, linewidth = PS_LINE_LW)
            guard_log_zoom!(ax1d)   # log–log: keep scroll-zoom from inverting limits
            push!(ps_layout_blocks, ax1d)

            if ps_layout_fit[] && length(k) >= 3
                valid_k = filter(>(0), k)
                auto_lo = isempty(valid_k) ? 0.0 : Float64(first(valid_k))
                auto_hi = isempty(k) ? Inf : Float64(last(k))
                kmin_txt = get_box_str(ps_kmin_box)
                kmax_txt = get_box_str(ps_kmax_box)
                kmin_v = isempty(kmin_txt) ? auto_lo : something(tryparse(Float64, kmin_txt), auto_lo)
                kmax_v = isempty(kmax_txt) ? auto_hi : something(tryparse(Float64, kmax_txt), auto_hi)
                k_guides = Float64[]
                isfinite(kmin_v) && kmin_v > 0 && push!(k_guides, kmin_v)
                isfinite(kmax_v) && kmax_v > 0 && kmax_v != kmin_v && push!(k_guides, kmax_v)
                isempty(k_guides) || vlines!(ax1d, k_guides; color = ui_error, linestyle = :dot, linewidth = 1.5)
                slope, intercept, n_used = fit_loglog_slope(k, prof; kmin = kmin_v, kmax = kmax_v)
                if isfinite(slope) && n_used >= 2
                    kfit = filter(ki -> ki > 0 && ki >= kmin_v && ki <= kmax_v, k)
                    if !isempty(kfit)
                        yfit = Float32.(10 .^ (slope .* log10.(Float64.(kfit)) .+ intercept))
                        lines!(ax1d, kfit, yfit; color = ui_error, linestyle = :dash, linewidth = 1.5)
                        push!(fit_chunks, "$(label) slope=$(round(slope; digits = 3)) [n=$(n_used)]")
                    end
                end
            end
        end

        fit_status = isempty(fit_chunks) ? "" : " • " * join(fit_chunks, " • ")
        compare_status = compare_visible[] ? " • A+B comparison" : ""
        ps_layout_status[] = ps_layout_status_text(first_meta) * compare_status * fit_status
        nothing
    end

    function apply_layout_mode!()
        if layout_mode[] === :power_spectrum
            colsize!(main_grid, 1, Relative(1 / 2))
            colsize!(main_grid, 2, Relative(1 / 2))
            rowsize!(main_grid, 1, Fixed(ps_plot_row_height))
            rowsize!(main_grid, 2, Fixed(controls_height))
            rowsize!(spec_grid, 1, Fixed(0))
            rowsize!(spec_grid, 2, Fixed(0))
            rowsize!(spec_grid, 3, Fixed(0))
            refresh_control_mode!()
            set_block_visible!(ax_img, true)
            set_compare_panel_visible!(compare_visible[])
            set_block_visible!(img_colorbar, true)
            set_block_visible!(info_box, false)
            set_block_visible!(lab_info, false)
            set_block_visible!(ax_spec, false)
            set_block_visible!(ax_hist, false)
            _set_legend_visible!(false)   # spec_legend must not float over PS panels
            set_embedded_ps_visible!(true)
            render_power_spectrum_layout!()
            set_status!("Power spectrum layout enabled.")
        else
            rowsize!(spec_grid, 1, compact_layout ? Fixed(0) : Auto())
            rowsize!(spec_grid, 2, Auto())
            rowsize!(spec_grid, 3, Auto())
            colsize!(main_grid, 1, Auto())
            colsize!(main_grid, 2, Auto())
            rowsize!(main_grid, 1, compact_layout ? Fixed(plot_row_height) : Auto())
            rowsize!(main_grid, 2, Fixed(controls_height))
            refresh_control_mode!()
            set_block_visible!(ax_img, true)
            set_compare_panel_visible!(compare_visible[])
            set_block_visible!(img_colorbar, true)
            set_block_visible!(info_box, !compact_layout)
            set_block_visible!(lab_info, !compact_layout)
            set_block_visible!(ax_spec, true)
            set_block_visible!(ax_hist, true)
            _set_legend_visible!(compare_visible[])  # restore legend to its normal gate
            ps_layout_clear!()
            set_embedded_ps_visible!(false)
            set_status!("Base layout restored.")
        end
        nothing
    end

    function focus_axis_size()
        u_max, v_max = slice_dims(axis[])
        aspect = max(Float64(v_max), 1.0) / max(Float64(u_max), 1.0)
        n_panels = compare_visible[] ? 2 : 1

        # Leave room for colorbars, inter-panel gap, axis labels/title, the
        # cube overview strip and the small Focus toolbar.
        panel_gap = compare_visible[] ? (compact_layout ? 16 : 28) : 0
        cbar_pad = n_panels * (compact_layout ? 34 : 42)
        usable_w = max(260, (fig_size[1] - (compact_layout ? 56 : 80) - cbar_pad - panel_gap) ÷ n_panels)
        usable_h = max(260, fig_size[2] - (compact_layout ? 145 : 170))

        w_from_h = floor(Int, usable_h * aspect)
        if w_from_h <= usable_w
            return (max(260, w_from_h), usable_h)
        else
            return (usable_w, max(260, floor(Int, usable_w / aspect)))
        end
    end

    function apply_focus_axis_size!()
        w, h = focus_axis_size()
        ax_img.width[] = w
        ax_img.height[] = h
        ax_cmp.width[] = w
        ax_cmp.height[] = h
        nothing
    end

    function restore_normal_axis_size!()
        ax_img.width[] = normal_axis_size.img_w
        ax_img.height[] = normal_axis_size.img_h
        ax_cmp.width[] = normal_axis_size.cmp_w
        ax_cmp.height[] = normal_axis_size.cmp_h
        nothing
    end

    function apply_focus_image_mode!()
        if focus_image[]
            ps_layout_clear!()
            set_embedded_ps_visible!(false)
            set_layout_contents_visible!(controls_grid, false)
            set_layout_contents_visible!(spec_grid, false)
            set_layout_contents_visible!(focus_bar, true)
            set_block_visible!(hint_label, false)
            set_block_visible!(status_footer_label, false)
            rowsize!(main_grid, 1, Relative(1))
            rowsize!(main_grid, 2, Fixed(0))
            rowsize!(main_grid, 3, Fixed(0))
            rowsize!(main_grid, 4, Fixed(0))
            rowgap!(main_grid, 0)
            colsize!(main_grid, 1, Relative(1))
            colsize!(main_grid, 2, Fixed(0))
            apply_focus_axis_size!()
            set_block_visible!(ax_img, true)
            set_compare_panel_visible!(compare_visible[])
            set_block_visible!(img_colorbar, true)
            set_block_visible!(info_box, false)
            set_block_visible!(lab_info, false)
            set_block_visible!(ax_spec, false)
            set_block_visible!(ax_hist, false)
            _set_legend_visible!(false)
            set_status!("Focus image: controls hidden. Press f or Exit to return.")
        else
            set_layout_contents_visible!(focus_bar, false)
            rowgap!(main_grid, main_row_gap)
            rowsize!(main_grid, 3, Auto())
            rowsize!(main_grid, 4, Auto())
            restore_normal_axis_size!()
            apply_layout_mode!()
            set_layout_contents_visible!(controls_grid, true)
            refresh_control_mode!()
            set_block_visible!(hint_label, !compact_layout)
            set_block_visible!(status_footer_label, !compact_layout)
            set_status!("Focus image closed.")
        end
        nothing
    end
    on(layout_mode) do _
        apply_layout_mode!()
        focus_image[] && apply_focus_image_mode!()
    end

    on(focus_image) do _
        apply_focus_image_mode!()
    end

    on(axis) do _
        focus_image[] && apply_focus_axis_size!()
    end

    on(focus_btn.clicks) do _
        focus_image[] = true
    end

    on(focus_exit_btn.clicks) do _
        focus_image[] = false
    end

    toggle_focus_image!() = begin
        focus_image[] = !focus_image[]
        nothing
    end

    on_mode(ps_btn.clicks, :navigation) do _
        layout_mode[] = :power_spectrum
    end

    on_mode(base_layout_btn.clicks, :navigation) do _
        layout_mode[] = :base
    end

    on(ps_src_menu.selection) do sel
        sel === nothing && return
        ps_layout_src[] = sel == "full" ? :full : :zoom
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    on(ps_win_menu.selection) do sel
        sel === nothing && return
        ps_layout_window[] = sel == "Hamming" ? :hamming : sel == "None" ? :none : :hann
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    on(ps_unit_menu.selection) do sel
        sel === nothing && return
        ps_layout_units[] = sel == "physical" ? :physical : :pixel
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    on(ps_cmap_menu.selection) do sel
        sel === nothing && return
        ps_cmap_name[] = Symbol(String(sel))
    end

    on(ps_cmap_name) do _
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    on(ps_pad_chk.checked) do v
        ps_layout_pad[] = v
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    on(ps_nanapo_chk.checked) do v
        ps_layout_nanapo[] = v
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    on(ps_fit_btn.clicks) do _
        ps_layout_fit[] = true
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
        set_status!("Power spectrum fit applied over k_min/k_max.")
    end

    on(ps_clear_fit_btn.clicks) do _
        ps_layout_fit[] = false
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
        set_status!("Power spectrum fit cleared.")
    end

    on(ps_kmin_box.stored_string) do _
        layout_mode[] === :power_spectrum && ps_layout_fit[] && render_power_spectrum_layout!()
    end

    on(ps_kmax_box.stored_string) do _
        layout_mode[] === :power_spectrum && ps_layout_fit[] && render_power_spectrum_layout!()
    end

    on(ps_refresh_btn.clicks) do _
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    on(compare_visible) do v
        set_compare_panel_visible!(v)
        layout_mode[] === :power_spectrum && set_block_visible!(img_colorbar, true)
        focus_image[] && apply_focus_axis_size!()
        # Always refresh the control panel so the Index-B slider appears / disappears
        # immediately when a comparison cube is loaded or unloaded, without requiring
        # the user to manually switch mode tabs.
        refresh_control_mode!()
        if layout_mode[] === :power_spectrum
            render_power_spectrum_layout!()
        end
    end

    on(slice_proc) do _
        layout_mode[] === :power_spectrum && render_power_spectrum_layout!()
    end

    # ---------- UI callbacks ----------
    # See src/views/cube/UICallbacksBundle.jl
    # Called here (after render_power_spectrum_layout! is defined) so the
    # axis-menu and slice-slider callbacks can invoke it safely.
    syncing_slice_controls = _cube_ui_callbacks_bundle!(;
        control_mode, on_mode, bypass_mode_gate,
        refresh_control_mode!, set_status!, set_box_text!, flag_box!,
        mode_nav_btn, mode_analysis_btn, mode_export_btn,
        axis, idx, compare_idx, siz, axes_labels, axis_menu, slice_slider,
        compare_slice_slider, compare_slice_label, compare_visible,
        uv_to_ijk,
        u_idx, v_idx, i_idx, j_idx, k_idx,
        clear_region!, refresh_all!, refresh_labels!,
        render_power_spectrum_layout!, layout_mode,
        view_product, data, slice_plot_preview,
        img_scale_mode, spec_scale_mode, cmap_name,
        img_scale_menu, spec_scale_menu, cmap_menu,
        hist_mode_obs, hist_bins_obs,
        hist_xlimits_manual, hist_xlimits_manual_value,
        hist_ylimits_manual, hist_ylimits_manual_value,
        hist_limits_obs, hist_y_obs, compare_hist_y_obs,
        hist_mode_menu, hist_apply_btn, hist_auto_btn,
        hist_y_apply_btn, hist_y_auto_btn,
        hist_bins_box, hist_xmin_box, hist_xmax_box,
        hist_ymin_box, hist_ymax_box,
        refresh_hist_axes!,
        spec_ylimits_source, spec_ylimits_value,
        spec_y_apply_btn, spec_y_auto_btn,
        spec_ymin_box, spec_ymax_box,
        refresh_spec_ylim!,
        invert_cmap, gauss_on, show_crosshair, show_marker, show_grid,
        invert_chk, gauss_chk, crosshair_chk, marker_chk, grid_chk,
        ax_img, ax_cmp, ax_spec,
        refresh_spectrum!,
        compare_mode, compare_path_box,
        btn_show_compare, btn_load_compare, compare_mode_menu,
        pick_compare_path, load_compare_cube!, show_compare_loader!,
        sigma, sigma_label, sigma_slider,
        use_manual, clims_manual, clims_safe,
        clim_apply_btn, clim_auto_btn, clim_fix_btn, clim_auto_nav_btn, clim_p1_btn, clim_p5_btn,
        clim_min_box, clim_max_box,
        apply_percentile_clims!,
        selection_mode, region_shape,
        region_mode_menu, region_clear_btn,
        show_contours, contour_use_manual,
        contour_manual_levels, contour_manual_colors,
        contour_chk, contour_apply_btn, contour_levels_box,
        asinh_softening,
    )

    # ---------- Keyboard shortcuts + mouse pick ----------
    # See src/views/cube/KeyboardBundle.jl
    _cube_keyboard_bundle!(fig, ax_img;
        axis, idx, siz, u_idx, v_idx, i_idx, j_idx, k_idx, uv_point,
        slice_dims, uv_to_ijk,
        invert_cmap, img_scale_mode, show_contours, compare_visible, layout_mode,
        zoom_drag_active, zoom_drag_start, zoom_drag_end,
        region_drag_active, region_start, region_end, region_uvs, selection_mode, anim_playing,
        slice_slider, axis_menu, axes_labels, img_scale_menu, invert_chk, contour_chk,
        clim_auto_btn, btn_save_img, help_btn, region_count_label, ax_cmp,
        refresh_labels!, refresh_spectrum!, apply_percentile_clims!,
        render_power_spectrum_layout!, reset_zoom!, clear_region!, update_region_from_drag!,
        toggle_dark_mode!, toggle_focus_image!, set_status!, bypass_mode_gate, syncing_slice_controls,
        textboxes = (clim_min_box, clim_max_box, fname_box, compare_path_box,
                     spec_ymin_box, spec_ymax_box, contour_levels_box,
                     hist_bins_box, hist_xmin_box, hist_xmax_box,
                     hist_ymin_box, hist_ymax_box,
                     start_box, stop_box, step_box, fps_box,
                     mask_lo_box, mask_hi_box, ps_kmin_box, ps_kmax_box),
        ui_theme = ui_theme_ref,
    )

    # ---------- Live cursor readout (DS9 / CARTA style) ----------
    # Pixel value + WCS world coordinates under the pointer, pushed into the
    # shared status footer on every move. Mouse picking and the crosshair use
    # native cube pixel coordinates (x = v, y = u), so the rounded data-space
    # position indexes the raw slice directly. Skipped while a drag / playback
    # is in progress so transient drag messages are not clobbered.
    function _cursor_readout_str(p, u_max::Integer, v_max::Integer)
        u = clamp(round(Int, p[2]), 1, u_max)
        v = clamp(round(Int, p[1]), 1, v_max)
        s = slice_raw[]
        val = (1 <= u <= size(s, 1) && 1 <= v <= size(s, 2)) ? (@inbounds s[u, v]) : NaN32
        u_dim, v_dim = slice_axis_dims(axis[])
        # Display order: x (= v), y (= u), then the slice-plane channel. The
        # sky pair (u_dim, v_dim) is deprojected exactly when a WCSTransform is
        # available; the spectral/slice axis stays linear.
        world = cursor_world_strings(wcs, wcs_xform,
                                     [v_dim, u_dim, axis[]], [v, u, idx[]])
        return format_cursor_readout(["x" => v, "y" => u], val, unit_label, world)
    end
    on(events(ax_img).mouseposition) do _
        (zoom_drag_active[] || region_drag_active[] || anim_playing[]) && return
        u_max, v_max = slice_dims(axis[])
        if Makie.is_mouseinside(ax_img.scene)
            p = mouseposition(ax_img)
        elseif compare_visible[] && Makie.is_mouseinside(ax_cmp.scene)
            p = mouseposition(ax_cmp)
        else
            return
        end
        (isfinite(p[1]) && isfinite(p[2])) || return
        set_status!(_cursor_readout_str(p, u_max, v_max))
    end

    # ---------- Ctrl+Z / Ctrl+Y keyboard shortcuts for undo/redo ----------
    # Note: `begin...end` blocks cannot appear directly inside typed array literals
    # `T[...]` (Julia parses `begin` as the array-start index sentinel there).
    # Extract the handlers into named closures first, then build the binding list.
    _undo_key_handler = function ()
        snap = undo!(_undo_stack)
        snap === nothing && (set_status!("Nothing to undo."); return)
        _apply_undo_snap!(snap); set_status!("Undo.")
    end
    _redo_key_handler = function ()
        snap = redo!(_undo_stack)
        snap === nothing && (set_status!("Nothing to redo."); return)
        _apply_undo_snap!(snap); set_status!("Redo.")
    end
    register_shortcuts!(fig, ShortcutBinding[
        ShortcutBinding(Keyboard.z, _undo_key_handler; description = "undo", modifier = :ctrl),
        ShortcutBinding(Keyboard.y, _redo_key_handler; description = "redo", modifier = :ctrl),
    ]; textboxes = (clim_min_box, clim_max_box, fname_box, compare_path_box,
                    spec_ymin_box, spec_ymax_box, contour_levels_box,
                    hist_bins_box, hist_xmin_box, hist_xmax_box,
                    hist_ymin_box, hist_ymax_box,
                    start_box, stop_box, step_box, fps_box,
                    mask_lo_box, mask_hi_box, ps_kmin_box, ps_kmax_box))

    # ---------- Settings: save_root, make_name, current_settings, apply_inline_state! ----------
    # See src/views/cube/SettingsBundle.jl
    (; save_root, resolved_settings_path, make_name,
       current_settings, apply_inline_state!, current_recipe_snippet) =
        _cube_settings_bundle!(;
            filepath, fname,
            save_dir, settings_path,
            axis, idx, compare_idx, siz,
            img_scale_mode, spec_scale_mode, cmap_name,
            invert_cmap, show_crosshair, show_marker, show_grid,
            show_contours, contour_use_manual,
            contour_manual_levels, contour_manual_colors,
            use_manual, clims_manual, clims_auto,
            mask_source_obs, compare_visible, compare_path_current,
            spec_ylimits_source, spec_ylimits_value, spec_y_buf,
            refresh_spec_ylim!,
            i_idx, j_idx, k_idx,
            ax_spec,
            axes_labels, axis_menu, slice_slider, compare_slice_slider,
            img_scale_menu, spec_scale_menu, cmap_menu,
            invert_chk, crosshair_chk, marker_chk, grid_chk,
            contour_chk, contour_levels_box,
            clim_min_box, clim_max_box, spec_ymin_box, spec_ymax_box,
            mask_source_menu, mask_op_menu,
            mask_lo_box, mask_hi_box, mask_i_box, mask_j_box, mask_k_box,
            btn_save_state, btn_copy_code, btn_load_state,
            control_mode, bypass_mode_gate, on_mode,
            apply_mask_source!, set_status!, set_box_text!,
            ui_success,
        )

    # ---------- Export: write_fits_array, save_moment_png!, export_fits_product!,
    # analysis/export mode callbacks (save image, spectrum, animation, GIF)
    # See src/views/cube/ExportBundle.jl
    _cube_export_bundle!(;
        ds, header, data, fname, save_root, make_name,
        axis, idx, siz, moment_order, sigma,
        moment_raw, slice_disp, slice_proc, cm_obs, clims_obs,
        contour_levels_obs, contour_colors_obs,
        region_uvs, mask_bits_obs, region_start, region_end, region_shape,
        u_idx, v_idx, uv_point, show_crosshair, show_marker, show_grid, show_contours,
        spec_x_raw, spec_y_disp, i_idx, j_idx, k_idx, spec_y_buf,
        unit_label, unit_label_tex, slice_axis_labels, slice_dims,
        ui_accent, ui_selection,
        anim_playing,
        btn_moment_png, btn_moment_fits, btn_save_fits, fits_product_menu,
        btn_save_img, btn_save_spec, play_btn, anim_btn,
        fmt_menu, fname_box, start_box, stop_box, step_box, fps_box,
        pingpong_chk, loop_chk, slice_slider, ax_spec,
        region_segments_from_points,
        on_mode, set_status!,
    )

    # ---------- Power spectrum window ----------
    ps_fig_ref = Ref{Any}(nothing)
    ps_alive_ref = Ref(false)

    # See src/views/cube/PSWindowBundle.jl
    (; open_power_spectrum_window!) = _cube_ps_window_bundle(;
        ps_fig_ref, ps_alive_ref,
        slice_proc, wcs, axis, slice_axis_dims,
        save_root, fname, fname_box, make_name,
        ui_text, ui_text_muted, ui_accent, ui_error,
        style_menu!, style_button!, style_button_primary!, style_button_ghost!, style_checkbox!, style_textbox!,
        set_status!, ps_cmap_name,
    )

    on(ps_popout_btn.clicks) do _
        try
            open_power_spectrum_window!()
        catch e
            msg = "Failed to open power spectrum: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
        end
    end

    # ---------- Init ----------
    if state !== nothing
        try
            apply_inline_state!(state; announce = false)
        catch e
            @warn "Failed to apply inline MANTA state" exception=(e, catch_backtrace())
        end
    end
    refresh_all!()
    refresh_hist_axes!()
    # Optional preload of a comparison cube passed via the `compare=` kwarg
    # (used by scripted invocations and headless tests). Silently no-ops if
    # the path can't be loaded — `load_compare_cube!` already logs through
    # `set_status!` so we don't double-report here.
    if compare !== nothing
        try
            load_compare_cube!(String(compare))
        catch e
            @warn "Failed to preload comparison cube" path=compare exception=e
        end
    end
    register_window_close!(fig) do
        anim_playing[] = false
    end
    enable_file_drop!(fig; activate_gl = activate_gl, display_fig = display_fig)
    if display_fig
        display(fig)
    end
    return fig
end

# Public alias for the dataset-driven cube viewer.
const view_cube = _view_cube
