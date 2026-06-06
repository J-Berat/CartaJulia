# path: src/views/cube/SlicePipelineBundle.jl
#
# Reactive data pipeline for the 3D cube viewer.
#
# Extracted from src/views/CubeView.jl during a structural refactor
# (no behaviour change). Contains all the Makie `lift` chains that turn
# raw cube data + user-controlled Observables into display-ready arrays:
#
#   slice_raw            — unprocessed 2-D slice (non-copying view; see note below)
#   compare_slice_raw    — unprocessed 2-D slice from the comparison cube (view)
#   base_slice_proc      — Gaussian-smoothed slice for dual-view A-side
#   compare_slice_proc   — Gaussian-smoothed comparison slice
#   compare_product_proc — A/B blended product (diff, ratio, …)
#   view_raw             — active view source: slice OR moment map
#   slice_proc           — smoothed slice for the main image panel
#   slice_disp           — scale-applied (lin/log/ln) main panel array
#   compare_slice_disp   — scale-applied comparison panel array
#   plot_x / plot_y      — native pixel coordinates for the displayed heatmaps
#   slice_plot           — downsampled, heatmap-oriented slice display array
#   compare_slice_plot   — downsampled, heatmap-oriented comparison display array
#   mask_slice           — 2-D BitMatrix of the active mask at current (axis, idx)
#   moment_raw           — cached moment map (order 0/1/2) along active axis
#   _moment_cache        — Dict keyed by (axis, order, mask token)
#   _get_gauss_kernel    — closure: returns (and caches) an ImageFiltering Gaussian kernel for σ
#
# Entry point: `_cube_slice_pipeline_bundle(st; kwargs...)` (preferred) or
# `_cube_slice_pipeline_bundle(; kwargs...)` (legacy keyword form).
# Returns a named tuple with all the items listed above.

"""
    _cube_slice_pipeline_bundle(st::CubeViewObservables; kwargs...) -> NamedTuple
    _cube_slice_pipeline_bundle(; kwargs...) -> NamedTuple

Build the full reactive slice-and-moment data pipeline for the cube viewer.

All returned values are `Observable`s (or mutable state objects) that stay
alive as long as the calling `_view_cube` frame is alive.  No Makie axes or
widgets are touched here — this is a pure data layer.

### Key inputs (all keyword arguments)

| Name | Description |
|---|---|
| `data` | `Array{Float32,3}` — cube data, immutable during the session |
| `siz` | `size(data)` |
| `wcs` | WCS descriptor (may be `nothing`) |
| `axis` | `Observable{Int}` — active slicing axis 1/2/3 |
| `idx` | `Observable{Int}` — current primary slice index |
| `compare_idx` | `Observable{Int}` — current comparison slice index |
| `gauss_on` | `Observable{Bool}` — toggle Gaussian smoothing |
| `sigma` | `Observable{Float32}` — smoothing σ in pixels |
| `compare_data` | `Observable{Any}` — second cube array or `nothing` |
| `compare_mode` | `Observable{Symbol}` — `:A`, `:B`, `:diff`, `:ratio`, `:residuals` |
| `view_product` | `Observable{Symbol}` — `:slice` or `:moment` |
| `rotation_axis` | `Observable{NTuple{3,Float32}}` — arbitrary rotation axis for `:rotproj` |
| `rotation_angle` | `Observable{Float32}` — angle in degrees for `:rotproj` |
| `rotation_projection_mode` | `Observable{Symbol}` — `:mean`, `:sum`, or `:max` |
| `mask_bits_obs` | `Observable{Union{Nothing,BitArray{3}}}` — active mask |
| `moment_order` | `Observable{Int}` — 0, 1, or 2 |
| `img_scale_mode` | `Observable{Symbol}` — `:lin`, `:log10`, or `:ln` |
| `asinh_softening` | `Real` — softening value for `:asinh` image scaling |
| `moment_threshold` | `Real` — noise floor passed to `moment_map` |
| `moment_nsigma` | `Union{Nothing,Real}` — σ-clip threshold for moments |
| `moment_channels` | `Union{Nothing,AbstractVector{Int}}` — channel range for moments |
| `display_max_pixels` | Maximum displayed heatmap pixels along either slice dimension |
| `display_downsample_mode` | `:block_mean` or `:subsample` for display downsampling |
"""
function _cube_slice_pipeline_bundle(st::CubeViewObservables; kwargs...)
    return _cube_slice_pipeline_bundle(;
        kwargs...,
        axis = st.axis,
        idx = st.idx,
        compare_idx = st.compare_idx,
        gauss_on = st.gauss_on,
        sigma = st.sigma,
        compare_data = st.compare_data,
        compare_mode = st.compare_mode,
        view_product = st.view_product,
        rotation_axis = st.rotation_axis,
        rotation_angle = st.rotation_angle,
        rotation_projection_mode = st.rotation_projection_mode,
        mask_bits_obs = st.mask_bits_obs,
        moment_order = st.moment_order,
        img_scale_mode = st.img_scale_mode,
    )
end

function _cube_slice_pipeline_bundle(;
    data,
    siz,
    wcs,
    axis,
    idx,
    compare_idx,
    gauss_on,
    sigma,
    compare_data,
    compare_mode,
    view_product,
    rotation_axis = nothing,
    rotation_angle = nothing,
    rotation_projection_mode = nothing,
    mask_bits_obs,
    moment_order,
    img_scale_mode,
    asinh_softening::Real = ASINH_SOFTENING_DEFAULT,
    moment_threshold::Real,
    moment_nsigma,
    moment_channels,
    display_max_pixels::Integer = 4096,
    display_downsample_mode::Symbol = :block_mean,
)
    display_max_pixels > 0 || throw(ArgumentError("display_max_pixels must be positive"))
    display_downsample_mode in (:block_mean, :subsample) ||
        throw(ArgumentError("display_downsample_mode must be :block_mean or :subsample"))

    # ---- local geometry helpers ---- #
    # (mirrors the helpers defined at the top of _view_cube; kept local so
    # this bundle does not depend on the calling frame's local functions)
    _slice_dims(a::Integer) = if a == 1
        (siz[2], siz[3])
    elseif a == 2
        (siz[1], siz[3])
    else
        (siz[1], siz[2])
    end

    _spectral_coords(dim::Integer) = Float32[
        has_wcs(wcs, dim) ? Float32(world_coord(wcs, dim, chan)) : Float32(chan)
        for chan in 1:siz[dim]
    ]

    # ---- Gaussian kernel cache ---- #
    # why: Kernel.gaussian((σ, σ)) is rebuilt on every slice update.
    # The viewer is dominated by frames at constant σ, so cache by σ value.
    # Scope is local to this bundle invocation — released when the figure GCs.
    _gauss_kernel_cache = Dict{Float32, Any}()
    _get_gauss_kernel(σ::Real) = get!(_gauss_kernel_cache, Float32(σ)) do
        ImageFiltering.Kernel.gaussian((Float32(σ), Float32(σ)))
    end

    # why: the *_proc lifts feed slices into `imfilter`, which already allocates
    # its own Float32 output. An unconditional `Float32.(s)` therefore adds a
    # full extra slice copy on every smoothing tick — ~64 MB per action on a
    # 4096² cube. For the default in-memory Float32 cubes the input is already
    # the right eltype, so only convert when it actually differs. `imfilter` is
    # non-mutating, so handing it the raw (possibly aliased) Float32 view is
    # safe under the read-only invariant documented above.
    _as_f32(s::AbstractArray{Float32}) = s
    _as_f32(s::AbstractArray) = Float32.(s)
    _matrix_lift(f, args...) = map!(f, Observable{AbstractMatrix{Float32}}(), args...)
    rot_axis_obs = rotation_axis === nothing ? Observable((0f0, 0f0, 1f0)) : rotation_axis
    rot_angle_obs = rotation_angle === nothing ? Observable(0f0) : rotation_angle
    rot_mode_obs = rotation_projection_mode === nothing ? Observable(:mean) : rotation_projection_mode

    # ---- slice Observables ---- #
    # why: clamp rather than throw so the UI never crashes on a transient
    # out-of-bounds value (e.g. when axis changes before idx is re-clamped by
    # the slider). @warn makes the anomaly visible in the REPL/log without
    # silently serving the wrong slice undetected.
    #
    # hot-path note: we expose a non-copying `get_slice_view` here rather than
    # `get_slice_copy`. For a big in-memory cube this saves a full 2-D slice
    # allocation on every navigation tick; for a `LazyFITSCube` it hands back
    # the matrix already materialised (and cached) by `read_slice!`. This is
    # safe because EVERY downstream consumer is read-only and allocates its own
    # buffer the moment a transformation is required (`Float32.(s)` /
    # `imfilter(...)` in the *_proc lifts, `apply_scale_display` in *_disp).
    # The cube itself is documented as immutable for the session, so aliasing
    # it cannot corrupt anything. If a future consumer mutates a raw slice in
    # place, switch that one consumer back to `get_slice_copy` / `copy(s)`.
    slice_raw = lift(axis, idx) do a, id
        n = siz[a]
        if !(1 <= id <= n)
            @warn "SlicePipelineBundle: idx=$id is out of bounds for axis=$a " *
                  "(size $n); clamping to $(clamp(id, 1, n))."
        end
        get_slice_view(data, a, clamp(id, 1, n))
    end

    compare_slice_raw = lift(compare_data, axis, compare_idx) do cmp, a, id
        cmp === nothing && return fill(NaN32, _slice_dims(a))
        # use size(cmp, a) — the comparison cube may differ from the primary
        n_cmp = size(cmp, a)
        if !(1 <= id <= n_cmp)
            @warn "SlicePipelineBundle: compare_idx=$id is out of bounds for " *
                  "axis=$a (comparison cube size $n_cmp); clamping to $(clamp(id, 1, n_cmp))."
        end
        get_slice_view(cmp, a, clamp(id, 1, n_cmp))
    end

    # Smoothed copy of the primary slice — used as the A-side of the dual view.
    base_slice_proc = _matrix_lift(slice_raw, gauss_on, sigma) do s, on, σ
        if on && σ > 0
            imfilter(_as_f32(s), _get_gauss_kernel(σ))
        else
            s
        end
    end

    # ---- moment observables ---- #
    # Δx along the slicing axis from the WCS step when available.
    moment_dx_for(a::Integer) = has_wcs(wcs, a) ? abs(Float64(wcs[a].cdelt)) : 1.0

    # why: data is immutable during a viewer session and threshold / nsigma /
    # channels are passed once at construction. Cache by (axis, order, mask)
    # so toggling axis or moment order doesn't recompute the whole map, while
    # direct mask Observable updates cannot serve a stale unmasked/masked map.
    _mask_cache_token(::Nothing) = UInt(0)
    _mask_cache_token(bits::BitArray{3}) = objectid(bits)
    _moment_cache = Dict{Tuple{Int,Int,UInt}, Matrix{Float32}}()
    moment_raw = lift(axis, moment_order, mask_bits_obs) do a, ord, mbits
        get!(_moment_cache, (Int(a), Int(ord), _mask_cache_token(mbits))) do
            moment_map(data, a, ord;
                coords    = _spectral_coords(a),
                threshold = moment_threshold,
                nsigma    = moment_nsigma,
                channels  = moment_channels === nothing ? (1:siz[a]) : moment_channels,
                dx        = moment_dx_for(a),
                mask      = mbits)
        end
    end

    rotation_projection_raw = _matrix_lift(axis, rot_axis_obs, rot_angle_obs, rot_mode_obs) do a, raxis, angle, mode
        _cube_rotated_projection(data, siz, a, raxis, angle, mode)
    end

    # ---- view_raw: route to slice or moment map ---- #
    view_raw = _matrix_lift(slice_raw, moment_raw, rotation_projection_raw, view_product) do s, m, rp, product
        product === :moment ? m : product === :rotproj ? rp : s
    end

    # ---- processed slices ---- #
    slice_proc = _matrix_lift(view_raw, gauss_on, sigma, view_product) do s, on, σ, product
        if product === :slice && on && σ > 0
            imfilter(_as_f32(s), _get_gauss_kernel(σ))
        else
            s
        end
    end

    compare_slice_proc = _matrix_lift(compare_slice_raw, gauss_on, sigma) do s, on, σ
        if on && σ > 0
            imfilter(_as_f32(s), _get_gauss_kernel(σ))
        else
            s
        end
    end

    compare_product_proc = lift(base_slice_proc, compare_slice_proc, compare_mode, compare_data) do a, b, mode, cmp
        cmp === nothing && return fill(NaN32, size(a))
        dual_view_product(a, b, mode)
    end

    # ---- display arrays (scale-applied, NaN-sanitised) ---- #
    # why: apply_scale_display fuses the scale step and the NaN→0 sanitization
    # into a single pass, saving one allocation per slice update.
    slice_disp = lift(slice_proc, img_scale_mode) do s, m
        apply_scale_display(s, m; asinh_softening = asinh_softening)
    end

    compare_slice_disp = lift(compare_product_proc, img_scale_mode) do s, m
        apply_scale_display(s, m; asinh_softening = asinh_softening)
    end

    # ---- plot-ready display arrays ---- #
    #
    # The raw slice orientation is `(u, v)` while Makie's explicit
    # `heatmap!(x, y, z)` expects `z` as `(x, y)`.  We therefore downsample in
    # native slice orientation, then transpose only the displayed copy.  The
    # x/y vectors remain in native cube pixel coordinates, so crosshairs,
    # regions, zoom boxes, and mouse picking keep using the same coordinates
    # even when a 8000×8000 slice is rendered as e.g. 4096×4096.
    _plot_stride(A::AbstractMatrix) = downsample_factor(size(A), display_max_pixels)
    _plot_downsample(A::AbstractMatrix, stride::Integer) =
        stride == 1 ? Float32.(A) :
        display_downsample_mode === :subsample ?
            downsample_subsample(A, stride) :
            downsample_block_mean(A, stride)
    _plot_image(A::AbstractMatrix, stride::Integer) =
        permutedims(_plot_downsample(A, stride))

    plot_stride = lift(slice_disp) do s
        _plot_stride(s)
    end
    plot_x = lift(slice_disp, plot_stride) do s, stride
        downsample_centers(size(s, 2), stride)
    end
    plot_y = lift(slice_disp, plot_stride) do s, stride
        downsample_centers(size(s, 1), stride)
    end
    slice_plot = lift(slice_disp, plot_stride) do s, stride
        _plot_image(s, stride)
    end
    compare_slice_plot = lift(compare_slice_disp, plot_stride) do s, stride
        _plot_image(s, stride)
    end

    # ---- mask_slice: 2-D mask aligned with the current (axis, idx) ---- #
    # Returns `nothing` when no mask is active so downstream lifts can
    # short-circuit to the unmasked path.
    mask_slice = map!(
        function (bits, a, id)
            bits === nothing && return nothing
            n_bits = size(bits, a)
            if !(1 <= id <= n_bits)
                @warn "SlicePipelineBundle: idx=$id is out of bounds for mask axis=$a " *
                      "(size $n_bits); clamping to $(clamp(id, 1, n_bits))."
            end
            id_safe = clamp(id, 1, n_bits)
            if a == 1
                collect(@view bits[id_safe, :, :])
            elseif a == 2
                collect(@view bits[:, id_safe, :])
            else
                collect(@view bits[:, :, id_safe])
            end
        end,
        Observable{Union{Nothing,AbstractMatrix{Bool}}}(),
        mask_bits_obs, axis, idx,
    )

    return (;
        slice_raw,
        compare_slice_raw,
        base_slice_proc,
        compare_slice_proc,
        compare_product_proc,
        view_raw,
        rotation_projection_raw,
        slice_proc,
        slice_disp,
        compare_slice_disp,
        plot_stride,
        plot_x,
        plot_y,
        slice_plot,
        compare_slice_plot,
        mask_slice,
        moment_raw,
        _moment_cache,
        _get_gauss_kernel,
        _gauss_kernel_cache,
    )
end
