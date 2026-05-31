# path: src/views/cube/SlicePipelineBundle.jl
#
# Reactive data pipeline for the 3D cube viewer.
#
# Extracted from src/views/CubeView.jl during a structural refactor
# (no behaviour change). Contains all the Makie `lift` chains that turn
# raw cube data + user-controlled Observables into display-ready arrays:
#
#   slice_raw            — unprocessed 2-D slice copy
#   compare_slice_raw    — unprocessed 2-D slice from the comparison cube
#   base_slice_proc      — Gaussian-smoothed slice for dual-view A-side
#   compare_slice_proc   — Gaussian-smoothed comparison slice
#   compare_product_proc — A/B blended product (diff, ratio, …)
#   view_raw             — active view source: slice OR moment map
#   slice_proc           — smoothed slice for the main image panel
#   slice_disp           — scale-applied (lin/log/ln) main panel array
#   compare_slice_disp   — scale-applied comparison panel array
#   mask_slice           — 2-D BitMatrix of the active mask at current (axis, idx)
#   moment_raw           — cached moment map (order 0/1/2) along active axis
#   _moment_cache        — Dict keyed by (axis, order); cleared by MaskBundle on mask change
#   _get_gauss_kernel    — closure: returns (and caches) an ImageFiltering Gaussian kernel for σ
#
# Entry point: `_cube_slice_pipeline_bundle(; kwargs...)`.
# Returns a named tuple with all the items listed above.

"""
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
| `mask_bits_obs` | `Observable{Union{Nothing,BitArray{3}}}` — active mask |
| `moment_order` | `Observable{Int}` — 0, 1, or 2 |
| `img_scale_mode` | `Observable{Symbol}` — `:lin`, `:log10`, or `:ln` |
| `moment_threshold` | `Real` — noise floor passed to `moment_map` |
| `moment_nsigma` | `Union{Nothing,Real}` — σ-clip threshold for moments |
| `moment_channels` | `Union{Nothing,AbstractVector{Int}}` — channel range for moments |
"""
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
    mask_bits_obs,
    moment_order,
    img_scale_mode,
    moment_threshold::Real,
    moment_nsigma,
    moment_channels,
)
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

    # ---- slice Observables ---- #
    # why: clamp rather than throw so the UI never crashes on a transient
    # out-of-bounds value (e.g. when axis changes before idx is re-clamped by
    # the slider). @warn makes the anomaly visible in the REPL/log without
    # silently serving the wrong slice undetected.
    slice_raw = lift(axis, idx) do a, id
        n = siz[a]
        if !(1 <= id <= n)
            @warn "SlicePipelineBundle: idx=$id is out of bounds for axis=$a " *
                  "(size $n); clamping to $(clamp(id, 1, n))."
        end
        get_slice_copy(data, a, clamp(id, 1, n))
    end

    compare_slice_raw = lift(compare_data, axis, compare_idx) do cmp, a, id
        cmp === nothing && return fill(NaN32, _slice_dims(a))
        # use size(cmp, a) — the comparison cube may differ from the primary
        n_cmp = size(cmp, a)
        if !(1 <= id <= n_cmp)
            @warn "SlicePipelineBundle: compare_idx=$id is out of bounds for " *
                  "axis=$a (comparison cube size $n_cmp); clamping to $(clamp(id, 1, n_cmp))."
        end
        get_slice_copy(cmp, a, clamp(id, 1, n_cmp))
    end

    # Smoothed copy of the primary slice — used as the A-side of the dual view.
    base_slice_proc = lift(slice_raw, gauss_on, sigma) do s, on, σ
        if on && σ > 0
            imfilter(Float32.(s), _get_gauss_kernel(σ))
        else
            s
        end
    end

    # ---- moment observables ---- #
    # Δx along the slicing axis from the WCS step when available.
    moment_dx_for(a::Integer) = has_wcs(wcs, a) ? abs(Float64(wcs[a].cdelt)) : 1.0

    # why: data is immutable during a viewer session and threshold / nsigma /
    # channels are passed once at construction. Cache by (axis, order) so
    # toggling axis or moment order doesn't recompute the whole map.
    # Changing the persistent mask invalidates the cache entirely — see
    # `apply_mask_source!` in MaskBundle.jl which empties _moment_cache.
    _moment_cache = Dict{Tuple{Int,Int}, Matrix{Float32}}()
    moment_raw = lift(axis, moment_order, mask_bits_obs) do a, ord, mbits
        get!(_moment_cache, (Int(a), Int(ord))) do
            moment_map(data, a, ord;
                coords    = _spectral_coords(a),
                threshold = moment_threshold,
                nsigma    = moment_nsigma,
                channels  = moment_channels === nothing ? (1:siz[a]) : moment_channels,
                dx        = moment_dx_for(a),
                mask      = mbits)
        end
    end

    # ---- view_raw: route to slice or moment map ---- #
    view_raw = lift(slice_raw, moment_raw, view_product) do s, m, product
        product === :moment ? m : s
    end

    # ---- processed slices ---- #
    slice_proc = lift(view_raw, gauss_on, sigma, view_product) do s, on, σ, product
        if product === :slice && on && σ > 0
            imfilter(Float32.(s), _get_gauss_kernel(σ))
        else
            s
        end
    end

    compare_slice_proc = lift(compare_slice_raw, gauss_on, sigma) do s, on, σ
        if on && σ > 0
            imfilter(Float32.(s), _get_gauss_kernel(σ))
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
        apply_scale_display(s, m)
    end

    compare_slice_disp = lift(compare_product_proc, img_scale_mode) do s, m
        apply_scale_display(s, m)
    end

    # ---- mask_slice: 2-D mask aligned with the current (axis, idx) ---- #
    # Returns `nothing` when no mask is active so downstream lifts can
    # short-circuit to the unmasked path.
    mask_slice = lift(mask_bits_obs, axis, idx) do bits, a, id
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
    end

    return (;
        slice_raw,
        compare_slice_raw,
        base_slice_proc,
        compare_slice_proc,
        compare_product_proc,
        view_raw,
        slice_proc,
        slice_disp,
        compare_slice_disp,
        mask_slice,
        moment_raw,
        _moment_cache,
        _get_gauss_kernel,
        _gauss_kernel_cache,
    )
end
