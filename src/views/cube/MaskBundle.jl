# path: src/views/cube/MaskBundle.jl
#
# Mask management closures for the 3D cube viewer.
#
# Extracted from src/views/CubeView.jl during a structural refactor
# (no behaviour change). These are pure closures over the viewer's
# shared state: parsing helpers, apply/reset, and UI→source conversion.
#
# Entry point: `_cube_mask_bundle(; ...)` returns a NamedTuple of five
# closures that are destructured at the call site inside `_view_cube`.

"""
    _cube_mask_bundle(; kwargs...) -> NamedTuple

Build the five mask-management closures for the cube viewer and return
them as a NamedTuple with fields:

| field                     | purpose                                      |
|---------------------------|----------------------------------------------|
| `_parse_int_range`        | parse "a", "a:b" textbox strings to Int pair |
| `_set_mask_status!`       | update the status label from a MaskSource    |
| `apply_mask_source!`      | materialise a MaskSource and push to state   |
| `reset_mask!`             | clear mask + reset all mask UI widgets       |
| `build_mask_source_from_ui` | read mask widgets → MaskSource (or nothing)|

All closures capture their dependencies through the keyword arguments
rather than through the outer `_view_cube` scope, so this file can be
included and compiled independently of CubeView.jl.
"""
function _cube_mask_bundle(;
    data,
    _moment_cache,
    mask_source_obs,
    mask_bits_obs,
    mask_status_obs,
    mask_status_label,
    mask_lo_box,
    mask_hi_box,
    mask_i_box,
    mask_j_box,
    mask_k_box,
    mask_source_menu,
    mask_op_menu,
    refresh_spectrum!,
    set_status!,
    set_box_text!,
    ui_mask,
    ui_error,
    ui_text_muted,
)
    # ------------------------------------------------------------------
    # Parse "a" or "a:b" strings from integer-range textboxes.
    # Returns (lo, hi, error_msg) where lo == hi == nothing on failure.
    # ------------------------------------------------------------------
    function _parse_int_range(s::AbstractString)
        s2 = strip(String(s))
        isempty(s2) && return (nothing, nothing, "")
        parts = split(s2, ':')
        if length(parts) == 1
            v = tryparse(Int, strip(parts[1]))
            v === nothing && return (nothing, nothing, "could not parse '$(s2)' as int or range")
            return (v, v, "")
        elseif length(parts) == 2
            lo = tryparse(Int, strip(parts[1]))
            hi = tryparse(Int, strip(parts[2]))
            (lo === nothing || hi === nothing) && return (nothing, nothing, "could not parse '$(s2)' as a:b range")
            return (lo, hi, "")
        else
            return (nothing, nothing, "expected 'a:b' or single int, got '$(s2)'")
        end
    end

    # ------------------------------------------------------------------
    # Push a human-readable summary to the status label.
    # ------------------------------------------------------------------
    function _set_mask_status!(src::MaskSource, bits::Union{Nothing,AbstractArray{Bool,3}})
        if src isa NoMaskSource || bits === nothing
            mask_status_obs[] = "No mask applied"
            mask_status_label.color[] = ui_text_muted
        else
            kept = count(bits)
            tot  = length(bits)
            frac = tot == 0 ? 0.0 : kept / tot
            label = if src isa FiniteSource
                "finite"
            elseif src isa ThresholdSource
                op = src.op === :ge ? "≥" : src.op === :le ? "≤" : src.op === :range ? "range" : "outside"
                "threshold $(op)"
            elseif src isa RectangleSource
                "rectangle"
            else
                "mask"
            end
            mask_status_obs[] = "$(label): $(kept)/$(tot) px kept (" *
                                string(round(100 * frac; digits = 2)) * "%)"
            mask_status_label.color[] = ui_mask
        end
        mask_status_label.text[] = mask_status_obs[]
        nothing
    end

    # ------------------------------------------------------------------
    # Materialise `src` into a BitArray and push to the viewer state.
    # Clears the moment cache first so the next moment_raw lift does not
    # serve a stale entry keyed on (axis, order) only.
    # ------------------------------------------------------------------
    function apply_mask_source!(src::MaskSource)
        bits = src isa NoMaskSource ? nothing : build_mask(src, data)
        # Clear the moment cache BEFORE retriggering the moment_raw lift via
        # mask_bits_obs, otherwise the lift would serve a stale entry keyed
        # on (axis, order) only.
        empty!(_moment_cache)
        mask_source_obs[] = src
        mask_bits_obs[]   = bits
        # Mask change re-runs moment_raw / hist_slice_obs automatically; only
        # the per-voxel/region spectrum needs an explicit refresh because it
        # is driven by a side-effecting function rather than a lift on data.
        refresh_spectrum!()
        _set_mask_status!(src, bits)
        nothing
    end

    # ------------------------------------------------------------------
    # Reset mask to NoMaskSource and clear all mask UI widgets.
    # ------------------------------------------------------------------
    function reset_mask!()
        apply_mask_source!(NoMaskSource())
        set_box_text!(mask_lo_box, "")
        set_box_text!(mask_hi_box, "")
        set_box_text!(mask_i_box,  "")
        set_box_text!(mask_j_box,  "")
        set_box_text!(mask_k_box,  "")
        mask_source_menu.selection[] = "none"
        mask_op_menu.selection[]     = "≥"
        set_status!("Mask reset.")
        nothing
    end

    # ------------------------------------------------------------------
    # Read the mask UI widgets and return (MaskSource, error_string).
    # Returns (nothing, msg) on validation failure.
    # ------------------------------------------------------------------
    function build_mask_source_from_ui()
        kind = String(something(mask_source_menu.selection[], "none"))
        if kind == "none"
            return (NoMaskSource(), "")
        elseif kind == "finite"
            return (FiniteSource(), "")
        elseif kind == "threshold"
            op_label = String(something(mask_op_menu.selection[], "≥"))
            op = op_label == "≥"      ? :ge      :
                 op_label == "≤"      ? :le      :
                 op_label == "range"  ? :range   : :outside
            lo_str = strip(get_box_str(mask_lo_box))
            hi_str = strip(get_box_str(mask_hi_box))
            lo = isempty(lo_str) ? 0.0 : something(tryparse(Float64, lo_str), 0.0)
            hi = isempty(hi_str) ? 0.0 : something(tryparse(Float64, hi_str), 0.0)
            if op == :ge && isempty(lo_str)
                mask_status_label.color[] = ui_error
                return (nothing, "threshold ≥ requires 'lo'")
            elseif op == :le && isempty(hi_str)
                mask_status_label.color[] = ui_error
                return (nothing, "threshold ≤ requires 'hi'")
            elseif op in (:range, :outside) && (isempty(lo_str) || isempty(hi_str))
                mask_status_label.color[] = ui_error
                return (nothing, "threshold $(op_label) requires both 'lo' and 'hi'")
            end
            try
                return (ThresholdSource(op, lo, hi), "")
            catch e
                return (nothing, "threshold error: $(sprint(showerror, e))")
            end
        elseif kind == "rectangle"
            i1, i2, msg_i = _parse_int_range(get_box_str(mask_i_box))
            isempty(msg_i) || return (nothing, "i: $(msg_i)")
            j1, j2, msg_j = _parse_int_range(get_box_str(mask_j_box))
            isempty(msg_j) || return (nothing, "j: $(msg_j)")
            k1, k2, msg_k = _parse_int_range(get_box_str(mask_k_box))
            isempty(msg_k) || return (nothing, "k: $(msg_k)")
            return (RectangleSource(i1 = i1, i2 = i2, j1 = j1, j2 = j2, k1 = k1, k2 = k2), "")
        else
            return (nothing, "unknown mask kind: $(kind)")
        end
    end

    return (;
        _parse_int_range,
        _set_mask_status!,
        apply_mask_source!,
        reset_mask!,
        build_mask_source_from_ui,
    )
end
