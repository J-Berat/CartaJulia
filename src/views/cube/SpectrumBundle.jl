# path: src/views/cube/SpectrumBundle.jl
#
# Spectrum-panel helpers for the 3D cube viewer.
#
# Extracted from src/views/CubeView.jl during a structural refactor
# (no behaviour change). Contains:
#   - `refresh_spec_ylim!` : update the spectrum y-axis limits from the
#                            current `spec_ylimits_source` / `spec_ylimits_value`
#   - `refresh_spectrum!`  : recompute the active spectrum (voxel or regional
#                            mean) for both cube A and comparison cube B, then
#                            call `refresh_spec_ylim!`.
#
# Both functions are inner closures that capture all their dependencies
# through the keyword arguments; they are returned as a named tuple so
# CubeView.jl can destructure them and pass them on to sibling bundles
# (KeyboardBundle, ExportBundle, etc.) that also call `refresh_spectrum!`.
#
# Entry point: `_cube_spectrum_bundle(; kwargs...)`.
# Returns `(; refresh_spec_ylim!, refresh_spectrum!)`.

"""
    _cube_spectrum_bundle(; kwargs...) -> NamedTuple

Build the two spectrum-update helpers for the cube viewer.

### Keyword arguments

| Name | Type | Description |
|---|---|---|
| `data` | `Array{Float32,3}` | The cube data (immutable during the session). |
| `siz` | `NTuple{3,Int}` | `size(data)`. |
| `axis` | `Observable{Int}` | Active slicing axis (1/2/3). |
| `i_idx`, `j_idx`, `k_idx` | `Observable{Int}` | Current voxel indices. |
| `region_uvs` | `Observable{Vector{Tuple{Int,Int}}}` | Selected region pixels. |
| `mask_bits_obs` | `Observable{Union{Nothing,BitArray{3}}}` | Active mask or `nothing`. |
| `compare_data` | `Observable{Any}` | Comparison cube array or `nothing`. |
| `spec_x_axes` | `NTuple{3,Vector}` | Pre-computed x-axis vectors for each axis. |
| `spec_x_raw` | `Observable` | Output: currently displayed x-axis. |
| `spec_y_raw` | `Observable` | Output: spectrum y-values for cube A. |
| `spec_y_buf` | `Vector{Float32}` | Mutable buffer reused by every spectrum update. |
| `spec_y_compare_raw` | `Observable` | Output: spectrum y-values for cube B. |
| `spec_y_compare_buf` | `Vector{Float32}` | Mutable buffer for cube B. |
| `spec_ylimits_source` | `Observable{Symbol}` | `:auto`, `:manual`, or `:contrast`. |
| `spec_ylimits_value` | `Observable{Tuple{Float32,Float32}}` | Manual y-axis limits. |
| `ax_spec` | `Makie.Axis` | The spectrum axis (limits are mutated directly). |

### Returns
`(; refresh_spec_ylim!, refresh_spectrum!)` — two zero-argument functions.
"""
function _cube_spectrum_bundle(;
    data,
    siz,
    axis,
    i_idx,
    j_idx,
    k_idx,
    region_uvs,
    mask_bits_obs,
    compare_data,
    spec_x_axes,
    spec_x_raw,
    spec_y_raw,
    spec_y_buf,
    spec_y_compare_raw,
    spec_y_compare_buf,
    spec_ylimits_source,
    spec_ylimits_value,
    ax_spec,
)
    # ------------------------------------------------------------------ #
    # refresh_spec_ylim!
    # ------------------------------------------------------------------ #
    # Applies the current y-axis policy to `ax_spec`.  Called after every
    # spectrum recompute and after the user changes limits/scale manually.
    function refresh_spec_ylim!()
        x_max = Float32(max(0, length(spec_y_buf) - 1))
        if spec_ylimits_source[] === :manual || spec_ylimits_source[] === :contrast
            vmin_, vmax_ = spec_ylimits_value[]
            limits!(ax_spec, nothing, nothing, vmin_, vmax_)
            xlims!(ax_spec, 0f0, x_max)
        else
            autolimits!(ax_spec)
            xlims!(ax_spec, 0f0, x_max)
        end
    end

    # ------------------------------------------------------------------ #
    # refresh_spectrum!
    # ------------------------------------------------------------------ #
    # Fills spec_y_buf (and spec_y_compare_buf) from the current viewer
    # state, then nudges the Observables so the plot lines update.
    function refresh_spectrum!()
        mbits = mask_bits_obs[]
        if !isempty(region_uvs[])
            # --- regional mean spectrum ---
            spec_x_raw[] = spec_x_axes[axis[]]
            y = mean_region_spectrum(data, axis[], region_uvs[]; mask = mbits)
            resize!(spec_y_buf, length(y))
            copyto!(spec_y_buf, y)
            ax_spec.title[] = latexstring("\\text{Mean spectrum in selected region}")
        elseif axis[] == 1
            spec_x_raw[] = spec_x_axes[1]
            resize!(spec_y_buf, siz[1])
            @views copyto!(spec_y_buf, data[:, j_idx[], k_idx[]])
            if mbits !== nothing
                @inbounds for c in 1:siz[1]
                    mbits[c, j_idx[], k_idx[]] || (spec_y_buf[c] = NaN32)
                end
            end
            ax_spec.title[] = L"\text{Spectrum at selected pixel}"
        elseif axis[] == 2
            spec_x_raw[] = spec_x_axes[2]
            resize!(spec_y_buf, siz[2])
            @views copyto!(spec_y_buf, data[i_idx[], :, k_idx[]])
            if mbits !== nothing
                @inbounds for c in 1:siz[2]
                    mbits[i_idx[], c, k_idx[]] || (spec_y_buf[c] = NaN32)
                end
            end
            ax_spec.title[] = L"\text{Spectrum at selected pixel}"
        else
            spec_x_raw[] = spec_x_axes[3]
            resize!(spec_y_buf, siz[3])
            @views copyto!(spec_y_buf, data[i_idx[], j_idx[], :])
            if mbits !== nothing
                @inbounds for c in 1:siz[3]
                    mbits[i_idx[], j_idx[], c] || (spec_y_buf[c] = NaN32)
                end
            end
            ax_spec.title[] = L"\text{Spectrum at selected pixel}"
        end
        spec_y_raw[] = spec_y_buf

        # Mirror the cube-A extraction on the comparison cube when loaded.
        # Same axis, same voxel / region — only the source array changes.
        cmp = compare_data[]
        if cmp !== nothing
            cmp_mask = (mbits !== nothing && size(cmp) == size(data)) ? mbits : nothing
            if !isempty(region_uvs[])
                y2 = mean_region_spectrum(cmp, axis[], region_uvs[]; mask = cmp_mask)
                resize!(spec_y_compare_buf, length(y2))
                copyto!(spec_y_compare_buf, y2)
            elseif axis[] == 1
                resize!(spec_y_compare_buf, siz[1])
                @views copyto!(spec_y_compare_buf, cmp[:, j_idx[], k_idx[]])
                if cmp_mask !== nothing
                    @inbounds for c in 1:siz[1]
                        cmp_mask[c, j_idx[], k_idx[]] || (spec_y_compare_buf[c] = NaN32)
                    end
                end
            elseif axis[] == 2
                resize!(spec_y_compare_buf, siz[2])
                @views copyto!(spec_y_compare_buf, cmp[i_idx[], :, k_idx[]])
                if cmp_mask !== nothing
                    @inbounds for c in 1:siz[2]
                        cmp_mask[i_idx[], c, k_idx[]] || (spec_y_compare_buf[c] = NaN32)
                    end
                end
            else
                resize!(spec_y_compare_buf, siz[3])
                @views copyto!(spec_y_compare_buf, cmp[i_idx[], j_idx[], :])
                if cmp_mask !== nothing
                    @inbounds for c in 1:siz[3]
                        cmp_mask[i_idx[], j_idx[], c] || (spec_y_compare_buf[c] = NaN32)
                    end
                end
            end
        else
            resize!(spec_y_compare_buf, length(spec_y_buf))
            fill!(spec_y_compare_buf, NaN32)
        end
        spec_y_compare_raw[] = spec_y_compare_buf
        refresh_spec_ylim!()
    end

    return (; refresh_spec_ylim!, refresh_spectrum!)
end
