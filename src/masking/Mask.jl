# path: src/masking/Mask.jl
#
# Persistent mask system for MANTA cube viewers.
#
# A *mask* is a `BitArray{3}` aligned with the underlying cube; `true` means
# "this voxel is kept by the mask, downstream computations should include it".
# Sources are declarative (`MaskSource` subtypes), so a mask can always be
# regenerated from the data + the source description — which is what gets
# stored in `viewer_settings.toml`.
#
# Supported source kinds (this iteration):
#   * `NoMaskSource`            — no mask, all voxels are kept (used as a
#     sentinel "disabled" value so the UI can carry a non-`nothing` state).
#   * `FiniteSource`            — keep only voxels where `isfinite(data)`.
#   * `ThresholdSource(:ge|:le|:range|:outside, lo, hi)` — keep voxels whose
#     value satisfies the threshold predicate. `:ge` uses `lo` alone; `:le`
#     uses `hi` alone; `:range` keeps `lo <= v <= hi`; `:outside` keeps
#     `v < lo || v > hi`. Non-finite voxels are NEVER kept by a threshold.
#   * `RectangleSource(i1, i2, j1, j2, k1, k2)` — keep voxels whose 1-based
#     index falls in the closed box. `nothing` for any axis means "all".
#
# Combinators (`:and` / `:or` / `:not`) compose any two `MaskSource` values
# into a new one. They are fully recursive — e.g. `AndSource(ThresholdSource(…),
# NotSource(RectangleSource(…)))` is valid and round-trips through TOML.
#
# Consumers integrate via an optional `mask::Union{Nothing,AbstractArray{Bool,3}}`
# kwarg (see `moment_map`, `mean_region_spectrum`) so passing `nothing` keeps
# the legacy behaviour identical.

# --- Public type hierarchy ----------------------------------------------

abstract type MaskSource end

"""
    NoMaskSource()

Sentinel "no mask" source. `build_mask(NoMaskSource(), data)` returns a
fully-true `BitArray{3}` so consumers can ignore the mask without branching.
"""
struct NoMaskSource <: MaskSource end

"""
    FiniteSource()

Keep voxels where `isfinite(data)`.
"""
struct FiniteSource <: MaskSource end

"""
    ThresholdSource(op::Symbol, lo::Float64, hi::Float64)

Keep voxels whose value satisfies the predicate selected by `op`:
- `:ge`       → `data >= lo`
- `:le`       → `data <= hi`
- `:range`    → `lo <= data <= hi`
- `:outside`  → `data < lo || data > hi`

Non-finite voxels are always rejected (no NaN propagation into downstream
sums). `lo` and `hi` are stored as `Float64` regardless of the cube eltype
so the source description survives a `Float32`/`Float64` round-trip cleanly.
"""
struct ThresholdSource <: MaskSource
    op::Symbol
    lo::Float64
    hi::Float64

    function ThresholdSource(op::Symbol, lo::Real, hi::Real)
        op in (:ge, :le, :range, :outside) || throw(ArgumentError(
            "ThresholdSource: op must be :ge, :le, :range, or :outside; got $(op)"))
        return new(op, Float64(lo), Float64(hi))
    end
end

"""
    RectangleSource(i1, i2, j1, j2, k1, k2)

Keep voxels whose 1-based `(i, j, k)` index falls within the closed box.
Any bound can be `nothing` meaning "no constraint on that side". Each pair
is automatically reordered if min > max.
"""
struct RectangleSource <: MaskSource
    i1::Union{Nothing,Int}
    i2::Union{Nothing,Int}
    j1::Union{Nothing,Int}
    j2::Union{Nothing,Int}
    k1::Union{Nothing,Int}
    k2::Union{Nothing,Int}
end

"""
    AndSource(a::MaskSource, b::MaskSource)

Keep voxels that are kept by **both** `a` and `b`. Equivalent to the
bitwise AND of the two materialised masks. Recursively composable:
any `MaskSource` (including another `AndSource`) may be used as operand.
"""
struct AndSource <: MaskSource
    a::MaskSource
    b::MaskSource
end

"""
    OrSource(a::MaskSource, b::MaskSource)

Keep voxels that are kept by `a` **or** `b` (inclusive). Equivalent to
the bitwise OR of the two materialised masks.
"""
struct OrSource <: MaskSource
    a::MaskSource
    b::MaskSource
end

"""
    NotSource(inner::MaskSource)

Invert a mask: keep voxels **rejected** by `inner`, reject those it keeps.
`NotSource(NoMaskSource())` therefore produces an all-false mask.
"""
struct NotSource <: MaskSource
    inner::MaskSource
end

function RectangleSource(;
    i1::Union{Nothing,Integer} = nothing,
    i2::Union{Nothing,Integer} = nothing,
    j1::Union{Nothing,Integer} = nothing,
    j2::Union{Nothing,Integer} = nothing,
    k1::Union{Nothing,Integer} = nothing,
    k2::Union{Nothing,Integer} = nothing,
)
    # Note: variable names here MUST NOT collide with names assigned in the
    # outer scope below — Julia's nested-function scoping would otherwise turn
    # `lo_`/`hi_` into closure references to the outer locals, and each `_pair`
    # call would silently overwrite the previous result.
    _pair(a, b) = begin
        lo_ = a === nothing ? nothing : Int(a)
        hi_ = b === nothing ? nothing : Int(b)
        if lo_ !== nothing && hi_ !== nothing && lo_ > hi_
            lo_, hi_ = hi_, lo_
        end
        (lo_, hi_)
    end
    (i_a, i_b) = _pair(i1, i2)
    (j_a, j_b) = _pair(j1, j2)
    (k_a, k_b) = _pair(k1, k2)
    return RectangleSource(i_a, i_b, j_a, j_b, k_a, k_b)
end

# --- Aggregate type carried in the viewer state -------------------------

"""
    MANTAMask(bits::BitArray{3}, source::MaskSource)

The materialised mask plus the declarative source it was built from. The
source is the part stored in `viewer_settings.toml`; `bits` is recomputed
via `build_mask(source, data)` on reload.
"""
struct MANTAMask
    bits::BitArray{3}
    source::MaskSource
end

# --- Bounds validation --------------------------------------------------

"""
    validate_bounds(src::RectangleSource, cube_size)

Throw `ArgumentError` if any specified (non-`nothing`) bound of `src` lies
outside the valid 1-based index range for `cube_size`.

Called automatically by `build_mask` so errors are reported at the call site
rather than inside low-level indexing. Can also be called by UI code as soon
as the cube dimensions are known — e.g. when the user finishes editing a mask
field — to surface the problem before materialisation.

```julia
validate_bounds(RectangleSource(i1=1, i2=50), (64, 64, 64))  # ok
validate_bounds(RectangleSource(i1=1, i2=1000), (64, 64, 64))  # ArgumentError
```
"""
function validate_bounds(src::RectangleSource, cube_size)
    nx, ny, nz = Int(cube_size[1]), Int(cube_size[2]), Int(cube_size[3])
    function _check(val, lo, hi, name)
        val === nothing && return
        (val < lo || val > hi) || return
        throw(ArgumentError(
            "RectangleSource: $name = $val is out of range [$lo, $hi] " *
            "for a cube of size ($nx, $ny, $nz)"))
    end
    _check(src.i1, 1, nx, "i1")
    _check(src.i2, 1, nx, "i2")
    _check(src.j1, 1, ny, "j1")
    _check(src.j2, 1, ny, "j2")
    _check(src.k1, 1, nz, "k1")
    _check(src.k2, 1, nz, "k2")
    return nothing
end

"""
    validate_bounds(src::MaskSource, cube_size)

No-op fallback for non-rectangle sources.
"""
validate_bounds(::MaskSource, _) = nothing

# --- Builders -----------------------------------------------------------

"""
    build_mask(source::MaskSource, data::AbstractArray{<:Real,3}) -> BitArray{3}

Materialise a mask aligned with `data` from a declarative source. Returns a
fresh `BitArray{3}` of the same shape as `data` — callers can mutate it
freely, in particular to combine it with the legacy `moment_threshold`.
"""
function build_mask end

function build_mask(::NoMaskSource, data::AbstractArray{<:Real,3})
    return trues(size(data))
end

function build_mask(::FiniteSource, data::AbstractArray{<:Real,3})
    out = BitArray(undef, size(data))
    @inbounds for i in eachindex(data, out)
        out[i] = isfinite(Float64(data[i]))
    end
    return out
end

function build_mask(src::ThresholdSource, data::AbstractArray{<:Real,3})
    out = BitArray(undef, size(data))
    op = src.op
    lo = src.lo
    hi = src.hi
    @inbounds for i in eachindex(data, out)
        v = Float64(data[i])
        out[i] = if !isfinite(v)
            false
        elseif op === :ge
            v >= lo
        elseif op === :le
            v <= hi
        elseif op === :range
            lo <= v <= hi
        else  # :outside (validated at construction)
            v < lo || v > hi
        end
    end
    return out
end

function build_mask(src::RectangleSource, data::AbstractArray{<:Real,3})
    validate_bounds(src, size(data))
    nx, ny, nz = size(data)
    i1 = src.i1 === nothing ? 1  : src.i1
    i2 = src.i2 === nothing ? nx : src.i2
    j1 = src.j1 === nothing ? 1  : src.j1
    j2 = src.j2 === nothing ? ny : src.j2
    k1 = src.k1 === nothing ? 1  : src.k1
    k2 = src.k2 === nothing ? nz : src.k2
    out = falses(nx, ny, nz)
    if i1 <= i2 && j1 <= j2 && k1 <= k2
        @inbounds out[i1:i2, j1:j2, k1:k2] .= true
    end
    return out
end

function build_mask(src::AndSource, data::AbstractArray{<:Real,3})
    a = build_mask(src.a, data)
    b = build_mask(src.b, data)
    return a .& b
end

function build_mask(src::OrSource, data::AbstractArray{<:Real,3})
    a = build_mask(src.a, data)
    b = build_mask(src.b, data)
    return a .| b
end

function build_mask(src::NotSource, data::AbstractArray{<:Real,3})
    inner = build_mask(src.inner, data)
    return .!inner
end

"""
    make_mask(source::MaskSource, data) -> MANTAMask

Convenience wrapper that builds the bits and packages them with the source.
"""
make_mask(source::MaskSource, data::AbstractArray{<:Real,3}) =
    MANTAMask(build_mask(source, data), source)

# --- Statistics ---------------------------------------------------------

"""
    mask_total(m) -> Int

Total number of voxels (i.e. `prod(size(m.bits))`).
"""
mask_total(m::MANTAMask) = length(m.bits)

"""
    mask_count(m) -> Int

Number of `true` voxels in the mask.
"""
mask_count(m::MANTAMask) = count(m.bits)

"""
    mask_fraction(m) -> Float64

Ratio of `true` voxels over total voxels.
"""
function mask_fraction(m::MANTAMask)
    n = mask_total(m)
    n == 0 ? 0.0 : mask_count(m) / n
end

# --- Combination with the legacy `moment_threshold` ---------------------
#
# CubeView keeps the existing `moment_threshold` semantics. The mask is ANDed
# with the threshold inside the inner loop of `_pixel_moment` via NaN-ing
# masked-out voxels before they enter the accumulator. The helper below is
# used by `moment_map` (see `helpers/Moments.jl`).

"""
    mask_value(mask, axis, u, v, c) -> Bool

Look up the mask at the `(u, v, c)` coordinates of the slice axis `axis`,
matching the conventions used by `moment_map` and `mean_region_spectrum`.
When `mask === nothing`, returns `true` so the consumer falls back to the
non-masked path with zero branching cost.
"""
@inline function mask_value(mask::Nothing, axis::Integer, u::Integer, v::Integer, c::Integer)
    return true
end

@inline function mask_value(mask::AbstractArray{Bool,3},
                            axis::Integer, u::Integer, v::Integer, c::Integer)
    @inbounds return if axis == 1
        mask[c, u, v]
    elseif axis == 2
        mask[u, c, v]
    else
        mask[u, v, c]
    end
end

# --- TOML I/O -----------------------------------------------------------
#
# `mask_source_to_toml(src) -> Dict{String,Any}` is intended to be merged
# into the top-level settings dict under the `"mask"` key. The inverse
# `mask_source_from_toml(d)` is permissive: unknown / malformed entries
# return `NoMaskSource()` so a corrupted settings file never crashes a
# viewer launch.
#
# **Settings-corruption strategy (uniform across all viewer settings)**
# Every field in the settings TOML is treated with the same fallback
# policy: on a type mismatch or missing key, emit `@warn` and continue
# with the current / default value rather than throwing.  The helpers
# `_safe_int`, `_safe_float32`, and `_safe_bool` in
# `src/views/cube/SettingsBundle.jl` implement this for scalar fields;
# `mask_source_from_toml` implements it for the structured mask subtree.
# `load_viewer_settings` is intentionally *strict* on TOML syntax errors
# (they are caught by the outer `try/catch` in the `on_mode` callback and
# reported to the status bar), because a syntactically broken file is a
# different problem from a file whose individual values have the wrong
# type.

"""
    mask_source_to_toml(src::MaskSource) -> Dict{String,Any}

Stable, human-readable serialisation. Keys: `"kind"` plus source-specific
fields. The output is a plain `Dict` ready to be written via `TOML.print`.
"""
function mask_source_to_toml(src::MaskSource)
    if src isa NoMaskSource
        return Dict{String,Any}("kind" => "none")
    elseif src isa FiniteSource
        return Dict{String,Any}("kind" => "finite")
    elseif src isa ThresholdSource
        return Dict{String,Any}(
            "kind" => "threshold",
            "op"   => String(src.op),
            "lo"   => src.lo,
            "hi"   => src.hi,
        )
    elseif src isa RectangleSource
        d = Dict{String,Any}("kind" => "rectangle")
        src.i1 === nothing || (d["i1"] = src.i1)
        src.i2 === nothing || (d["i2"] = src.i2)
        src.j1 === nothing || (d["j1"] = src.j1)
        src.j2 === nothing || (d["j2"] = src.j2)
        src.k1 === nothing || (d["k1"] = src.k1)
        src.k2 === nothing || (d["k2"] = src.k2)
        return d
    elseif src isa AndSource
        return Dict{String,Any}(
            "kind" => "and",
            "a"   => mask_source_to_toml(src.a),
            "b"   => mask_source_to_toml(src.b),
        )
    elseif src isa OrSource
        return Dict{String,Any}(
            "kind" => "or",
            "a"   => mask_source_to_toml(src.a),
            "b"   => mask_source_to_toml(src.b),
        )
    elseif src isa NotSource
        return Dict{String,Any}(
            "kind"  => "not",
            "inner" => mask_source_to_toml(src.inner),
        )
    end
    return Dict{String,Any}("kind" => "none")
end

"""
    mask_source_from_toml(d) -> MaskSource

Parse a dict (typically from `TOML.parsefile`) back into a `MaskSource`.
Returns `NoMaskSource()` for missing / unknown / malformed inputs so that a
broken settings file degrades gracefully.
"""
function mask_source_from_toml(d::AbstractDict)
    kind = get(d, "kind", "none")
    if kind == "finite"
        return FiniteSource()
    elseif kind == "threshold"
        op = Symbol(get(d, "op", "ge"))
        lo = _toml_float(d, "lo", 0.0)
        hi = _toml_float(d, "hi", 0.0)
        op in (:ge, :le, :range, :outside) || return NoMaskSource()
        return ThresholdSource(op, lo, hi)
    elseif kind == "rectangle"
        return RectangleSource(;
            i1 = _toml_int_or_nothing(d, "i1"),
            i2 = _toml_int_or_nothing(d, "i2"),
            j1 = _toml_int_or_nothing(d, "j1"),
            j2 = _toml_int_or_nothing(d, "j2"),
            k1 = _toml_int_or_nothing(d, "k1"),
            k2 = _toml_int_or_nothing(d, "k2"),
        )
    elseif kind == "and"
        a = mask_source_from_toml(get(d, "a", nothing))
        b = mask_source_from_toml(get(d, "b", nothing))
        return AndSource(a, b)
    elseif kind == "or"
        a = mask_source_from_toml(get(d, "a", nothing))
        b = mask_source_from_toml(get(d, "b", nothing))
        return OrSource(a, b)
    elseif kind == "not"
        inner = mask_source_from_toml(get(d, "inner", nothing))
        return NotSource(inner)
    end
    return NoMaskSource()
end

mask_source_from_toml(::Nothing) = NoMaskSource()

@inline function _toml_float(d::AbstractDict, key::AbstractString, fallback::Float64)
    v = get(d, key, fallback)
    return v isa Real ? Float64(v) : fallback
end

@inline function _toml_int_or_nothing(d::AbstractDict, key::AbstractString)
    haskey(d, key) || return nothing
    v = d[key]
    return v isa Integer ? Int(v) : nothing
end
