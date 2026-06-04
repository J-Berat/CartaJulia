# path: src/helpers/FITSHeaders.jl
#
# FITS header construction for exported products (slice, moment map, region
# spectrum, filtered cube). Builds a passthrough copy of the source header,
# renumbers axis-bound keywords as needed, and appends the right HISTORY /
# BUNIT cards for each product. Depends on WCS.jl (header_has/get,
# wcs_axis_label, spectral_quantity_word, …). Extracted from helpers/Helpers.jl.


############################
# FITS header export helpers
############################

# Per-axis WCS keyword prefixes bound to a numeric axis index.
const _FITS_AXIS_KEY_PREFIXES = ("CTYPE", "CRPIX", "CRVAL", "CDELT", "CUNIT",
                                 "CROTA", "CNAME", "CRDER", "CSYER", "PS",
                                 "WCSNAME")

# Keys the FITSIO writer rebuilds itself (`write(::FITS, arr; header=hdr)`
# refuses these in the user-supplied header). We strip them when copying.
const _FITS_FORBIDDEN_HEADER_KEYS = (
    "SIMPLE", "EXTEND", "BITPIX", "NAXIS",
    "NAXIS1", "NAXIS2", "NAXIS3", "NAXIS4",
    "XTENSION", "PCOUNT", "GCOUNT", "GROUPS",
    "TFIELDS", "THEAP", "END",
)

_is_forbidden_header_key(key::AbstractString) =
    uppercase(String(key)) in _FITS_FORBIDDEN_HEADER_KEYS

# True when `key` binds a header card to FITS axis number `axis`.
function _fits_key_binds_axis(key::AbstractString, axis::Integer)
    s = uppercase(String(key))
    suffix = string(axis)
    for prefix in _FITS_AXIS_KEY_PREFIXES
        s == prefix * suffix && return true
        prefix == "PS" && startswith(s, "PS" * suffix * "_") && return true
    end
    m = match(r"^(PC|CD)(\d+)_(\d+)$", s)
    if m !== nothing
        i = parse(Int, m.captures[2])
        j = parse(Int, m.captures[3])
        return i == axis || j == axis
    end
    m2 = match(r"^PV(\d+)_(\d+)$", s)
    if m2 !== nothing
        i = parse(Int, m2.captures[1])
        return i == axis
    end
    return false
end

# Renumber a WCS key bound to axis `from` to axis `to`. Pass-through otherwise.
function _fits_key_renumber(key::AbstractString, from::Integer, to::Integer)
    s = uppercase(String(key))
    suffix_from = string(from)
    for prefix in _FITS_AXIS_KEY_PREFIXES
        s == prefix * suffix_from && return prefix * string(to)
    end
    m = match(r"^(PC|CD)(\d+)_(\d+)$", s)
    if m !== nothing
        tag = m.captures[1]
        i = parse(Int, m.captures[2])
        j = parse(Int, m.captures[3])
        i == from && (i = to)
        j == from && (j = to)
        return string(tag, i, "_", j)
    end
    m2 = match(r"^PV(\d+)_(\d+)$", s)
    if m2 !== nothing
        i = parse(Int, m2.captures[1])
        i == from && (i = to)
        return string("PV", i, "_", m2.captures[2])
    end
    return String(key)
end

_set_history!(ks::Vector{String}, vs::Vector{Any}, cs::Vector{String}, text::AbstractString) = begin
    push!(ks, "HISTORY")
    push!(vs, nothing)
    push!(cs, String(text))
    return nothing
end

_set_comment_card!(ks::Vector{String}, vs::Vector{Any}, cs::Vector{String}, text::AbstractString) = begin
    push!(ks, "COMMENT")
    push!(vs, nothing)
    push!(cs, String(text))
    return nothing
end

_set_card!(ks::Vector{String}, vs::Vector{Any}, cs::Vector{String},
           key::AbstractString, value, comment::AbstractString = "") = begin
    target = uppercase(String(key))
    idx = findlast(k -> uppercase(k) == target, ks)
    if idx === nothing
        push!(ks, String(key))
        push!(vs, value)
        push!(cs, String(comment))
    else
        vs[idx] = value
        isempty(comment) || (cs[idx] = String(comment))
    end
    return nothing
end

# Read the (key, value, comment) triplet vectors from a FITSIO.FITSHeader.
# Falls back to iterating with `keys(hdr)` if the struct fields differ.
function _fits_header_triplets(src)
    if hasproperty(src, :keys) && hasproperty(src, :values) && hasproperty(src, :comments)
        return (collect(String, src.keys),
                Any[src.values[i] for i in eachindex(src.values)],
                collect(String, src.comments))
    end
    ks = collect(String, keys(src))
    vs = Any[src[k] for k in ks]
    cs = String[]
    for k in ks
        comment = try
            FITSIO.get_comment(src, k)
        catch
            ""
        end
        push!(cs, String(comment))
    end
    return (ks, vs, cs)
end

function _fits_header_value_string(v)
    v === nothing && return ""
    if v isa AbstractString
        return "'" * String(v) * "'"
    end
    return string(v)
end

"""
    fits_header_display_lines(src) -> Vector{String}

Format a FITS header as human-readable card lines for UI display. Returns a
single explanatory line when no header is available.
"""
function fits_header_display_lines(src)
    src === nothing && return ["(no FITS header available)"]
    ks, vs, cs = _fits_header_triplets(src)
    isempty(ks) && return ["(empty FITS header)"]
    lines = String[]
    @inbounds for i in eachindex(ks)
        key = uppercase(String(ks[i]))
        comment = strip(String(cs[i]))
        if key == "COMMENT" || key == "HISTORY"
            body = isempty(comment) ? _fits_header_value_string(vs[i]) : comment
            push!(lines, rpad(key, 8) * "  " * body)
        else
            line = rpad(key, 8) * "= " * _fits_header_value_string(vs[i])
            isempty(comment) || (line *= " / " * comment)
            push!(lines, line)
        end
    end
    return lines
end

# Copy the source header into fresh vectors, stripping the keys the FITSIO
# writer rebuilds (NAXIS, BITPIX, ...). Caller adds axis-specific changes.
function _copy_passthrough_header(src; drop_axes::Tuple = ())
    ks, vs, cs = _fits_header_triplets(src)
    out_k = String[]
    out_v = Any[]
    out_c = String[]
    @inbounds for i in eachindex(ks)
        k = ks[i]
        _is_forbidden_header_key(k) && continue
        skip = false
        for ax in drop_axes
            if _fits_key_binds_axis(k, ax)
                skip = true
                break
            end
        end
        skip && continue
        push!(out_k, k)
        push!(out_v, vs[i])
        push!(out_c, cs[i])
    end
    return out_k, out_v, out_c
end

function _renumber_axes!(ks::Vector{String}, mapping::AbstractDict)
    @inbounds for i in eachindex(ks)
        k = ks[i]
        for (from, to) in mapping
            from == to && continue
            if _fits_key_binds_axis(k, from)
                ks[i] = _fits_key_renumber(k, from, to)
                break
            end
        end
    end
    return nothing
end

# Spectral CUNIT lookup for moment BUNIT composition.
function _spectral_cunit(src, axis::Integer)
    src === nothing && return ""
    s = strip(String(header_get(src, "CUNIT$(axis)", "")))
    return String(s)
end

# Spectral CTYPE lookup, used for HISTORY context only.
function _spectral_ctype(src, axis::Integer)
    src === nothing && return ""
    s = strip(String(header_get(src, "CTYPE$(axis)", "")))
    return String(s)
end

"""
    fits_header_for_slice(src, axis, idx; source_id="") -> Union{Nothing,FITSHeader}

Build a 2D `FITSHeader` from the cube's primary header by dropping the WCS
keywords of the collapsed `axis` and renumbering the remaining axes so that
they are contiguous 1..2. Returns `nothing` when `src` is missing. The
original `BUNIT` and provenance keys (OBJECT/TELESCOP/INSTRUME/...) are
passed through, and a `HISTORY MANTA slice axis=… index=…` card is appended.
"""
function fits_header_for_slice(src, axis::Integer, idx::Integer;
                               source_id::AbstractString = "")
    src === nothing && return nothing
    axis in (1, 2, 3) || throw(ArgumentError("axis must be 1, 2, or 3"))
    ks, vs, cs = _copy_passthrough_header(src; drop_axes = (axis,))
    kept = filter(!=(axis), (1, 2, 3))
    _renumber_axes!(ks, Dict(kept[1] => 1, kept[2] => 2))
    msg = "MANTA slice axis=$(axis) index=$(idx)"
    isempty(source_id) || (msg *= " source=$(source_id)")
    _set_history!(ks, vs, cs, msg)
    return FITSIO.FITSHeader(ks, vs, cs)
end

"""
    fits_header_for_moment(src, axis, order; source_id="") -> Union{Nothing,FITSHeader}

Build a 2D `FITSHeader` for a moment map computed along `axis` (order 0/1/2).
Drops the collapsed axis WCS, renumbers the survivors and rewrites `BUNIT`
to reflect the moment definition used by `moments`:

* order 0: `BUNIT` stays equal to the source (`moments` sums samples, no
  `dv` factor) with a `COMMENT BUNIT` card describing the operation.
* order 1: `BUNIT` becomes the spectral `CUNIT` (intensity-weighted mean of
  the spectral coordinate). `?` if `CUNIT` is missing.
* order 2: same as order 1 (sqrt of intensity-weighted variance).

Always appends `HISTORY MANTA moment order=… axis=…`.
"""
function fits_header_for_moment(src, axis::Integer, order::Integer;
                                source_id::AbstractString = "")
    src === nothing && return nothing
    axis in (1, 2, 3) || throw(ArgumentError("axis must be 1, 2, or 3"))
    order in (0, 1, 2) || throw(ArgumentError("moment order must be 0, 1, or 2"))
    ks, vs, cs = _copy_passthrough_header(src; drop_axes = (axis,))
    kept = filter(!=(axis), (1, 2, 3))
    _renumber_axes!(ks, Dict(kept[1] => 1, kept[2] => 2))

    bunit_src = strip(String(header_get(src, "BUNIT", "")))
    cunit_ax = _spectral_cunit(src, axis)
    ctype_ax = _spectral_ctype(src, axis)

    new_bunit, bunit_comment = if order == 0
        unit = isempty(bunit_src) ? "?" : bunit_src
        (unit, "sum of samples along axis $(axis); no dv factor")
    elseif order == 1
        unit = isempty(cunit_ax) ? "?" : cunit_ax
        (unit, "intensity-weighted mean of $(isempty(ctype_ax) ? "axis $(axis)" : ctype_ax)")
    else
        unit = isempty(cunit_ax) ? "?" : cunit_ax
        (unit, "sqrt of intensity-weighted variance of $(isempty(ctype_ax) ? "axis $(axis)" : ctype_ax)")
    end

    _set_card!(ks, vs, cs, "BUNIT", new_bunit, "MANTA moment-$(order) unit")
    _set_comment_card!(ks, vs, cs, "BUNIT: " * bunit_comment)
    msg = "MANTA moment order=$(order) axis=$(axis)"
    isempty(source_id) || (msg *= " source=$(source_id)")
    _set_history!(ks, vs, cs, msg)
    return FITSIO.FITSHeader(ks, vs, cs)
end

"""
    fits_header_for_region_spectrum(src, axis, npix; source_id="") -> Union{Nothing,FITSHeader}

Build a 1D `FITSHeader` for a region-averaged spectrum. Keeps only the WCS of
the spectral `axis` (renumbered to axis 1) plus photometric/provenance keys.
SPECSYS/RESTFRQ/VELREF/ALTRVAL/ALTRPIX/ZSOURCE are copied as-is when present.
Appends `HISTORY MANTA region_spectrum axis=… npix=…`.
"""
function fits_header_for_region_spectrum(src, axis::Integer, npix::Integer;
                                         source_id::AbstractString = "")
    src === nothing && return nothing
    axis in (1, 2, 3) || throw(ArgumentError("axis must be 1, 2, or 3"))
    dropped = filter(!=(axis), (1, 2, 3))
    ks, vs, cs = _copy_passthrough_header(src; drop_axes = (dropped[1], dropped[2]))
    _renumber_axes!(ks, Dict(axis => 1))
    msg = "MANTA region_spectrum axis=$(axis) npix=$(npix)"
    isempty(source_id) || (msg *= " source=$(source_id)")
    _set_history!(ks, vs, cs, msg)
    return FITSIO.FITSHeader(ks, vs, cs)
end

"""
    fits_header_for_filtered_cube(src, axis, sigma; source_id="") -> Union{Nothing,FITSHeader}

Build a 3D `FITSHeader` for a per-slice Gaussian-filtered cube. WCS and
provenance keys are kept; only `HISTORY MANTA filtered axis=… sigma=…` is
appended.
"""
function fits_header_for_filtered_cube(src, axis::Integer, sigma::Real;
                                       source_id::AbstractString = "")
    src === nothing && return nothing
    axis in (1, 2, 3) || throw(ArgumentError("axis must be 1, 2, or 3"))
    ks, vs, cs = _copy_passthrough_header(src)
    msg = "MANTA filtered axis=$(axis) sigma=$(sigma)"
    isempty(source_id) || (msg *= " source=$(source_id)")
    _set_history!(ks, vs, cs, msg)
    return FITSIO.FITSHeader(ks, vs, cs)
end
