# path: src/helpers/WCS.jl
#
# Lightweight FITS WCS: per-axis `SimpleWCSAxis`, full `WCSTransform` with the
# CD / PC matrix, sky and spectral classification, pixel-to-world helpers
# (`world_coord`, `sky_world_coords`, `pixel_scale`), formatting
# (`wcs_axis_label`, `format_world_coord`, `data_unit_label`) and the shared
# `header_has` / `header_get` accessors reused by FITSHeaders.jl. Extracted
# from helpers/Helpers.jl.


############################
# Simple FITS WCS
############################

"""
    SimpleWCSAxis

Per-axis linear WCS description plus a small classifier derived from
`CTYPE`. Keeps the historical 6-arg constructor working — additional
fields (`ctype_base`, `projection`, `kind`, `spectral_quantity`) are
auto-populated by parsing the raw `CTYPE` string.

Field summary:

| field              | meaning                                                       |
| ------------------ | ------------------------------------------------------------- |
| `ctype`            | raw `CTYPEi` string (preserved for FITS round-trip & display) |
| `cunit`            | raw `CUNITi` string                                           |
| `crval`            | `CRVALi`                                                      |
| `crpix`            | `CRPIXi`                                                      |
| `cdelt`            | `CDELTi` (diagonal fallback when CD/PC is absent)             |
| `available`        | true if any of CTYPE / CRVAL / CDELT was actually present     |
| `ctype_base`       | "RA", "DEC", "GLON", "FREQ", … (CTYPE prefix, upper-cased)    |
| `projection`       | "TAN", "SIN", "CAR", "ARC", "ZEA", … or "" if linear          |
| `kind`             | `:ra | :dec | :glon | :glat | :elon | :elat | :spectral | :stokes | :unknown` |
| `spectral_quantity`| `:velocity | :frequency | :wavelength | :other` (for `:spectral` axes) |
"""
struct SimpleWCSAxis
    ctype::String
    cunit::String
    crval::Float64
    crpix::Float64
    cdelt::Float64
    available::Bool
    ctype_base::String
    projection::String
    kind::Symbol
    spectral_quantity::Symbol
end

function SimpleWCSAxis(ctype::AbstractString, cunit::AbstractString,
                       crval::Real, crpix::Real, cdelt::Real, available::Bool)
    s = String(ctype)
    base, proj = _split_ctype(s)
    k, qty = _classify_ctype(base)
    return SimpleWCSAxis(s, String(cunit), Float64(crval), Float64(crpix),
                         Float64(cdelt), Bool(available), base, proj, k, qty)
end

function _split_ctype(s::AbstractString)
    s = String(s)
    stripped = strip(s)
    isempty(stripped) && return ("", "")
    # FITS sky CTYPE format: "RA---TAN", "GLON-CAR", "DEC--SIN" etc.
    idx = findfirst(==('-'), s)
    if idx === nothing
        return (String(uppercase(stripped)), "")
    end
    base = uppercase(strip(s[1:idx-1]))
    proj_raw = s[idx:end]
    # collapse the run of '-' separators and pick whatever non-empty token follows
    proj = uppercase(strip(replace(proj_raw, '-' => "")))
    return (String(base), String(proj))
end

function _classify_ctype(base::AbstractString)
    up = uppercase(strip(String(base)))
    if up == "RA"
        return (:ra, :other)
    elseif up == "DEC"
        return (:dec, :other)
    elseif startswith(up, "GLON")
        return (:glon, :other)
    elseif startswith(up, "GLAT")
        return (:glat, :other)
    elseif startswith(up, "ELON")
        return (:elon, :other)
    elseif startswith(up, "ELAT")
        return (:elat, :other)
    elseif startswith(up, "VRAD") || startswith(up, "VOPT") ||
           startswith(up, "VELO") || up == "VELOCITY"
        return (:spectral, :velocity)
    elseif startswith(up, "FREQ") || startswith(up, "FELO")
        return (:spectral, :frequency)
    elseif startswith(up, "WAVE") || startswith(up, "AWAV")
        return (:spectral, :wavelength)
    elseif startswith(up, "STOKES")
        return (:stokes, :other)
    else
        return (:unknown, :other)
    end
end

header_has(header, key::AbstractString) = try
    haskey(header, String(key))
catch
    false
end

header_get(header, key::AbstractString, default) = header_has(header, key) ? header[String(key)] : default

"""
    read_simple_wcs(header, naxes) -> Vector{SimpleWCSAxis}

Read a lightweight linear WCS from FITS header keywords. This intentionally
handles the common `CTYPE/CRVAL/CRPIX/CDELT/CUNIT` case without requiring a
full WCS dependency. CD/PC matrices and projection-aware deprojections are
exposed via the richer [`WCSTransform`](@ref) returned by
[`read_wcs_transform`](@ref).
"""
function read_simple_wcs(header, naxes::Integer)
    axes = SimpleWCSAxis[]
    for dim in 1:naxes
        ctype = String(header_get(header, "CTYPE$(dim)", ""))
        cunit = String(header_get(header, "CUNIT$(dim)", ""))
        crval = Float64(header_get(header, "CRVAL$(dim)", 0.0))
        crpix = Float64(header_get(header, "CRPIX$(dim)", 1.0))
        cdelt = Float64(header_get(header, "CDELT$(dim)", 1.0))
        available = !isempty(ctype) || header_has(header, "CRVAL$(dim)") || header_has(header, "CDELT$(dim)")
        push!(axes, SimpleWCSAxis(ctype, cunit, crval, crpix, cdelt, available))
    end
    return axes
end

"""
    WCSTransform

Aggregates a `Vector{SimpleWCSAxis}` with the optional FITS CD matrix and
high-level helpers (which axes form the sky pair, which axis is spectral).

`WCSTransform` behaves like `Vector{SimpleWCSAxis}` for indexed lookups
(`wcs[dim]`, `length`, iteration), so existing call sites that expect that
shape keep working — see also the compat methods just below the struct.
The CD matrix is filled either from `CDi_j` headers (preferred) or from
`PCi_j × CDELTi` when CD is absent.
"""
struct WCSTransform
    axes::Vector{SimpleWCSAxis}
    cd::Union{Nothing, Matrix{Float64}}
    has_pc::Bool                       # true if the matrix was reconstructed from PC + CDELT
    sky_dims::Tuple{Int,Int}            # (lon_dim, lat_dim) or (0, 0)
    spectral_dim::Int                   # 0 if none
end

# Compat: `WCSTransform` is interchangeable with `Vector{SimpleWCSAxis}` at
# index / length / iterate sites used throughout the viewers.
Base.length(w::WCSTransform) = length(w.axes)
Base.getindex(w::WCSTransform, i::Integer) = w.axes[i]
Base.iterate(w::WCSTransform, st::Integer = 1) =
    st > length(w.axes) ? nothing : (w.axes[st], st + 1)
Base.eachindex(w::WCSTransform) = eachindex(w.axes)
Base.firstindex(w::WCSTransform) = 1
Base.lastindex(w::WCSTransform) = length(w.axes)
Base.keys(w::WCSTransform) = eachindex(w.axes)

"""
    read_wcs_transform(header, naxes) -> WCSTransform

Read both the per-axis WCS and the optional CD/PC linear transform. CD
takes precedence; otherwise PC is multiplied by the diagonal CDELT to
reconstruct an equivalent CD matrix. Falls back gracefully when neither
matrix is present — the returned transform then carries `cd = nothing`
and the consumers downgrade to the diagonal CDELT.
"""
function read_wcs_transform(header, naxes::Integer)
    n = Int(naxes)
    axes_v = read_simple_wcs(header, n)
    cd = nothing
    has_cd = false
    for i in 1:n, j in 1:n
        if header_has(header, "CD$(i)_$(j)")
            has_cd = true
            break
        end
    end
    if has_cd
        cd = zeros(Float64, n, n)
        # Seed diagonal with CDELT in case the file only specifies the off-diagonals.
        for i in 1:n
            cd[i, i] = axes_v[i].cdelt
        end
        for i in 1:n, j in 1:n
            k = "CD$(i)_$(j)"
            if header_has(header, k)
                cd[i, j] = Float64(header_get(header, k, 0.0))
            end
        end
    end
    has_pc = false
    if cd === nothing
        for i in 1:n, j in 1:n
            if header_has(header, "PC$(i)_$(j)")
                has_pc = true
                break
            end
        end
        if has_pc
            pc = zeros(Float64, n, n)
            for i in 1:n
                pc[i, i] = 1.0
            end
            for i in 1:n, j in 1:n
                k = "PC$(i)_$(j)"
                header_has(header, k) && (pc[i, j] = Float64(header_get(header, k, i == j ? 1.0 : 0.0)))
            end
            cd = similar(pc)
            for i in 1:n, j in 1:n
                cd[i, j] = pc[i, j] * axes_v[i].cdelt
            end
        end
    end
    return WCSTransform(axes_v, cd, has_pc, _sky_pair(axes_v), _spectral_dim(axes_v))
end

function _sky_pair(axes_v)
    lon = 0
    lat = 0
    for (i, ax) in pairs(axes_v)
        if ax.kind in (:ra, :glon, :elon)
            lon = i
        elseif ax.kind in (:dec, :glat, :elat)
            lat = i
        end
    end
    return (lon, lat)
end

function _spectral_dim(axes_v)
    for (i, ax) in pairs(axes_v)
        ax.kind === :spectral && return i
    end
    return 0
end

"""
    sky_dims(wcs) -> Tuple{Int,Int}

Return `(lon_dim, lat_dim)` for the celestial pair, or `(0, 0)` if no pair
is identified. Accepts either a `WCSTransform` or a plain
`Vector{SimpleWCSAxis}`.
"""
sky_dims(wcs::WCSTransform) = wcs.sky_dims
sky_dims(wcs::AbstractVector{<:SimpleWCSAxis}) = _sky_pair(wcs)

"""
    spectral_dim(wcs) -> Int

Return the spectral axis index (1-based) or `0` when none is identified.
"""
spectral_dim(wcs::WCSTransform) = wcs.spectral_dim
spectral_dim(wcs::AbstractVector{<:SimpleWCSAxis}) = _spectral_dim(wcs)

"""
    spectral_quantity(wcs, dim) -> Symbol

Return `:velocity | :frequency | :wavelength | :other`. `has_wcs(wcs, dim)`
must be true; otherwise returns `:other`.
"""
spectral_quantity(wcs, dim::Integer) =
    has_wcs(wcs, dim) ? wcs[dim].spectral_quantity : :other

"""
    spectral_quantity_word(qty) -> String

Map a spectral-quantity symbol to a human label suitable for axis titles
and moment captions: `"velocity"`, `"frequency"`, `"wavelength"`, or the
generic fallback `"value"`.
"""
spectral_quantity_word(qty::Symbol) =
    qty === :velocity ? "velocity" :
    qty === :frequency ? "frequency" :
    qty === :wavelength ? "wavelength" : "value"

has_wcs(wcs, dim::Integer) = 1 <= dim <= length(wcs) && wcs[dim].available

world_coord(wcs, dim::Integer, pix::Real) =
    has_wcs(wcs, dim) ? wcs[dim].crval + (Float64(pix) - wcs[dim].crpix) * wcs[dim].cdelt : Float64(pix)

"""
    pixel_scale(wcs, dim; ref=nothing) -> Float64

Pixel scale along axis `dim`, in the axis' native unit (e.g. degrees for
sky axes, Hz for `FREQ`). For longitude axes (`RA`, `GLON`, `ELON`) the
scale is multiplied by `cos(latitude)`, evaluated at the latitude axis'
`CRVAL` (or at the pixel value supplied via `ref` if you want the scale
at a specific latitude). For a `WCSTransform` carrying a CD matrix, the
column-norm of CD is used so off-diagonal terms (rotated FITS frames) are
honoured; otherwise the diagonal `CDELT` is the fallback.
"""
function pixel_scale(wcs::AbstractVector{<:SimpleWCSAxis}, dim::Integer; ref = nothing)
    1 <= dim <= length(wcs) || return 1.0
    return abs(wcs[dim].cdelt) * _cos_lat_factor(wcs, dim, ref)
end

function pixel_scale(wcs::WCSTransform, dim::Integer; ref = nothing)
    1 <= dim <= length(wcs.axes) || return 1.0
    base = if wcs.cd === nothing
        abs(wcs.axes[dim].cdelt)
    else
        s = 0.0
        @inbounds for i in 1:length(wcs.axes)
            s += wcs.cd[i, dim]^2
        end
        sqrt(s)
    end
    return base * _cos_lat_factor(wcs, dim, ref)
end

function _cos_lat_factor(wcs, dim::Integer, ref)
    ax = wcs[dim]
    ax.kind in (:ra, :glon, :elon) || return 1.0
    lat_ax = nothing
    lat_idx = 0
    for (i, a) in pairs(wcs)
        if a.kind in (:dec, :glat, :elat)
            lat_ax = a
            lat_idx = i
            break
        end
    end
    lat_ax === nothing && return 1.0
    lat_deg = ref === nothing ? lat_ax.crval : world_coord(wcs, lat_idx, ref)
    return cosd(Float64(lat_deg))
end

"""
    sky_world_coords(wcs::WCSTransform, pix1, pix2) -> (lon_deg, lat_deg) | nothing

Apply the 2-D celestial WCS — CD matrix plus the projection encoded in
`CTYPE` — to a pair of pixel coordinates and return `(longitude, latitude)`
in degrees. Returns `nothing` if the transform has no identified sky pair
or if the deprojection falls outside its valid domain (e.g. ρ > 1 for
SIN).

Supported projections: `TAN` (gnomonic), `SIN` (orthographic),
`CAR` (plate carrée / equirectangular), and a linear pass-through for anything else.
"""
function sky_world_coords(wcs::WCSTransform, pix1::Real, pix2::Real)
    lon_d, lat_d = wcs.sky_dims
    (lon_d == 0 || lat_d == 0) && return nothing
    lon_ax = wcs.axes[lon_d]
    lat_ax = wcs.axes[lat_d]
    n = length(wcs.axes)
    dpix = zeros(Float64, n)
    dpix[lon_d] = Float64(pix1) - lon_ax.crpix
    dpix[lat_d] = Float64(pix2) - lat_ax.crpix
    if wcs.cd === nothing
        xi  = dpix[lon_d] * lon_ax.cdelt
        eta = dpix[lat_d] * lat_ax.cdelt
    else
        v = wcs.cd * dpix
        xi  = v[lon_d]
        eta = v[lat_d]
    end
    return _deproject_sky(uppercase(lon_ax.projection), lon_ax.crval, lat_ax.crval, xi, eta)
end

function _deproject_sky(proj::AbstractString, lon0_deg::Real, lat0_deg::Real,
                        xi_deg::Real, eta_deg::Real)
    if proj == "TAN" || isempty(proj)
        ξ = deg2rad(xi_deg); η = deg2rad(eta_deg)
        ρ = sqrt(ξ^2 + η^2)
        ρ == 0 && return (Float64(lon0_deg), Float64(lat0_deg))
        c = atan(ρ)
        lat0 = deg2rad(lat0_deg); lon0 = deg2rad(lon0_deg)
        sinφ = cos(c) * sin(lat0) + (η * sin(c) * cos(lat0)) / ρ
        φ = asin(clamp(sinφ, -1.0, 1.0))
        λ = lon0 + atan(ξ * sin(c), ρ * cos(lat0) * cos(c) - η * sin(lat0) * sin(c))
        return (rad2deg(λ), rad2deg(φ))
    elseif proj == "SIN"
        ξ = deg2rad(xi_deg); η = deg2rad(eta_deg)
        ρ = sqrt(ξ^2 + η^2)
        ρ > 1 && return nothing
        ρ == 0 && return (Float64(lon0_deg), Float64(lat0_deg))
        c = asin(ρ)
        lat0 = deg2rad(lat0_deg); lon0 = deg2rad(lon0_deg)
        sinφ = cos(c) * sin(lat0) + (η * sin(c) * cos(lat0)) / ρ
        φ = asin(clamp(sinφ, -1.0, 1.0))
        λ = lon0 + atan(ξ * sin(c), ρ * cos(lat0) * cos(c) - η * sin(lat0) * sin(c))
        return (rad2deg(λ), rad2deg(φ))
    elseif proj == "CAR"
        return (Float64(lon0_deg) + Float64(xi_deg),
                Float64(lat0_deg) + Float64(eta_deg))
    else
        # Unknown projection → linear fall-through; viewers should
        # treat the result as approximate.
        return (Float64(lon0_deg) + Float64(xi_deg),
                Float64(lat0_deg) + Float64(eta_deg))
    end
end

function wcs_axis_label(wcs, dim::Integer; fallback::AbstractString = "pixel")
    if !has_wcs(wcs, dim)
        return latexstring("\\text{", latex_safe(fallback), "}")
    end
    ax = wcs[dim]
    name = if ax.kind === :ra
        "RA"
    elseif ax.kind === :dec
        "Dec"
    elseif ax.kind === :glon
        "Galactic longitude"
    elseif ax.kind === :glat
        "Galactic latitude"
    elseif ax.kind === :elon
        "Ecliptic longitude"
    elseif ax.kind === :elat
        "Ecliptic latitude"
    elseif ax.kind === :spectral
        spectral_quantity_word(ax.spectral_quantity)
    elseif isempty(ax.ctype_base)
        "world $(dim)"
    else
        ax.ctype_base
    end
    unit = isempty(ax.cunit) ? "" : " [$(ax.cunit)]"
    return latexstring("\\text{", latex_safe(name * unit), "}")
end

function format_world_coord(wcs, dim::Integer, pix::Real)
    if !has_wcs(wcs, dim)
        return "pix$(dim)=" * string(round(Float64(pix); digits = 2))
    end
    ax = wcs[dim]
    # CTYPE/CUNIT can be whitespace-only strings (malformed FITS header);
    # normalise them to avoid ugly "  =0.0" labels.
    ctype_clean = strip(String(ax.ctype))
    ctype = isempty(ctype_clean) ? "axis$(dim)" : ctype_clean
    val = world_coord(wcs, dim, pix)
    unit_clean = strip(String(ax.cunit))
    unit = isempty(unit_clean) ? "" : " $(unit_clean)"
    return "$(ctype)=" * string(round(val; digits = 5)) * unit
end

"""
    data_unit_label(header; fallback="value") -> String

Return the FITS image data unit from `BUNIT` when present, otherwise a
generic fallback label for scalar pixel values.
"""
function data_unit_label(header; fallback::AbstractString = "value")
    header === nothing && return String(fallback)
    unit = header_get(header, "BUNIT", "")
    s = strip(String(unit))
    return isempty(s) ? String(fallback) : s
end
