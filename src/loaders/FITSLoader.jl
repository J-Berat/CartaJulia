# path: src/loaders/FITSLoader.jl
#
# Convert a FITS file path into an `AbstractMANTADataset`.
#
# Centralizes FITS reads previously inlined in `MANTA.manta(filepath)`
# and at the top of the HEALPix viewers. The dispatch order — HEALPix-PPV
# 2D cube first, then HEALPix 1D map, then 3D cube — and all error texts
# match the original behavior. The integration smoke testset is the
# regression contract.

using FITSIO

# Public entry. `kwargs` mirror the `manta(filepath)` knobs that the loader
# itself needs (HEALPix column, spectral-axis defaults, HDU selection).
# Viewer-only kwargs (cmap, vmin, …) are not handled here.
#
# `hdu` selects which FITS HDU to read (1-indexed). `hdu = 0` means
# "auto-pick the first HDU that looks like an image/cube" — historical
# behavior for files that put the data after a header-only primary HDU.
function load_fits(
    filepath::AbstractString;
    column::Int = 1,
    v0::Real = 0.0,
    dv::Real = 1.0,
    vunit::AbstractString = "km/s",
    hdu::Integer = 1,
    lazy::Bool = false,
)
    require_file(filepath;
        hint = "Vérifie le chemin (cwd = $(pwd())) ou utilise un chemin absolu.")

    Int(hdu) >= 0 || invalid_kwarg(:hdu, hdu;
        hint = "Doit être un entier ≥ 0 (1 = HDU primaire, 0 = auto-détection).")

    # Lazy path: shortcut HEALPix detection, hand off to the lazy loader,
    # and wrap the result in the appropriate dataset (without ever
    # reading the full data array).
    if lazy
        Int(hdu) >= 1 || invalid_kwarg(:hdu, hdu;
            hint = "Le chargement lazy nécessite un HDU explicite ≥ 1.")
        header, arr, _ = open_lazy_fits(filepath; hdu = Int(hdu))
        if arr isa LazyFITSImage
            return _load_image_fits_lazy(filepath, arr, header)
        else  # LazyFITSCube
            return _load_cube_fits_lazy(filepath, arr, header)
        end
    end

    # Read selected HDU once. Tolerates an empty/header-only HDU (HEALPix
    # BinTable files have no image in HDU 1). We capture the original
    # exception so we can surface it via `UnsupportedFormatError` if needed.
    hdu_index = max(Int(hdu), 1)   # 0 means "auto" — we still start at 1
    auto_pick = Int(hdu) == 0
    header = nothing
    primary_error = nothing
    n_hdus = 0
    raw = try
        FITS(filepath) do f
            n_hdus = length(f)
            if auto_pick
                # Walk HDUs and pick the first one that yields a non-empty array.
                for k in 1:n_hdus
                    try
                        cand = read(f[k])
                        if cand !== nothing && ndims(cand) > 0
                            hdu_index = k
                            header = try read_header(f[k]) catch _ nothing end
                            return cand
                        end
                    catch _
                    end
                end
                return nothing
            else
                hdu_index <= n_hdus || invalid_hdu(filepath, hdu_index, n_hdus)
                header = try read_header(f[hdu_index]) catch _ nothing end
                return read(f[hdu_index])
            end
        end
    catch e
        e isa MANTAError && rethrow()
        primary_error = e
        nothing
    end

    # 1) HEALPix-PPV 2D cube: tested BEFORE is_healpix_fits because such cubes
    #    embed PIXTYPE=HEALPIX in their primary header.
    if raw !== nothing && ndims(raw) == 2
        s = size(raw)
        if valid_healpix_npix(s[1]) > 0 || valid_healpix_npix(s[2]) > 0
            return _load_healpix_cube_fits(filepath, raw, header, v0, dv, vunit)
        end
    end

    # 2) HEALPix 1D map (BinTable).
    if is_healpix_fits(filepath)
        return _load_healpix_map_fits(filepath; column = column)
    end

    # 3) Image / cube / vector. If we never got the HDU, raise a structured
    #    error pointing at the most likely fix (try another HDU, fix the file).
    if raw === nothing
        msg = primary_error === nothing ?
            "HDU vide ou illisible." :
            sprint(showerror, primary_error)
        throw(UnsupportedFormatError(
            String(filepath),
            "FITS",
            "Échec lecture HDU #$(hdu_index)" *
            (n_hdus > 0 ? " (le fichier en contient $(n_hdus))." : ".") *
            "\n     Essaie un autre `hdu=...` ou vérifie l'intégrité du fichier." *
            "\n     Détail: $(msg)"))
    end
    if ndims(raw) == 1
        return _load_vector_fits(filepath, raw, header)
    end
    if ndims(raw) == 2
        return _load_image_fits(filepath, raw, header)
    end
    if ndims(raw) != 3
        throw(DatasetShapeError(
            "tableau FITS de dimension $(ndims(raw)) (taille $(size(raw))) " *
            "dans $(abspath(filepath)).",
            "MANTA gère 1D (vecteur), 2D (image) et 3D (cube). " *
            "Réduis la dimensionnalité ou choisis un autre `hdu=...`."))
    end
    return _load_cube_fits(filepath, raw, header)
end

# ---- internal helpers (one per dataset kind) ----

function _load_vector_fits(filepath::AbstractString, raw, header)
    data = as_float32(raw)
    # Parse the WCS for axis 1. `read_simple_wcs` always returns a length-1
    # vector here; we keep the axis only when it is actually available so the
    # viewer's fast `ds.wcs !== nothing` branch implies a usable mapping.
    wcs_axes = header === nothing ? SimpleWCSAxis[] : read_simple_wcs(header, 1)
    raw_axis = isempty(wcs_axes) ? nothing : wcs_axes[1]
    wcs_axis = (raw_axis === nothing || !raw_axis.available) ? nothing : raw_axis
    # `axis_label` is stored as a plain String on the dataset. When WCS is
    # absent, fall back to "index"; otherwise expose the raw CTYPE base name
    # (with unit) so downstream tools can echo it back without parsing LaTeX.
    axis_label = if wcs_axis === nothing
        "index"
    else
        name = isempty(wcs_axis.ctype_base) ? "axis1" : wcs_axis.ctype_base
        isempty(wcs_axis.cunit) ? name : "$(name) [$(wcs_axis.cunit)]"
    end
    unit_label = data_unit_label(header; fallback = "value")
    fname = String(replace(basename(filepath), r"\.fits(\.gz)?$" => ""))
    meta = Dict{Symbol,Any}(:fits_header => header,
                            :fits_path => abspath(String(filepath)))
    return VectorDataset(data;
        axis_label = axis_label,
        wcs = wcs_axis,
        unit_label = unit_label,
        source_id = fname,
        metadata = meta,
    )
end

function _load_image_fits(filepath::AbstractString, raw, header)
    data = as_float32(raw)
    wcs = header === nothing ? SimpleWCSAxis[] : read_simple_wcs(header, 2)
    wcs_xform = header === nothing ? nothing : read_wcs_transform(header, 2)
    unit_label = data_unit_label(header; fallback = "value")
    fname = String(replace(basename(filepath), r"\.fits(\.gz)?$" => ""))
    meta = Dict{Symbol,Any}(:fits_header => header,
                            :fits_path => abspath(String(filepath)))
    wcs_xform === nothing || (meta[:wcs_transform] = wcs_xform)
    return ImageDataset(data;
        axis_labels = ["axis1", "axis2"],
        wcs = wcs,
        unit_label = unit_label,
        source_id = fname,
        metadata = meta,
    )
end

function _load_cube_fits(filepath::AbstractString, raw, header)
    data = as_float32(raw)
    wcs = header === nothing ? SimpleWCSAxis[] : read_simple_wcs(header, 3)
    wcs_xform = header === nothing ? nothing : read_wcs_transform(header, 3)
    unit_label = data_unit_label(header; fallback = "value")
    fname_full = basename(filepath)
    fname = String(replace(fname_full, r"\.fits(\.gz)?$" => ""))
    meta = Dict{Symbol,Any}(:fits_header => header,
                            :fits_path => abspath(String(filepath)))
    wcs_xform === nothing || (meta[:wcs_transform] = wcs_xform)
    return CubeDataset(data;
        axis_labels = ["axis1", "axis2", "axis3"],
        wcs = wcs,
        unit_label = unit_label,
        source_id = fname,
        metadata = meta,
    )
end

# ---- lazy variants (no eager `as_float32`) ----
#
# These mirror the dataset shape of the eager loaders but keep the
# underlying array as a `LazyFITSImage` / `LazyFITSCube`. The rest of
# MANTA only touches the data via `get_slice_*` / `view(...)`, so a
# lazy array is observationally equivalent — modulo per-slice I/O cost.

function _load_image_fits_lazy(filepath::AbstractString, lazy_img::LazyFITSImage,
                                header)
    wcs = header === nothing ? SimpleWCSAxis[] : read_simple_wcs(header, 2)
    wcs_xform = header === nothing ? nothing : read_wcs_transform(header, 2)
    unit_label = data_unit_label(header; fallback = "value")
    fname = String(replace(basename(filepath), r"\.fits(\.gz)?$" => ""))
    meta = Dict{Symbol,Any}(:fits_header => header,
                            :fits_path => abspath(String(filepath)),
                            :lazy => true)
    wcs_xform === nothing || (meta[:wcs_transform] = wcs_xform)
    return ImageDataset(lazy_img;
        axis_labels = ["axis1", "axis2"],
        wcs = wcs,
        unit_label = unit_label,
        source_id = fname,
        metadata = meta,
    )
end

function _load_cube_fits_lazy(filepath::AbstractString, lazy_cube::LazyFITSCube,
                               header)
    wcs = header === nothing ? SimpleWCSAxis[] : read_simple_wcs(header, 3)
    wcs_xform = header === nothing ? nothing : read_wcs_transform(header, 3)
    unit_label = data_unit_label(header; fallback = "value")
    fname = String(replace(basename(filepath), r"\.fits(\.gz)?$" => ""))
    meta = Dict{Symbol,Any}(:fits_header => header,
                            :fits_path => abspath(String(filepath)),
                            :lazy => true)
    wcs_xform === nothing || (meta[:wcs_transform] = wcs_xform)
    return CubeDataset(lazy_cube;
        axis_labels = ["axis1", "axis2", "axis3"],
        wcs = wcs,
        unit_label = unit_label,
        source_id = fname,
        metadata = meta,
    )
end

function _load_healpix_map_fits(filepath::AbstractString; column::Int = 1)
    m, hdr = read_healpix_map(filepath; column = column)
    fname = String(replace(basename(filepath), r"\.fits(\.gz)?$" => ""))
    unit_str = strip(String(get(hdr, "TUNIT$column", get(hdr, "BUNIT", ""))))
    unit_label = isempty(unit_str) ? "value" : String(unit_str)
    return HealpixMapDataset(m;
        column = column,
        unit_label = unit_label,
        source_id = fname,
        metadata = Dict{Symbol,Any}(:fits_header => hdr,
                                    :fits_path => abspath(String(filepath))),
    )
end

# Mirrors the prologue of the HEALPix-PPV cube viewer: detects which dim is
# the velocity axis, orients the cube to (npix, nv), and computes
# (v0, dv, vunit).
function _load_healpix_cube_fits(
    filepath::AbstractString, raw, header,
    v0::Real, dv::Real, vunit::AbstractString,
)
    ndims(raw) == 2 || throw(ArgumentError(
        "MANTA: expected 2D array (npix×nv), got ndims=$(ndims(raw))"))
    data_unit = data_unit_label(header; fallback = "value")

    s = size(raw)
    nside1 = valid_healpix_npix(s[1])
    nside2 = valid_healpix_npix(s[2])

    user_set_wcs = !(v0 == 0.0 && dv == 1.0)
    wcs_info = user_set_wcs ? nothing : detect_velocity_axis(filepath, 2)

    nside, npix, nv, vaxis, v0_eff, dv_eff, vunit_eff = if wcs_info !== nothing
        (vax, v0_h, dv_h, unit_h) = wcs_info
        hpix_dim = vax == 1 ? 2 : 1
        nside_h = valid_healpix_npix(s[hpix_dim])
        nside_h == 0 && throw(ArgumentError(
            "MANTA: header indique CTYPE$(vax) spectral mais NAXIS$(hpix_dim)=$(s[hpix_dim]) " *
            "n'est pas un npix HEALPix valide (12·nside²)."))
        vax_sym = vax == 2 ? :last : :first
        @info "Velocity axis from FITS header" fits_axis=vax CRVAL=v0_h CDELT=dv_h CUNIT=unit_h
        (nside_h, s[hpix_dim], s[vax], vax_sym, Float64(v0_h), Float64(dv_h), String(unit_h))
    elseif nside1 > 0
        (nside1, s[1], s[2], :last,  Float64(v0), Float64(dv), String(vunit))
    elseif nside2 > 0
        (nside2, s[2], s[1], :first, Float64(v0), Float64(dv), String(vunit))
    else
        throw(ArgumentError(
            "MANTA: neither dimension of $(s) is a valid HEALPix npix=12·nside² " *
            "and no spectral CTYPE in header."))
    end

    no_wcs = (wcs_info === nothing) && !user_set_wcs
    if no_wcs
        vunit_eff = "channel"
        v0_eff = 1.0
        dv_eff = 1.0
    end
    @info "HEALPix PPV cube" path=abspath(filepath) nside npix nv vaxis v0=v0_eff dv=dv_eff unit=vunit_eff

    cube = vaxis === :last ? as_float32(raw) : as_float32(permutedims(raw))
    fname = String(replace(basename(filepath), r"\.fits(\.gz)?$" => ""))

    return HealpixCubeDataset(cube;
        nside = nside,
        v0 = v0_eff, dv = dv_eff, vunit = vunit_eff,
        unit_label = data_unit,
        source_id = fname,
        metadata = Dict{Symbol,Any}(:fits_header => header,
                                    :fits_path => abspath(String(filepath))),
    )
end
