# path: src/loaders/HDF5Loader.jl
#
# Convert an HDF5 path + internal address ("file.h5:/group/dataset") into an
# `AbstractMANTADataset`. Metadata is read from HDF5 attributes attached to
# the addressed dataset (or its parent group).
#
# Attribute conventions (case-insensitive lookup):
#   - "units" / "bunit"           → unit_label (fallback "value")
#   - "AXIS{i}NAME"               → axis_labels[i]
#   - "CTYPE{i}" / "CRVAL{i}" /
#     "CRPIX{i}" / "CDELT{i}" /
#     "CUNIT{i}"                  → SimpleWCSAxis (linear WCS)
#   - "PIXTYPE" == "HEALPIX" or
#     "healpix" truthy            → HEALPix routing
#   - "v0", "dv", "vunit"         → spectral axis for HEALPix-PPV cubes
#   - "ORDERING", "NSIDE",
#     "COORDSYS"                  → kept in metadata for HEALPix tools

using HDF5

"""
    parse_hdf5_spec(s) -> (path, address) | nothing

Splits "file.h5:/group/ds" on the LAST `:`, but only when the prefix has an
HDF5 extension (.h5 / .hdf5 / .he5) and the suffix begins with `/`.
This rejects Windows drive letters ("C:/path/file.h5") and plain FITS paths.

Kept as a thin wrapper around the shared `parse_path_spec` helper so callers
that want HDF5-only behaviour keep a tight API.
"""
function parse_hdf5_spec(s::AbstractString)
    kind, args... = parse_path_spec(s)
    if kind === :hdf5 && length(args) == 2
        return (String(args[1]), String(args[2]))
    end
    return nothing
end

# --- Dict-based attribute helpers ---------------------------------------------

"""
    read_attrs(obj) -> Dict{String,Any}

Snapshot HDF5 attributes attached to `obj` into a plain Julia `Dict` so the
loader can drop the file handle as soon as possible. Any attribute that
fails to read is mapped to the empty string instead of raising; the loader
itself can re-validate when it tries to interpret a value.
"""
function read_attrs(obj)
    out = Dict{String,Any}()
    try
        attrs = HDF5.attributes(obj)
        for k in HDF5.keys(attrs)
            key = String(k)
            try
                out[key] = read(attrs[key])
            catch
                out[key] = ""
            end
        end
    catch
        # No attributes on this object: leave the dict empty.
    end
    return out
end

"""
    get_attr(attrs::Dict, key, default=nothing)

Case-insensitive attribute fetch. Returns `default` when no key matches.
"""
function get_attr(attrs::AbstractDict, key::AbstractString, default = nothing)
    haskey(attrs, key) && return attrs[key]
    target = lowercase(String(key))
    for (k, v) in attrs
        lowercase(String(k)) == target && return v
    end
    return default
end

"""
    get_attr_first(attrs::Dict, keys, default=nothing)

Try each candidate key in order (case-insensitive) and return the first value
that resolves. Useful for attributes that may live under several spellings
(`units` / `UNITS` / `bunit` / `BUNIT`).
"""
function get_attr_first(attrs::AbstractDict, keys, default = nothing)
    for k in keys
        v = get_attr(attrs, String(k), nothing)
        v !== nothing && return v
    end
    return default
end

function get_attr_str(attrs::AbstractDict, key::AbstractString, default::AbstractString = "")
    v = get_attr(attrs, key, nothing)
    v === nothing && return String(default)
    try
        return String(strip(string(v)))
    catch
        return String(default)
    end
end

function get_attr_float(attrs::AbstractDict, key::AbstractString, default::Real)
    v = get_attr(attrs, key, nothing)
    v === nothing && return Float64(default)
    try
        return Float64(v)
    catch
        return Float64(default)
    end
end

function _build_wcs_from_h5_attrs(attrs::AbstractDict, ndim::Int)
    axes = SimpleWCSAxis[]
    any_present = false
    for dim in 1:ndim
        ctype = get_attr_str(attrs, "CTYPE$(dim)", "")
        cunit = get_attr_str(attrs, "CUNIT$(dim)", "")
        crval = get_attr_float(attrs, "CRVAL$(dim)", 0.0)
        crpix = get_attr_float(attrs, "CRPIX$(dim)", 1.0)
        cdelt = get_attr_float(attrs, "CDELT$(dim)", 1.0)
        present = !isempty(ctype) ||
                  get_attr(attrs, "CRVAL$(dim)", nothing) !== nothing ||
                  get_attr(attrs, "CDELT$(dim)", nothing) !== nothing
        any_present |= present
        push!(axes, SimpleWCSAxis(ctype, cunit, crval, crpix, cdelt, present))
    end
    return any_present ? axes : SimpleWCSAxis[]
end

function _axis_labels_from_h5_attrs(attrs::AbstractDict, ndim::Int)
    out = String[]
    for dim in 1:ndim
        name = get_attr_str(attrs, "AXIS$(dim)NAME", "")
        push!(out, isempty(name) ? "axis$(dim)" : name)
    end
    return out
end

function _is_healpix_h5(attrs::AbstractDict)
    pixtype = uppercase(get_attr_str(attrs, "PIXTYPE", ""))
    pixtype == "HEALPIX" && return true
    v = get_attr(attrs, "healpix", nothing)
    if v !== nothing
        try
            return Bool(v)
        catch
            return false
        end
    end
    return false
end

function _select_hdf5_dataset(obj, address::AbstractString)
    if obj isa HDF5.Dataset
        return obj
    elseif obj isa HDF5.Group || obj isa HDF5.File
        # Look for a single dataset child or an attribute "default_dataset".
        default = try
            read_attribute(obj, "default_dataset")
        catch
            nothing
        end
        if default !== nothing && haskey(obj, String(default))
            child = obj[String(default)]
            child isa HDF5.Dataset || throw(ArgumentError(
                "MANTA: HDF5 default_dataset=$(default) in $(address) is not a dataset."))
            return child
        end
        children = collect(HDF5.keys(obj))
        ds_children = filter(k -> obj[k] isa HDF5.Dataset, children)
        length(ds_children) == 1 ||
            throw(ArgumentError(
                "MANTA: HDF5 group $(address) is ambiguous: contains $(length(ds_children)) datasets " *
                "($(ds_children)). Specify the full address."))
        return obj[ds_children[1]]
    else
        throw(ArgumentError("MANTA: HDF5 object at $(address) is neither a dataset nor a group"))
    end
end

function _hdf5_chunk_tuple(ds::HDF5.Dataset, nd::Int)
    HDF5.ischunked(ds) || return nothing
    ch = Int.(HDF5.get_chunk(ds))
    length(ch) == nd || return nothing
    return nd == 2 ? (ch[1], ch[2]) :
           nd == 3 ? (ch[1], ch[2], ch[3]) :
                     nothing
end

function _lazy_hdf5_array(path::AbstractString, address::AbstractString,
                          sz::Tuple, chunk)
    length(sz) == 2 && return LazyHDF5Image{Float32}(
        String(path), String(address), (Int(sz[1]), Int(sz[2])), chunk)
    length(sz) == 3 && return LazyHDF5Cube{Float32}(
        String(path), String(address), (Int(sz[1]), Int(sz[2]), Int(sz[3])), chunk)
    throw(ArgumentError(
        "MANTA: HDF5 lazy loading only supports 2D images and 3D cubes " *
        "(dataset $(address) has ndims=$(length(sz)))."))
end

"""
    load_hdf5(path[, address]; column=1, v0=0.0, dv=1.0, vunit="km/s", lazy=false)

Open `path` (HDF5 file), navigate to `address` (defaults to `"/"`), read the
dataset metadata, and build an `AbstractMANTADataset`. With `lazy=true`, 2-D
images and 3-D cubes are wrapped for on-demand hyperslab reads instead of
materialising the full dataset. When `address` points to a group, the loader
picks a single child dataset (if unambiguous) or follows the `default_dataset`
attribute when present.
"""
load_hdf5(path::AbstractString; kwargs...) = load_hdf5(path, "/"; kwargs...)

function load_hdf5(
    path::AbstractString,
    address::AbstractString;
    column::Int = 1,
    v0::Real = 0.0,
    dv::Real = 1.0,
    vunit::AbstractString = "km/s",
    # `hdu` is FITS-only; accepted here so `load_dataset` can forward kwargs
    # uniformly without filtering.  It is silently ignored for HDF5 files.
    hdu::Integer = 1,
    lazy::Bool = false,
)
    isfile(path) || throw(FileNotFoundError(String(path),
        "Check the path (cwd = $(pwd())) or use an absolute path."))

    data, attrs_dict, resolved_address = try
        h5open(path, "r") do f
            if !haskey(f, address) && address != "/"
                throw(ArgumentError("MANTA: HDF5 address $(address) not found in $(abspath(path))"))
            end
            obj = address == "/" ? f : f[address]
            ds = _select_hdf5_dataset(obj, address)
            attrs = read_attrs(ds)
            resolved = String(HDF5.name(ds))
            if lazy && !_is_healpix_h5(attrs)
                sz = size(ds)
                nd = length(sz)
                nd in (2, 3) || throw(ArgumentError(
                    "MANTA: HDF5 lazy loading only supports 2D images and 3D cubes " *
                    "(dataset $(resolved) has ndims=$(nd))."))
                eltype(ds) <: Real || throw(ArgumentError(
                    "MANTA: HDF5 lazy loading requires a numeric dataset " *
                    "($(resolved) has eltype=$(eltype(ds)))."))
                (_lazy_hdf5_array(path, resolved, sz, _hdf5_chunk_tuple(ds, nd)), attrs, resolved)
            else
                (read(ds), attrs, resolved)
            end
        end
    catch e
        e isa ArgumentError && rethrow()
        throw(ArgumentError(
            "MANTA: failed to read HDF5 dataset $(address) in $(abspath(path)). " *
            "Original error: $(sprint(showerror, e))"))
    end

    return _build_dataset_from_h5(path, resolved_address, data, attrs_dict;
        column = column, v0 = v0, dv = dv, vunit = vunit)
end

function _build_dataset_from_h5(path, address, data, attrs::AbstractDict;
        column::Int, v0::Real, dv::Real, vunit::AbstractString)

    is_healpix_flag = _is_healpix_h5(attrs)

    unit_label = let
        s = get_attr_str(attrs, "units",
            get_attr_str(attrs, "UNITS",
                get_attr_str(attrs, "bunit",
                    get_attr_str(attrs, "BUNIT", ""))))
        isempty(s) ? "value" : s
    end

    sid_addr = replace(String(address), "/" => "_")
    sid_addr = lstrip(sid_addr, '_')
    base_name = String(splitext(basename(path))[1])
    source_id = isempty(sid_addr) ? base_name : base_name * "_" * sid_addr
    base_meta = Dict{Symbol,Any}(:hdf5_path => abspath(String(path)),
                                 :hdf5_address => String(address),
                                 :hdf5_attrs => attrs)
    if data isa AbstractLazyHDF5
        base_meta[:lazy] = true
        data.chunk === nothing || (base_meta[:hdf5_chunk] = data.chunk)
    end

    nd = ndims(data)

    # HEALPix routing
    if is_healpix_flag
        if nd == 1
            nside_guess = valid_healpix_npix(length(data))
            nside_guess > 0 || throw(ArgumentError(
                "MANTA: HDF5 dataset $(address) flagged HEALPix but length $(length(data)) is not 12·nside²."))
            m = Healpix.HealpixMap{Float64,Healpix.RingOrder,Vector{Float64}}(Float64.(vec(data)))
            return HealpixMapDataset(m;
                column = column, unit_label = unit_label,
                source_id = source_id, metadata = base_meta)
        elseif nd == 2
            s = size(data)
            ns1 = valid_healpix_npix(s[1])
            ns2 = valid_healpix_npix(s[2])
            v0_eff = get_attr_float(attrs, "v0", v0)
            dv_eff = get_attr_float(attrs, "dv", dv)
            vunit_eff = let u = get_attr_str(attrs, "vunit", String(vunit))
                isempty(u) ? String(vunit) : u
            end
            if ns1 > 0
                cube = as_float32(data)
                return HealpixCubeDataset(cube; nside = ns1,
                    v0 = v0_eff, dv = dv_eff, vunit = vunit_eff,
                    unit_label = unit_label,
                    source_id = source_id, metadata = base_meta)
            elseif ns2 > 0
                cube = as_float32(permutedims(data))
                return HealpixCubeDataset(cube; nside = ns2,
                    v0 = v0_eff, dv = dv_eff, vunit = vunit_eff,
                    unit_label = unit_label,
                    source_id = source_id, metadata = base_meta)
            else
                throw(ArgumentError(
                    "MANTA: HDF5 dataset $(address) flagged HEALPix but neither dim of $(s) is 12·nside²."))
            end
        else
            throw(ArgumentError(
                "MANTA: HDF5 dataset $(address) flagged HEALPix but ndims=$(nd) is unsupported."))
        end
    end

    # Generic routing by ndims
    if nd == 1
        return VectorDataset(vec(data);
            axis_label = first(_axis_labels_from_h5_attrs(attrs, 1)),
            unit_label = unit_label,
            source_id = source_id,
            metadata = base_meta)
    elseif nd == 2
        return ImageDataset(data;
            axis_labels = _axis_labels_from_h5_attrs(attrs, 2),
            wcs = _build_wcs_from_h5_attrs(attrs, 2),
            unit_label = unit_label,
            source_id = source_id,
            metadata = base_meta)
    elseif nd == 3
        return CubeDataset(data;
            axis_labels = _axis_labels_from_h5_attrs(attrs, 3),
            wcs = _build_wcs_from_h5_attrs(attrs, 3),
            unit_label = unit_label,
            source_id = source_id,
            metadata = base_meta)
    else
        throw(ArgumentError(
            "MANTA: HDF5 dataset $(address) has unsupported ndims=$(nd) (size=$(size(data)))"))
    end
end
