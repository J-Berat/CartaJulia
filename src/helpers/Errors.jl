# path: src/helpers/Errors.jl
#
# Actionable, structured error surface for MANTA. Loaders and public
# entry points should raise these instead of bare `ArgumentError`s when
# the user can fix the situation by changing an input.
#
# Each error carries:
#   - `msg`       : the short, user-facing problem statement
#   - `hint`      : ONE concrete next step (file path, kwarg name, value…)
#   - `context`   : optional Dict of structured pairs (e.g. `:path => "…"`)
#
# `MANTAError` is the abstract root so callers can `try … catch ::MANTAError`
# uniformly. Subtypes mirror the most common failure modes.

abstract type MANTAError <: Exception end

# ---- concrete subtypes ---------------------------------------------------

"""
Raised when a file path is required but missing/unreadable on disk.
"""
struct FileNotFoundError <: MANTAError
    path::String
    hint::String
end

"""
Raised when the on-disk format does not match what the loader expects.
"""
struct UnsupportedFormatError <: MANTAError
    path::String
    detected::String
    hint::String
end

"""
Raised when an HDU index is out of range or refers to an unreadable HDU.
"""
struct HDUSelectionError <: MANTAError
    path::String
    requested::Int
    available::Int
    hint::String
end

"""
Raised when a kwarg value is unusable (wrong type, out-of-range, …).
"""
struct InvalidArgumentError <: MANTAError
    name::Symbol
    value::Any
    hint::String
end

"""
Raised when the dataset shape contradicts the requested view (e.g. asking
for a HEALPix view on a non-HEALPix file).
"""
struct DatasetShapeError <: MANTAError
    msg::String
    hint::String
end

# ---- pretty showerror ----------------------------------------------------

function Base.showerror(io::IO, e::FileNotFoundError)
    print(io, "MANTA: fichier introuvable: ", e.path)
    isempty(e.hint) || print(io, "\n  → ", e.hint)
end

function Base.showerror(io::IO, e::UnsupportedFormatError)
    print(io, "MANTA: format non supporté pour ", e.path,
              " (détecté: ", e.detected, ")")
    isempty(e.hint) || print(io, "\n  → ", e.hint)
end

function Base.showerror(io::IO, e::HDUSelectionError)
    print(io, "MANTA: HDU #", e.requested,
              " indisponible dans ", e.path,
              " (le fichier en contient ", e.available, ")")
    isempty(e.hint) || print(io, "\n  → ", e.hint)
end

function Base.showerror(io::IO, e::InvalidArgumentError)
    print(io, "MANTA: argument invalide ", e.name, " = ", repr(e.value))
    isempty(e.hint) || print(io, "\n  → ", e.hint)
end

function Base.showerror(io::IO, e::DatasetShapeError)
    print(io, "MANTA: ", e.msg)
    isempty(e.hint) || print(io, "\n  → ", e.hint)
end

# ---- convenience constructors -------------------------------------------

"""
    require_file(path; hint="")

Throw [`FileNotFoundError`](@ref) when `path` is not a regular file on disk.
"""
function require_file(path::AbstractString; hint::AbstractString = "")
    if !isfile(path)
        h = isempty(hint) ?
            "Vérifie le chemin (cwd = $(pwd())) ou utilise un chemin absolu." :
            String(hint)
        throw(FileNotFoundError(String(path), h))
    end
    return String(path)
end

"""
    rethrow_actionable(e, path; format_hint="")

Wrap a low-level FITSIO/HDF5 exception in [`UnsupportedFormatError`](@ref).
The original error text is preserved in `hint` so the user can still see it.
"""
function rethrow_actionable(e::Exception, path::AbstractString;
                            format_hint::AbstractString = "")
    detected = string(typeof(e).name.name)
    user_hint = isempty(format_hint) ?
        "Erreur originale: $(sprint(showerror, e))" :
        "$(format_hint)\n     (erreur originale: $(sprint(showerror, e)))"
    throw(UnsupportedFormatError(String(path), detected, user_hint))
end

"""
    invalid_kwarg(name, value; hint)

Throw [`InvalidArgumentError`](@ref). Use this in public entry points to
report bad kwarg values with a concrete remediation.
"""
function invalid_kwarg(name::Symbol, value; hint::AbstractString)
    throw(InvalidArgumentError(name, value, String(hint)))
end

"""
    invalid_hdu(path, requested, available; hint="")

Throw [`HDUSelectionError`](@ref).
"""
function invalid_hdu(path::AbstractString, requested::Integer, available::Integer;
                     hint::AbstractString = "")
    h = isempty(hint) ?
        (available <= 0 ? "Le fichier ne contient aucune HDU lisible." :
         "Choisis une HDU entre 1 et $(available) via le kwarg `hdu`.") :
        String(hint)
    throw(HDUSelectionError(String(path), Int(requested), Int(available), h))
end

export MANTAError, FileNotFoundError, UnsupportedFormatError,
       HDUSelectionError, InvalidArgumentError, DatasetShapeError
export require_file, rethrow_actionable, invalid_kwarg, invalid_hdu
