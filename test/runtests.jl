# path: test/runtests.jl
# Orchestrateur : charge les dépendances communes, puis inclut chaque
# fichier de tests par domaine. Tous les fichiers partagent ce namespace —
# ils n'ont pas besoin de répéter les `using`.
#
#   helpers.jl   — scaling, mapping, products, latex, io, ui, validation,
#                  WCS, settings, power spectrum, slice utils, path parsing,
#                  FITS export headers, shortcuts, errors, progress,
#                  downsampling, undo/redo, plugins, backend selection
#   healpix.jl   — mollweide, regions projetées, viewer headless regression
#   loaders.jl   — datasets (in-memory + paths), abstract types, hdu kwarg
#   cube_view.jl — smoke + erreurs, dispatch manta, vector viewer, view_cube
#   masks.jl     — pure helpers, TOML, integration, combinators, smoke
#   lazy_fits.jl — 2D + 3D lazy FITS
#   drag_drop.jl — drag-and-drop file loading (format detection + in-window reload)
using Test

# load the local module
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using MANTA

# deps used by the helpers
using Observables
using Makie
using LaTeXStrings
using ColorTypes
using FITSIO
using HDF5
using Statistics: mean
using Healpix

include("helpers.jl")
include("healpix.jl")
include("loaders.jl")
include("cube_view.jl")
include("masks.jl")
include("lazy_fits.jl")
include("lazy_hdf5.jl")
include("drag_drop.jl")
