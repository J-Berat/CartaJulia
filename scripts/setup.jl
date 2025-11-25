# path: scripts/setup.jl
import Pkg

# Activate repo root
project_dir = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(project_dir)

# Instantiate existing resolution (if any)
Pkg.instantiate()

# Add/ensure runtime deps (UUIDs for determinism)
deps = [
    Pkg.PackageSpec(name = "GLMakie",        uuid = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"),
    Pkg.PackageSpec(name = "CairoMakie",     uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"),
    Pkg.PackageSpec(name = "Makie",          uuid = "ee78f7c6-11fb-53cd-9871-cc25717d1b9b"),
    Pkg.PackageSpec(name = "Observables",    uuid = "510215fc-4207-5dde-b226-833fc4488ee2"),
    Pkg.PackageSpec(name = "ImageFiltering", uuid = "6a3955dd-da59-5b1f-98d4-96b0c1f18a3e"),
    Pkg.PackageSpec(name = "LaTeXStrings",   uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"),
    Pkg.PackageSpec(name = "FITSIO",         uuid = "525bcba6-941b-5504-bd06-fd0dc1a4d2eb"),
    Pkg.PackageSpec(name = "GLFW",           uuid = "f7f18e0c-5ee9-5ccd-a5bf-e8befd85ed98"),
]

try
    Pkg.add(deps)
catch e
    @error "Failed to add some dependencies" error=e
    rethrow()
end

# Precompile to catch issues early
Pkg.precompile()

# Print a concise status
Pkg.status(mode = Pkg.PKGMODE_PROJECT)
println("\n✅ Setup completed for project at: $project_dir")
