# path: scripts/setup.jl
import Pkg

# Activate repo root
project_dir = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(project_dir)

# Instantiate existing resolution (if any)
Pkg.instantiate()

# Add/ensure runtime deps.
# We intentionally resolve by package name here to avoid UUID mismatches
# between historical setup scripts and the active Project.toml.
runtime_dep_names = [
    "GLMakie",
    "CairoMakie",
    "Makie",
    "Observables",
    "ImageFiltering",
    "LaTeXStrings",
    "FITSIO",
    "GLFW",
    "PrecompileTools",
]

project_deps = Set(keys(Pkg.project().dependencies))
missing_dep_names = [name for name in runtime_dep_names if !(name in project_deps)]
deps = [Pkg.PackageSpec(name = name) for name in missing_dep_names]

try
    if !isempty(deps)
        Pkg.add(deps)
    end
catch e
    @error "Failed to add some dependencies" error=e
    rethrow()
end

# Precompile to catch issues early
Pkg.precompile()

# Print a concise status
Pkg.status(mode = Pkg.PKGMODE_PROJECT)
println("\n✅ Setup completed for project at: $project_dir")
