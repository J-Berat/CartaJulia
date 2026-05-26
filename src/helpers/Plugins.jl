# path: src/helpers/Plugins.jl
#
# Minimal plugin/extension system for MANTA.
#
# Goal: let users (or sister packages) hook into the viewer without
# patching MANTA itself. Three extension surfaces are exposed:
#
#   * `:loader`      — additional file-format loaders. Each plugin is a
#                       (matcher, loader) pair: `matcher(path)::Bool` returns
#                       true when the loader can handle the file; `loader(path; kwargs...)`
#                       returns an `AbstractMANTADataset`.
#   * `:dataset_view` — additional named "view modes" for an existing dataset
#                       type. Each plugin is a `(dataset_type, name, fn)` triple;
#                       `fn(ds; activate_gl, display_fig, kwargs...)` returns a Figure.
#   * `:postprocess`  — callbacks invoked after a view is built, with signature
#                       `(fig, ds, opts) -> nothing`. Use these to draw extra
#                       overlays, register custom shortcuts, etc.
#
# Plugins are stored in a per-kind `Vector` and dispatched in registration
# order. There is no priority system; users with conflicting plugins should
# unregister the ones they don't want.

const _PLUGIN_REGISTRY = Dict{Symbol, Vector{Any}}(
    :loader        => Any[],
    :dataset_view  => Any[],
    :postprocess   => Any[],
)

"""
    register_plugin!(kind, plugin)

Register a plugin under `kind` (`:loader`, `:dataset_view`, `:postprocess`).
Returns the plugin so it can be removed later with `unregister_plugin!`.
"""
function register_plugin!(kind::Symbol, plugin)
    haskey(_PLUGIN_REGISTRY, kind) || throw(ArgumentError(
        "Unknown plugin kind :$(kind). " *
        "Known: $(collect(keys(_PLUGIN_REGISTRY)))"))
    push!(_PLUGIN_REGISTRY[kind], plugin)
    return plugin
end

"""
    unregister_plugin!(kind, plugin)

Remove a previously-registered plugin (by identity). Returns the number
of entries actually removed (0 if the plugin wasn't registered).
"""
function unregister_plugin!(kind::Symbol, plugin)
    haskey(_PLUGIN_REGISTRY, kind) || return 0
    v = _PLUGIN_REGISTRY[kind]
    n = length(v)
    filter!(p -> p !== plugin, v)
    return n - length(v)
end

"Snapshot of currently-registered plugins for a kind (copy, safe to iterate)."
function list_plugins(kind::Symbol)
    haskey(_PLUGIN_REGISTRY, kind) || return Any[]
    return copy(_PLUGIN_REGISTRY[kind])
end

"Drop all plugins (used by tests)."
function clear_plugins!()
    for v in values(_PLUGIN_REGISTRY)
        empty!(v)
    end
    return nothing
end

# ---- dispatch helpers used by the loader / viewer -----------------------

"""
    plugin_load(path; kwargs...) -> dataset or nothing

Try each registered loader plugin in order; return the dataset from the
first matcher that accepts the path. Returns `nothing` when no plugin
matches — the caller should then fall back to the built-in loader.
"""
function plugin_load(path::AbstractString; kwargs...)
    for entry in _PLUGIN_REGISTRY[:loader]
        matcher, loader = entry
        try
            matcher(path) || continue
        catch
            continue
        end
        return loader(path; kwargs...)
    end
    return nothing
end

"""
    plugin_view(ds, name; kwargs...) -> Figure or nothing

Run the first `:dataset_view` plugin whose `(dataset_type, name)` matches
`(typeof(ds), name)`. Returns `nothing` when no matching plugin is registered.
"""
function plugin_view(ds, name::Symbol; kwargs...)
    for entry in _PLUGIN_REGISTRY[:dataset_view]
        ds_type, plugin_name, fn = entry
        if ds isa ds_type && plugin_name === name
            return fn(ds; kwargs...)
        end
    end
    return nothing
end

"""
    run_postprocess!(fig, ds; opts...)

Invoke every `:postprocess` plugin in order. Exceptions in user callbacks
are caught and logged via `@warn` so a faulty plugin can't crash the viewer.
"""
function run_postprocess!(fig, ds; opts...)
    for cb in _PLUGIN_REGISTRY[:postprocess]
        try
            cb(fig, ds, opts)
        catch e
            @warn "MANTA plugin postprocess raised an error" exception=(e, catch_backtrace())
        end
    end
    return fig
end

export register_plugin!, unregister_plugin!, list_plugins, clear_plugins!
export plugin_load, plugin_view, run_postprocess!
