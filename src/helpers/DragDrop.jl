# path: src/helpers/DragDrop.jl
#
# Drag-and-drop file loading.
#
# Hooks `events(fig).dropped_files` — a Makie-provided `Observable{Vector{String}}`
# that GLMakie fills with absolute paths whenever the user drops one or more
# files onto the window — and, when a *supported* file is dropped onto a MANTA
# window, reloads it **in place**: a fresh viewer is built for the dropped file
# with `manta(path; ...)` and rendered into the *same* OS window via the existing
# GLMakie screen, replacing the current view (per the user-chosen behaviour
# "recharger dans la fenêtre").
#
# Format detection reuses `parse_path_spec`, so the drop target accepts exactly
# the extensions MANTA already opens from the command line (`.fits` / `.fit` /
# `.fits.gz`, `.h5` / `.hdf5` / `.he5`, and the `file.h5:/group` address form).
# When several files are dropped at once the first supported one wins; anything
# unsupported is ignored with a warning and the current view is left untouched.
#
# Headless robustness: `enable_file_drop!` is safe with `activate_gl=false` /
# `display_fig=false`. The listener is still registered, but with no GLMakie
# screen attached the reload is built without displaying a window — so the test
# suite (`activate_gl=false, display_fig=false`) and Docker exercise the same
# code path as the interactive viewer.

# Formats `parse_path_spec` recognises and the loaders can ingest.
const _DROP_SUPPORTED_KINDS = (:fits, :hdf5)

# Figure root scenes whose drop listener is already armed. A `WeakKeyDict` so
# that this registry never keeps a viewer alive — `forget!`/GC stay in charge of
# lifetime, exactly like `_KEEP_ALIVE` (see the block comment at the top of
# MANTA.jl). We key by `fig.scene` rather than `fig` because Makie.Figure is
# immutable on newer Makie/Julia combinations, and `WeakKeyDict` keys must be
# finalizable mutable objects.
const _DROP_ARMED = WeakKeyDict{Any,Nothing}()

"""
    supported_drop_path(paths) -> Union{String,Nothing}

Return the first entry of `paths` that names an existing file in a format MANTA
can open (FITS or HDF5, as classified by [`parse_path_spec`]), or `nothing` when
none qualify. Used to pick the file to load from a multi-file drop.
"""
function supported_drop_path(paths)
    for p in paths
        s = String(p)
        isfile(s) || continue
        kind = :unknown
        try
            kind = first(parse_path_spec(s))
        catch
            kind = :unknown
        end
        kind in _DROP_SUPPORTED_KINDS && return s
    end
    return nothing
end

"""
    enable_file_drop!(fig; activate_gl=true, display_fig=true, reopen=nothing, kwargs...)

Arm drag-and-drop file loading on `fig`. When the user drops file(s) onto the
window, the first supported file (FITS/HDF5) is opened and rendered into the
**same** window, replacing the current view; unsupported drops are ignored.

By default the dropped file is opened with `manta(path; activate_gl=activate_gl,
display_fig=false, kwargs...)` (display is handled here via window reuse). Pass a
custom `reopen` callback — `path -> Figure` — to override how the replacement
figure is built (used by the headless tests).

Idempotent: calling it twice on the same figure registers a single listener.
Returns `fig`.
"""
function enable_file_drop!(fig; activate_gl::Bool = true,
                           display_fig::Bool = true, reopen = nothing, kwargs...)
    # Only arm once per figure.
    armed_key = fig.scene
    haskey(_DROP_ARMED, armed_key) && return fig

    drop_obs = try
        events(fig).dropped_files
    catch
        nothing
    end
    # Backend without a `dropped_files` event: nothing to hook, stay silent.
    drop_obs === nothing && return fig

    reopen_fn = reopen === nothing ?
        (path -> manta(path; activate_gl = activate_gl,
                       display_fig = false, kwargs...)) :
        reopen

    on(drop_obs) do paths
        _handle_drop(fig, paths, reopen_fn; display_fig = display_fig)
        return nothing
    end
    _DROP_ARMED[armed_key] = nothing
    return fig
end

# Core drop handler: validate, build a replacement figure, and swap it into the
# current window. Pulled out of the closure so tests can call it directly.
function _handle_drop(old_fig, paths, reopen_fn; display_fig::Bool = true)
    path = supported_drop_path(paths)
    if path === nothing
        isempty(paths) ||
            @warn "MANTA: dropped file(s) are not a supported format (FITS/HDF5); ignoring." paths
        return nothing
    end

    new_fig = try
        reopen_fn(path)
    catch e
        @warn "MANTA: failed to open dropped file; keeping current view." path exception = (e, catch_backtrace())
        return nothing
    end
    new_fig isa Figure || return nothing

    # Reuse the OS window backing the previous figure when one is attached.
    screen = nothing
    try
        screen = Makie.getscreen(old_fig.scene)
    catch
        screen = nothing
    end

    if screen !== nothing
        reused = false
        try
            display(screen, new_fig)        # render the new figure in the same window
            reused = true
        catch
            try
                display(screen, new_fig.scene)  # older GLMakie: scene-level reuse
                reused = true
            catch e
                @warn "MANTA: could not reuse the window for the dropped file; opening a new window." exception = (e, catch_backtrace())
            end
        end
        if reused
            forget!(old_fig)                # release the replaced figure from the GC root
        elseif display_fig
            display(new_fig)
        end
    elseif display_fig
        display(new_fig)
    end

    return new_fig
end
