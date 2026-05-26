# path: src/helpers/Backend.jl
#
# Backend activation helpers.
#
# MANTA targets two Makie backends:
#   * `GLMakie` for the interactive desktop viewer (needs an OpenGL context).
#   * `CairoMakie` for headless rendering, PNG/PDF/GIF exports, and CI tests.
#
# The viewers currently sprinkle `activate_gl ? GLMakie.activate!() : CairoMakie.activate!()`
# at construction time. This file centralises:
#
#   1. `pick_backend!(activate_gl)` — sets the right backend AND degrades
#      gracefully when GLMakie can't initialise (no DISPLAY, no GPU, …).
#   2. `with_export_backend(f)` — switch to CairoMakie for the duration of
#      `f()` and restore the original backend on exit. Used by every export
#      path (PNG, PDF, GIF) so we never accidentally screenshot a GL window.
#
# Both helpers are no-throw: failure to activate a backend is reported as a
# `@warn` so the caller can fall back to the headless path instead of
# crashing the viewer.

"""
    is_headless_env() -> Bool

Best-effort detection of an OpenGL-less environment. Returns `true` when:
  * `MANTA_HEADLESS` is set to a truthy value, or
  * we are running on Unix-like systems with no `DISPLAY` and no `WAYLAND_DISPLAY`.

This is intentionally conservative — when in doubt, we say "not headless"
and let GLMakie try.
"""
function is_headless_env()
    v = lowercase(strip(String(get(ENV, "MANTA_HEADLESS", ""))))
    v in ("1", "true", "yes", "on") && return true
    if Sys.isunix() && !Sys.isapple()
        isempty(get(ENV, "DISPLAY", "")) &&
            isempty(get(ENV, "WAYLAND_DISPLAY", "")) && return true
    end
    return false
end

"""
    pick_backend!(activate_gl::Bool) -> Symbol

Activate the requested backend, downgrading to CairoMakie when GLMakie
can't initialise (no display, no GPU, headless CI, …). Returns the
backend symbol that was actually activated (`:GLMakie` or `:CairoMakie`).

The caller can use the return value to decide whether interactive widgets
are usable — for instance the "Save GIF" button stays enabled either way,
but the `display(fig)` call is skipped on `:CairoMakie`.
"""
function pick_backend!(activate_gl::Bool)
    if !activate_gl || is_headless_env()
        try
            CairoMakie.activate!()
        catch e
            @warn "MANTA: CairoMakie.activate! failed" exception=(e, catch_backtrace())
        end
        return :CairoMakie
    end
    try
        GLMakie.activate!()
        return :GLMakie
    catch e
        @warn "MANTA: GLMakie.activate! failed, falling back to CairoMakie" exception=(e, catch_backtrace())
        try
            CairoMakie.activate!()
        catch e2
            @warn "MANTA: CairoMakie.activate! also failed" exception=(e2, catch_backtrace())
        end
        return :CairoMakie
    end
end

"""
    with_export_backend(f) -> result

Activate CairoMakie for the duration of `f()`, then restore whichever
backend was active before. Use this around every export path:

```julia
with_export_backend() do
    save(path, fig)
end
```

The active backend is queried via `Makie.current_backend()`. If that
returns `nothing` (Makie hasn't been activated yet in this session),
the fallback is GLMakie — matches MANTA's default.
"""
function with_export_backend(f::Function)
    prev_sym = _current_backend_symbol()
    try
        try
            CairoMakie.activate!()
        catch e
            @warn "MANTA: CairoMakie.activate! failed during export" exception=(e, catch_backtrace())
        end
        return f()
    finally
        try
            _activate_by_symbol(prev_sym)
        catch e
            @warn "MANTA: failed to restore previous backend" backend=prev_sym exception=(e, catch_backtrace())
        end
    end
end

function _current_backend_symbol()
    # Makie's API for "which backend is active" has shifted across versions;
    # we go through a defensive try/catch ladder so this helper compiles on
    # everything from Makie 0.20 to 0.24.
    try
        backend = Makie.current_backend()
        backend === nothing && return :GLMakie
        return Symbol(nameof(typeof(backend)))
    catch
    end
    return :GLMakie
end

function _activate_by_symbol(sym::Symbol)
    if sym === :CairoMakie
        CairoMakie.activate!()
    else
        GLMakie.activate!()
    end
    return nothing
end

export is_headless_env, pick_backend!, with_export_backend
