# path: src/helpers/Progress.jl
#
# Lightweight cooperative progress + cancellation primitives.
#
# The viewer's long-running tasks (GIF export, moment-map computation,
# big FITS reads) used to be fire-and-forget. With this module callers can:
#
#   1. Build a `CancelToken` that any background task can poll via
#      `is_cancelled(tok)` and the UI can flip via `cancel!(tok)`.
#   2. Build a `ProgressTracker` that the task can `tick!` and the UI can
#      observe via the `progress` Observable (0..1).
#
# Both objects are designed to be passed around as kwargs (default
# values mean "no-op"), so existing callers don't need to change.
#
# The tracker also exposes `status::Observable{String}` for a free-form
# message ("loading slice 12/40", "rendering GIF…", etc.).
#
# Nothing here depends on Makie itself except `Observable`, which is
# already imported via `using Observables` in MANTA.jl. This file is
# therefore safe to include before any view is compiled.

# ---- Cancellation -------------------------------------------------------

"""
    CancelToken()

Thread-safe cooperative cancellation flag.

```julia
tok = CancelToken()
task = @async begin
    for k in 1:N
        is_cancelled(tok) && break
        # … work …
    end
end
cancel!(tok)
```
"""
mutable struct CancelToken
    @atomic flag::Bool
    CancelToken() = new(false)
end

"Flip the token to cancelled. Idempotent."
function cancel!(t::CancelToken)
    @atomic t.flag = true
    return t
end

"Has the token been cancelled?"
is_cancelled(t::CancelToken) = (@atomic t.flag)

"Has the token been cancelled? `nothing` is treated as `false`."
is_cancelled(::Nothing) = false

"Reset the token to a fresh state (use sparingly — usually create a new one)."
function reset!(t::CancelToken)
    @atomic t.flag = false
    return t
end

# ---- Progress tracking --------------------------------------------------

"""
    ProgressTracker(; total = 0, label = "")

Cooperative progress object. `progress` is an `Observable{Float64}` in `[0, 1]`,
`status` an `Observable{String}` for free-form messages. `total` is the number
of expected `tick!` calls; pass `0` for indeterminate work and call
`set_progress!` directly.
"""
mutable struct ProgressTracker
    progress::Observable{Float64}
    status::Observable{String}
    total::Int
    done::Int
    label::String
end

function ProgressTracker(; total::Integer = 0, label::AbstractString = "")
    ProgressTracker(Observable(0.0), Observable(String(label)),
                    Int(total), 0, String(label))
end

"Advance the counter by 1 and update `progress`."
function tick!(p::ProgressTracker; status::Union{Nothing,AbstractString} = nothing)
    p.done += 1
    if p.total > 0
        p.progress[] = clamp(p.done / p.total, 0.0, 1.0)
    end
    status === nothing || (p.status[] = String(status))
    return p
end

"Set progress directly to a normalized value in [0, 1]."
function set_progress!(p::ProgressTracker, v::Real;
                       status::Union{Nothing,AbstractString} = nothing)
    p.progress[] = clamp(Float64(v), 0.0, 1.0)
    status === nothing || (p.status[] = String(status))
    return p
end

"Mark the tracker as finished (progress = 1)."
function finish!(p::ProgressTracker;
                 status::Union{Nothing,AbstractString} = nothing)
    p.progress[] = 1.0
    p.done = max(p.done, p.total)
    status === nothing || (p.status[] = String(status))
    return p
end

# `nothing`-friendly overloads so call sites can write
# `tick!(progress)` whether or not the caller passed a tracker.
tick!(::Nothing; status = nothing) = nothing
set_progress!(::Nothing, value; status = nothing) = nothing
finish!(::Nothing; status = nothing) = nothing

# ---- Convenience: combined try-block ------------------------------------

"""
    with_progress(f, label; total = 0)

Run `f(prog, tok)` with a fresh `ProgressTracker`/`CancelToken` pair.
On normal exit, finishes the tracker; on exception (other than the
cancellation sentinel), re-throws. Returns whatever `f` returns,
or `nothing` if the token was cancelled mid-run.

```julia
result = with_progress("Rendering GIF", total=N) do prog, tok
    for k in 1:N
        is_cancelled(tok) && return nothing
        # … work …
        tick!(prog; status = "frame \$k/\$N")
    end
    return out
end
```
"""
function with_progress(f::Function, label::AbstractString;
                       total::Integer = 0)
    prog = ProgressTracker(; total = total, label = label)
    tok  = CancelToken()
    try
        result = f(prog, tok)
        is_cancelled(tok) || finish!(prog)
        return result
    catch e
        # Cancellation isn't an error — caller decides.
        rethrow()
    end
end

export CancelToken, cancel!, is_cancelled
export ProgressTracker, tick!, set_progress!, finish!
export with_progress
