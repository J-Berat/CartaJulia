# path: src/helpers/UndoRedo.jl
#
# Generic undo/redo stack for Observable-driven viewers.
#
# Design choice: snapshot-based rather than command-based. The viewer
# state we want to roll back (clims, scale mode, axis index, slice index,
# colormap, contour visibility…) is a flat bag of small primitive values.
# A timed snapshot is cheaper to reason about than command inversion,
# and trivially correct because we replay the whole tuple.
#
# The stack is bounded (`capacity = UNDO_STACK_CAPACITY` by default) so memory
# stays predictable even after thousands of UI interactions.
#
# Integration pattern:
#
#   state = UndoRedoStack{NamedTuple}(...)  # capacity defaults to UNDO_STACK_CAPACITY
#   register_state!(state, (clims = ..., scale = :lin, axis = 3, slice = 1))
#   # … user changes things …
#   register_state!(state, (clims = ..., scale = :log10, axis = 3, slice = 1))
#   # On Ctrl-Z:
#   snap = undo!(state)
#   snap === nothing || apply_state!(viewer, snap)
#
# `register_state!` deduplicates consecutive identical snapshots, so a
# slider that fires 100 events per drag still produces a sensible history.

"""
    UndoRedoStack{T}(; capacity = UNDO_STACK_CAPACITY)

Bounded snapshot history with an Observable cursor. Useful when wiring
Ctrl-Z / Ctrl-Shift-Z into a Makie figure.
"""
mutable struct UndoRedoStack{T}
    snapshots::Vector{T}
    cursor::Int                    # index of current snapshot in `snapshots`
    capacity::Int
    can_undo::Observable{Bool}
    can_redo::Observable{Bool}
    suppress::Bool                 # internal: ignore register_state! when replaying
end

function UndoRedoStack{T}(; capacity::Integer = UNDO_STACK_CAPACITY) where {T}
    capacity > 0 || throw(ArgumentError("capacity must be positive"))
    UndoRedoStack{T}(T[], 0, Int(capacity),
                     Observable(false), Observable(false), false)
end

UndoRedoStack(initial::T; capacity::Integer = UNDO_STACK_CAPACITY) where {T} =
    (s = UndoRedoStack{T}(; capacity); register_state!(s, initial); s)

"Number of recorded snapshots."
Base.length(s::UndoRedoStack) = length(s.snapshots)

"Is the stack empty?"
Base.isempty(s::UndoRedoStack) = isempty(s.snapshots)

"Currently recorded snapshot (or `nothing` if empty)."
current(s::UndoRedoStack) = isempty(s) ? nothing : s.snapshots[s.cursor]

"""
    register_state!(stack, snapshot)

Push a new snapshot. No-op when the value is equal to the current one
(so dragging a slider 100 times produces ONE entry, not 100). When the
cursor is in the middle of the history (after an `undo!`), any "future"
snapshots are discarded, matching standard editor semantics.

Suppress flag: when `stack.suppress` is `true` (set during replay),
calls are silently ignored so that the act of replaying a snapshot
doesn't itself create a new entry.
"""
function register_state!(s::UndoRedoStack{T}, snapshot::T) where {T}
    s.suppress && return s
    if !isempty(s.snapshots) && s.snapshots[s.cursor] == snapshot
        return s
    end
    # Drop any "redo" tail.
    if s.cursor < length(s.snapshots)
        resize!(s.snapshots, s.cursor)
    end
    push!(s.snapshots, snapshot)
    s.cursor = length(s.snapshots)
    # Enforce capacity from the bottom.
    while length(s.snapshots) > s.capacity
        popfirst!(s.snapshots)
        s.cursor -= 1
    end
    s.can_undo[] = s.cursor > 1
    s.can_redo[] = false
    return s
end

"""
    undo!(stack) -> snapshot or `nothing`

Move the cursor back one step and return the resulting snapshot, or
`nothing` when already at the bottom. The caller is responsible for
applying the snapshot to the live UI Observables — typically inside a
`with_suppression` block to avoid recording the replay.
"""
function undo!(s::UndoRedoStack)
    s.cursor > 1 || return nothing
    s.cursor -= 1
    s.can_undo[] = s.cursor > 1
    s.can_redo[] = s.cursor < length(s.snapshots)
    return s.snapshots[s.cursor]
end

"""
    redo!(stack) -> snapshot or `nothing`

Move the cursor forward one step, or return `nothing` when no redo is
available.
"""
function redo!(s::UndoRedoStack)
    s.cursor < length(s.snapshots) || return nothing
    s.cursor += 1
    s.can_undo[] = s.cursor > 1
    s.can_redo[] = s.cursor < length(s.snapshots)
    return s.snapshots[s.cursor]
end

"""
    with_suppression(f, stack)

Run `f()` with the stack's recording disabled. Use this around the code
that applies an undone/redone snapshot to your Observables — otherwise
each Observable write would push a fresh entry.
"""
function with_suppression(f::Function, s::UndoRedoStack)
    prev = s.suppress
    s.suppress = true
    try
        return f()
    finally
        s.suppress = prev
    end
end

"Clear the stack and the can_undo/can_redo flags."
function clear!(s::UndoRedoStack)
    empty!(s.snapshots)
    s.cursor = 0
    s.can_undo[] = false
    s.can_redo[] = false
    return s
end

# `clear!` and `current` are deliberately NOT exported — they are too generic
# and would clash with Base. Access via the qualified `MANTA.clear!(stack)`
# / `MANTA.current(stack)` when needed.
export UndoRedoStack, register_state!, undo!, redo!, with_suppression
