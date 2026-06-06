# path: src/helpers/Shortcuts.jl
#
# Reusable keyboard shortcut machinery for MANTA views.
#
# Design notes:
#   - `ShortcutBinding` describes a single binding: key + optional modifier,
#     a zero-arg callable that performs the action, and a short human-readable
#     description used to format the help string.
#   - `register_shortcuts!` installs a single `keyboardbutton` handler on the
#     figure. It guards against firing while the user is typing in a
#     `Makie.Textbox` (any view-provided `is_blocked` callback is also
#     honored — useful for active region drags, animation playback, etc.).
#   - `format_shortcut_help` / `shortcut_help_message` render the binding list
#     into a one-line status hint, so the same source of truth drives both the
#     dispatch and the UI.
#
# This file is included from `src/helpers/Helpers.jl` and the exported
# symbols land in the parent MANTA module.

const _SHORTCUT_MODIFIERS = (:none, :shift, :ctrl, :alt)

"""
    ShortcutBinding(key, action; description="", modifier=:none)

Single keyboard binding.

- `key`: a `Makie.Keyboard.Button` (e.g. `Keyboard.r`, `Keyboard.page_up`).
- `action`: a zero-argument callable invoked on key press.
- `description`: short human-readable label used by `format_shortcut_help`.
- `modifier`: one of `:none`, `:shift`, `:ctrl`, `:alt`. `:none` means the
  binding fires when neither Ctrl nor Alt is held (Shift state is ignored
  so that letter keys remain consistent across keyboard layouts).
"""
struct ShortcutBinding
    key::Makie.Keyboard.Button
    action::Function
    description::String
    modifier::Symbol
    function ShortcutBinding(key::Makie.Keyboard.Button, action::Function;
                              description::AbstractString = "",
                              modifier::Symbol = :none)
        modifier in _SHORTCUT_MODIFIERS || error(
            "modifier must be one of $(_SHORTCUT_MODIFIERS), got :$(modifier)")
        return new(key, action, String(description), modifier)
    end
end

"""
    _textbox_focused(textboxes) -> Bool

Return true if any of the supplied `Makie.Textbox` widgets is currently
focused. Designed to tolerate widgets that do not expose `focused`
(returns false for those rather than crashing the keyboard handler).
"""
function _textbox_focused(textboxes)
    for tb in textboxes
        try
            if hasproperty(tb, :focused) && tb.focused[]
                return true
            end
        catch
            # Be defensive: a missing or non-Observable `focused` should not
            # break the global keyboard handler.
        end
    end
    return false
end

"""
    register_shortcuts!(fig, bindings;
                        textboxes=(), is_blocked=()->false, consume=true)

Install a single `keyboardbutton` handler on `fig` that dispatches to the
supplied `ShortcutBinding` list. Returns `fig`.

Keyword arguments:
- `textboxes`: an iterable of `Makie.Textbox` widgets. Dispatch is skipped
  while any of them is focused so that typing in a control field does not
  trigger shortcuts.
- `is_blocked`: an extra zero-arg predicate. Dispatch is skipped while it
  returns true (used to gate on drag flags, animation playback, etc.).
- `consume`: whether to return `Consume(true)` after firing a binding so
  that other listeners (zoom, axis interactions, …) do not also react to
  the same key press. Defaults to `true`.
"""
function register_shortcuts!(fig::Makie.Figure,
                             bindings::AbstractVector{ShortcutBinding};
                             textboxes = (),
                             is_blocked::Function = () -> false,
                             consume::Bool = true)
    tbs = collect(textboxes)
    bindings_copy = collect(bindings)
    on(events(fig).keyboardbutton) do ev
        ev.action == Keyboard.press || return Consume(false)
        try
            is_blocked() && return Consume(false)
        catch
            # A misbehaving `is_blocked` should not crash the handler.
        end
        _textbox_focused(tbs) && return Consume(false)
        # `keyboardstate` is the standard Makie name for the Set of currently
        # pressed keys. Be defensive in case a Makie version exposes it
        # differently — treat a missing state as "no modifier pressed" so
        # only unmodified bindings still fire.
        ctrl_down = false
        alt_down  = false
        shift_down = false
        try
            kb = events(fig).keyboardstate
            ctrl_down  = (Keyboard.left_control in kb) || (Keyboard.right_control in kb)
            alt_down   = (Keyboard.left_alt in kb)     || (Keyboard.right_alt in kb)
            shift_down = (Keyboard.left_shift in kb)   || (Keyboard.right_shift in kb)
        catch
        end
        for b in bindings_copy
            ev.key == b.key || continue
            mod_ok = if b.modifier === :none
                !ctrl_down && !alt_down
            elseif b.modifier === :shift
                shift_down && !ctrl_down && !alt_down
            elseif b.modifier === :ctrl
                ctrl_down && !alt_down
            else # :alt
                alt_down && !ctrl_down
            end
            mod_ok || continue
            try
                b.action()
            catch e
                @warn "Shortcut handler error" key = b.key exception = (e, catch_backtrace())
            end
            return consume ? Consume(true) : Consume(false)
        end
        return Consume(false)
    end
    return fig
end

# ---------------------------------------------------------------------------
# Centered zoom — reusable `+` / `-` machinery shared by every interactive view
# ---------------------------------------------------------------------------

"""
Default multiplicative factors applied to the current view span on each
`+` / `-` press. `< 1` zooms in (tighter span), `> 1` zooms out.
"""
const ZOOM_IN_FACTOR  = 0.8
const ZOOM_OUT_FACTOR = 1.25

# Scale functions whose displayed interval is `(0, ∞)`, i.e. the span must be
# scaled in transformed (log) space and any zoom must keep both bounds > 0.
_is_log_scale(f) = f === log10 || f === log2 || f === log

"""
    zoom_limits((xmin, xmax, ymin, ymax), factor; xlog=false, ylog=false)
        -> (xmin, xmax, ymin, ymax)

Pure helper: scale each axis span by `factor` about its own midpoint and
return the new `(xmin, xmax, ymin, ymax)`. `factor < 1` zooms in, `> 1`
zooms out. The scaling is uniform across both axes, so an aspect-locked
(`DataAspect`) view stays consistent.

When `xlog` / `ylog` is set, the corresponding span is scaled in *display*
(`log10`) space and mapped back with `exp10`. A log-scaled axis only ever
displays a strictly-positive interval, so this keeps both new bounds `> 0`
for any `factor` — which is exactly what prevents `log10` from producing
`NaN` limits (and the downstream Makie `all(low .<= high)` assertion in
`update_axis_camera`) when zooming out on a log axis.

The inputs are returned unchanged when any limit is non-finite, when
`factor` is not a finite positive number, when a log axis is asked to scale a
non-positive interval, or when a span collapses to zero — so the caller can
never produce `NaN`/inverted limits.
"""
function zoom_limits(lims::NTuple{4,<:Real}, factor::Real;
                     xlog::Bool = false, ylog::Bool = false)
    xmin, xmax, ymin, ymax = lims
    all(isfinite, (xmin, xmax, ymin, ymax)) || return lims
    (isfinite(factor) && factor > 0) || return lims
    (xlog && !(xmin > 0 && xmax > 0)) && return lims
    (ylog && !(ymin > 0 && ymax > 0)) && return lims
    fx = xlog ? log10 : identity
    fy = ylog ? log10 : identity
    gx = xlog ? exp10 : identity
    gy = ylog ? exp10 : identity
    tx0, tx1 = fx(xmin), fx(xmax)
    ty0, ty1 = fy(ymin), fy(ymax)
    xc = (tx0 + tx1) / 2
    yc = (ty0 + ty1) / 2
    xh = (tx1 - tx0) / 2 * factor
    yh = (ty1 - ty0) / 2 * factor
    (xh == 0 || yh == 0) && return lims
    new = (gx(xc - xh), gx(xc + xh), gy(yc - yh), gy(yc + yh))
    all(isfinite, new) || return lims
    return new
end

"""
    zoom_axis!(ax, factor) -> ax

Zoom Makie `Axis` `ax` by `factor` about the centre of its *currently
displayed* view (`ax.finallimits`). `factor < 1` zooms in, `> 1` zooms out.
Reading `finallimits` (rather than `targetlimits`) means the zoom always
starts from what the user actually sees, including any `DataAspect`
adjustment. The axis scales (`ax.xscale` / `ax.yscale`) are honoured, so a
`log10` axis is scaled in log space and stays strictly positive. No-ops on
degenerate / non-finite limits.
"""
function zoom_axis!(ax, factor::Real)
    fl   = ax.finallimits[]
    o, w = fl.origin, fl.widths
    xmin = Float64(o[1]); ymin = Float64(o[2])
    cur  = (xmin, xmin + Float64(w[1]), ymin, ymin + Float64(w[2]))
    xlog = _is_log_scale(ax.xscale[])
    ylog = _is_log_scale(ax.yscale[])
    nlim = zoom_limits(cur, factor; xlog = xlog, ylog = ylog)
    nlim === cur && return ax
    limits!(ax, nlim[1], nlim[2], nlim[3], nlim[4])
    return ax
end

"""
    LogSafeScrollZoom(speed)

Drop-in replacement for Makie's built-in `:scrollzoom` interaction that is
safe on `log10`-scaled axes. The scroll delta is mapped to a multiplicative
zoom `factor` and applied through [`zoom_axis!`](@ref), which scales the span
in log space about the view centre. Because the new bounds are produced with
`exp10`, they stay strictly positive for any scroll amount, so zooming out can
never drive a log limit to `≤ 0` (the root cause of the Makie
`AssertionError: all(low .<= high)` raised from `update_axis_camera` inside the
render loop).

Unlike Makie's default scroll zoom this is anchored at the view centre rather
than the cursor — matching the `+` / `-` keyboard zoom — which is a deliberate
trade-off for crash safety on log axes.
"""
struct LogSafeScrollZoom
    speed::Float64
end

# Per-notch multiplier; scroll up (`event.y > 0`) zooms in (factor < 1).
function Makie.process_interaction(s::LogSafeScrollZoom, event::Makie.ScrollEvent,
                                   ax::Makie.Axis)
    zoom = Float64(event.y)
    (isfinite(zoom) && zoom != 0) || return Makie.Consume(false)
    factor = (1.0 - s.speed) ^ zoom
    zoom_axis!(ax, factor)
    return Makie.Consume(true)
end

const LOGSAFE_SCROLL_SPEED = 0.10

"""
    guard_log_zoom!(ax; speed=LOGSAFE_SCROLL_SPEED) -> ax

Make scroll-wheel zooming safe on a `log10`-scaled `Axis`. Replaces the
default `:scrollzoom` interaction with a [`LogSafeScrollZoom`](@ref) so that
zooming out can never push a log-axis limit to `≤ 0` and crash the GLMakie
render loop with `AssertionError: all(low .<= high)`.

No-op on axes whose `x` and `y` scales are both non-log (linear axes keep
Makie's native cursor-anchored scroll zoom). Safe to call on axes whose scale
is decided at runtime — it only swaps the interaction when at least one axis is
actually log-scaled.
"""
function guard_log_zoom!(ax::Makie.Axis; speed::Real = LOGSAFE_SCROLL_SPEED)
    (_is_log_scale(ax.xscale[]) || _is_log_scale(ax.yscale[])) || return ax
    try
        Makie.deregister_interaction!(ax, :scrollzoom)
    catch
        # No default :scrollzoom registered (or already replaced) — nothing to
        # remove; registering below still installs the safe handler.
    end
    Makie.register_interaction!(ax, :scrollzoom, LogSafeScrollZoom(Float64(speed)))
    return ax
end

"""
    zoom_shortcut_bindings(ax; also=(), zoom_in=ZOOM_IN_FACTOR,
                           zoom_out=ZOOM_OUT_FACTOR, on_change=nothing)
        -> Vector{ShortcutBinding}

Build the standard centered-zoom bindings for `ax`: `=`/`+` and numpad `+`
zoom in, `-` and numpad `-` zoom out. The numpad variants carry no
description so the help window lists each action only once.

Keyword arguments:
- `also`: extra axes kept in sync. Either an iterable of axes, or a zero-arg
  function returning such an iterable (use the function form when the set of
  axes changes at runtime, e.g. a compare axis that toggles visibility).
  `nothing` entries are skipped, so `() -> (visible ? ax2 : nothing,)` works.
- `on_change`: optional one-arg callback invoked with a short status string
  (`"Zoomed in."` / `"Zoomed out."`) after each zoom — wire it to the view's
  `set_status!` to echo the action in the status bar.
"""
function zoom_shortcut_bindings(ax;
                                also = (),
                                zoom_in::Real  = ZOOM_IN_FACTOR,
                                zoom_out::Real = ZOOM_OUT_FACTOR,
                                on_change::Union{Nothing,Function} = nothing)
    _extra() = also isa Function ? also() : also
    function _zoom(factor::Real, label::AbstractString)
        zoom_axis!(ax, factor)
        for a in _extra()
            a === nothing && continue
            zoom_axis!(a, factor)
        end
        on_change === nothing || on_change(label)
        return nothing
    end
    return ShortcutBinding[
        ShortcutBinding(Keyboard.equal,       () -> _zoom(zoom_in,  "Zoomed in.");
                        description = "zoom in"),
        ShortcutBinding(Keyboard.kp_add,      () -> _zoom(zoom_in,  "Zoomed in.")),
        ShortcutBinding(Keyboard.minus,       () -> _zoom(zoom_out, "Zoomed out.");
                        description = "zoom out"),
        ShortcutBinding(Keyboard.kp_subtract, () -> _zoom(zoom_out, "Zoomed out.")),
    ]
end

"""
    _key_label(key) -> String

Pretty name for a `Makie.Keyboard.Button` used by `format_shortcut`.
"""
function _key_label(key::Makie.Keyboard.Button)
    s = String(Symbol(key))
    aliases = (
        "left"      => "←",
        "right"     => "→",
        "up"        => "↑",
        "down"      => "↓",
        "page_up"   => "PgUp",
        "page_down" => "PgDn",
        "home"      => "Home",
        "end"       => "End",
        "escape"    => "Esc",
        "slash"     => "/",
        "equal"       => "+",
        "minus"       => "−",
        "kp_add"      => "+",
        "kp_subtract" => "−",
        "tab"       => "Tab",
        "space"     => "Space",
        "enter"     => "Enter",
        "backspace" => "BkSp",
        "delete"    => "Del",
    )
    for (k, v) in aliases
        s == k && return v
    end
    length(s) == 1 && return uppercase(s)
    return uppercase(s[1:1]) * s[2:end]
end

"""
    format_shortcut(b::ShortcutBinding) -> String

Render a single binding, e.g. `"R"`, `"Shift + /"`, `"PgUp"`.
"""
function format_shortcut(b::ShortcutBinding)
    name = _key_label(b.key)
    if b.modifier === :none
        return name
    elseif b.modifier === :shift
        return "Shift + " * name
    elseif b.modifier === :ctrl
        return "Ctrl + " * name
    else
        return "Alt + " * name
    end
end

"""
    format_shortcut_help(bindings; sep=" · ") -> String

One-line summary of `bindings` suitable for the status bar.
Bindings without a description are skipped.
"""
function format_shortcut_help(bindings::AbstractVector{ShortcutBinding};
                              sep::AbstractString = " · ")
    parts = String[]
    for b in bindings
        isempty(b.description) && continue
        push!(parts, string(format_shortcut(b), " ", b.description))
    end
    return join(parts, sep)
end

"""
    shortcut_help_message(bindings; prefix="Shortcuts — ") -> String

Status-bar message listing all described shortcuts. Bindings without a
description are skipped, matching `format_shortcut_help`.
"""
function shortcut_help_message(bindings::AbstractVector{ShortcutBinding};
                               prefix::AbstractString = "Shortcuts — ")
    help = format_shortcut_help(bindings)
    return isempty(help) ? String(prefix) * "none" : String(prefix) * help
end

"""
    open_shortcut_help_window(bindings;
                              title="Keyboard shortcuts",
                              theme=current_ui_theme(),
                              activate_gl=true,
                              display_fig=true,
                              figsize=nothing) -> Figure

Open a small Makie figure listing the keyboard shortcuts as a two-column
table (`shortcut · description`). Bindings without a description are
skipped, matching `format_shortcut_help` / `shortcut_help_message`.

Used by the "Help" button of each viewer; the same call signature is
exposed publicly so that downstream code can build a Help window for a
custom binding set.

Keyword arguments:
- `title`     : window header text.
- `theme`     : `MANTAUITheme` instance (defaults to `current_ui_theme()`).
- `activate_gl`: when `false`, fall back to a headless backend via
  `pick_backend!(false)` — required for the headless test suite.
- `display_fig`: when `false`, build the figure but do not call `display`.
- `figsize`   : optional `(w, h)` override; default scales the height with
  the number of described bindings.
"""
function open_shortcut_help_window(bindings::AbstractVector{ShortcutBinding};
                                   title::AbstractString = "Keyboard shortcuts",
                                   theme = current_ui_theme(),
                                   activate_gl::Bool = true,
                                   display_fig::Bool = true,
                                   figsize::Union{Nothing,Tuple{Int,Int}} = nothing)
    # Only show documented bindings. The "this help" entry itself stays in
    # so users can see how to reopen the window.
    listed = collect(b for b in bindings if !isempty(b.description))
    n = length(listed)

    # Headless-safe backend selection. `pick_backend!` is defined later in
    # the include chain (helpers/Backend.jl); it is resolved at call time.
    backend_sym = pick_backend!(activate_gl)

    # Figure geometry — width fixed, height scales with the binding count.
    row_h  = 26
    base_h = 110
    w = figsize === nothing ? 560                       : Int(first(figsize))
    h = figsize === nothing ? base_h + max(n, 1) * row_h : Int(last(figsize))

    fig = Figure(size = (w, h); backgroundcolor = theme.background)

    # Title row
    Label(fig[1, 1]; text = String(title),
          halign = :center, tellwidth = false,
          fontsize = 18, color = theme.text,
          padding = (0, 0, 10, 6))

    # Subtle separator
    Box(fig[2, 1]; color = theme.border, strokecolor = theme.border,
        strokewidth = 0, height = 1, tellwidth = false)

    # Body — two-column GridLayout for shortcut / description.
    body = GridLayout(fig[3, 1]; alignmode = Outside(8, 8, 4, 4))
    if n == 0
        Label(body[1, 1]; text = "(no documented shortcuts)",
              halign = :center, tellwidth = false,
              fontsize = 14, color = theme.text_muted)
    else
        for (i, b) in enumerate(listed)
            Label(body[i, 1]; text = format_shortcut(b),
                  halign = :right, tellwidth = false,
                  fontsize = 14, color = theme.accent_strong,
                  padding = (8, 14, 3, 3))
            Label(body[i, 2]; text = b.description,
                  halign = :left, tellwidth = false,
                  fontsize = 14, color = theme.text,
                  padding = (0, 8, 3, 3))
        end
        # Keep the shortcut column narrow relative to the description column
        # so long descriptions get the space they need.
        colsize!(body, 1, Relative(0.38))
        colsize!(body, 2, Relative(0.62))
    end

    # Footer hint
    Label(fig[4, 1]; text = "Press Shift + / or the Help button to reopen.",
          halign = :center, tellwidth = false,
          fontsize = 11, color = theme.text_muted,
          padding = (0, 0, 8, 8))

    # Row sizing — title / separator / body / footer.
    rowsize!(fig.layout, 1, Auto())
    rowsize!(fig.layout, 2, Auto())
    rowsize!(fig.layout, 3, Auto())
    rowsize!(fig.layout, 4, Auto())

    # Open the Help in a *dedicated* GLMakie screen so it appears as a
    # second window next to the main MANTA UI. Without this, GLMakie reuses
    # its default screen and the Help replaces the main viewer — which is
    # what made users feel that closing the Help also closed everything.
    # CairoMakie / headless fall back to plain `display(fig)`.
    if display_fig
        if backend_sym === :GLMakie
            try
                # `title` keyword gives the OS window a clear label so users
                # can tell the Help apart from the main MANTA window in their
                # taskbar / window switcher.
                screen = GLMakie.Screen(; title = String(title),
                                          focus_on_show = true)
                display(screen, fig)
            catch e
                @warn "MANTA: failed to open help in dedicated GLMakie window, falling back to default display" exception = (e, catch_backtrace())
                try
                    display(fig)
                catch
                end
            end
        else
            display(fig)
        end
    end
    return fig
end
