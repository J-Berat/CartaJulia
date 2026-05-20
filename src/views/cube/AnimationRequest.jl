# path: src/views/cube/AnimationRequest.jl
#
# Pure helper for the 3D cube viewer: parses the current state of the four
# animation textboxes (start / stop / step / FPS) and the pingpong checkbox
# into a validated GIF request (frames vector + fps + status message).
#
# Lifted out of `_view_cube` in src/views/CubeView.jl during a structural
# refactor. The original closure captured `axis`, `siz`, the four textboxes
# and `pingpong_chk` from the enclosing scope. This top-level version takes
# them explicitly so it can be tested in isolation and so each call site at
# the use site is self-documenting.
#
# All parsing/validation is delegated to `parse_gif_request` in
# src/helpers/UIBits.jl — this helper exists only to wire the UI widgets to
# that pure parser.

"""
    current_animation_request(axis_val, siz, start_box, stop_box, step_box, fps_box,
                              pingpong_chk)
        -> (ok::Bool, frames::Vector{Int}, fps::Int, msg::String)

Read the start / stop / step / FPS textboxes and the pingpong checkbox of
the cube viewer, then forward them to `parse_gif_request` along with the
maximum index allowed for the currently selected slicing axis
(`siz[axis_val]`).

Arguments:
- `axis_val`     -- integer slicing axis currently selected (1 / 2 / 3).
- `siz`          -- the cube size tuple `(nx, ny, nz)`. `siz[axis_val]` gives
                    the inclusive upper bound for valid frame indices.
- `start_box`, `stop_box`, `step_box`, `fps_box` -- `Makie.Textbox`-like
                    widgets read through `get_box_str`. Any object whose
                    string contents `get_box_str` can extract works.
- `pingpong_chk` -- a `Makie.Checkbox`-like object whose `.checked` field is
                    an `Observable{Bool}` (`pingpong_chk.checked[]` is the
                    current state).

Returns the same 4-tuple as `parse_gif_request`. Callers should report `msg`
to the status bar regardless of `ok`, and only consume `frames` / `fps`
when `ok == true`.
"""
function current_animation_request(axis_val::Integer,
                                   siz,
                                   start_box,
                                   stop_box,
                                   step_box,
                                   fps_box,
                                   pingpong_chk)
    amax = siz[axis_val]
    return parse_gif_request(
        get_box_str(start_box),
        get_box_str(stop_box),
        get_box_str(step_box),
        get_box_str(fps_box),
        amax;
        pingpong = pingpong_chk.checked[],
    )
end
