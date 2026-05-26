# path: src/views/cube/CompareBundle.jl
#
# Comparison / dual-cube / reprojection closures for the 3D cube viewer.
#
# Extracted from src/views/CubeView.jl during a structural refactor
# (no behaviour change). Covers:
#   - show/hide the comparison loader UI strip
#   - OS-native file-picker (macOS / Linux / Windows)
#   - WCS compatibility and reprojection checks
#   - loading and aligning the second cube
#
# Entry point: `_cube_compare_bundle(; ...)`.
# Returns a NamedTuple of closures destructured at the call site.

"""
    _cube_compare_bundle(; kwargs...) -> NamedTuple

Build the comparison-cube management closures for the cube viewer.
Returned fields:

| field                 | purpose                                                  |
|-----------------------|----------------------------------------------------------|
| `show_compare_loader!`| reveal the path textbox + Load button                   |
| `hide_compare_loader!`| collapse the loader strip (leaves mode menu if loaded)   |
| `resolve_compare_path`| canonicalise a path relative to the primary cube        |
| `pick_compare_path`   | OS-native file dialog; returns "" on cancel / no backend |
| `load_compare_cube!`  | validate, read, align, and activate the second cube     |

`wcs_axes_compatible`, `wcs_axes_reprojectable`, and `prepare_compare_cube`
are internal helpers bundled here for locality but not returned.
"""
function _cube_compare_bundle(;
    filepath,
    wcs,
    data,
    siz,
    compare_data,
    compare_visible,
    compare_name,
    compare_path_current,
    btn_show_compare,
    compare_mode_menu,
    compare_path_box,
    btn_load_compare,
    compare_state_label,
    ax_cmp,
    img_colorbar_cmp,
    img_grid,
    show_grid,
    ui_text_muted,
    ui_success,
    refresh_spectrum!,
    set_status!,
    set_block_visible!,
    set_box_text!,
)
    # ------------------------------------------------------------------
    # Reveal the path textbox + "Load cube" button.
    # ------------------------------------------------------------------
    function show_compare_loader!()
        btn_show_compare.label[]    = ""
        btn_show_compare.width[]    = 0
        compare_mode_menu.width[]   = 0
        compare_path_box.placeholder[] = "second cube FITS path"
        compare_path_box.width[]    = 310
        btn_load_compare.label[]    = "Load cube"
        btn_load_compare.width[]    = 104
        compare_state_label.text[]  = "Comparison: waiting for cube path"
        compare_state_label.color[] = ui_text_muted
        set_status!("Enter the second cube FITS path, then click Load cube.")
        nothing
    end

    # ------------------------------------------------------------------
    # Collapse the loader strip; optionally restore the "Compare cube…"
    # button when no cube is loaded yet.
    # ------------------------------------------------------------------
    function hide_compare_loader!()
        btn_show_compare.label[]  = ""
        btn_show_compare.width[]  = 0
        compare_path_box.placeholder[] = ""
        compare_path_box.width[]  = 0
        btn_load_compare.label[]  = ""
        btn_load_compare.width[]  = 0
        compare_mode_menu.width[] = compare_visible[] ? 150 : 0
        if !compare_visible[]
            btn_show_compare.label[]        = "Compare cube..."
            btn_show_compare.width[]        = 138
            compare_state_label.text[]      = "Comparison: no cube loaded"
            compare_state_label.color[]     = ui_text_muted
        end
        nothing
    end

    # ------------------------------------------------------------------
    # Resolve `path_txt` relative to the primary cube directory when it
    # is not an absolute path that already exists on disk.
    # ------------------------------------------------------------------
    function resolve_compare_path(path_txt::AbstractString)
        p = strip(String(path_txt))
        isempty(p) && return ""
        isfile(p)  && return p
        beside_primary = joinpath(dirname(abspath(filepath)), p)
        isfile(beside_primary) && return beside_primary
        return p
    end

    # ------------------------------------------------------------------
    # Open a native OS file dialog.
    # Returns "" when the user cancels or no dialog backend is available
    # (CI / headless). Falls back gracefully so callers just see "".
    # ------------------------------------------------------------------
    function pick_compare_path()
        initial = filepath != "" ? dirname(abspath(filepath)) : pwd()
        out = ""
        try
            if Sys.isapple()
                script = "POSIX path of (choose file with prompt \"Select FITS cube to compare\" default location POSIX file \"$(initial)\")"
                out = readchomp(`osascript -e $script`)
            elseif Sys.islinux()
                zenity  = Sys.which("zenity")
                if zenity !== nothing
                    out = readchomp(`$zenity --file-selection --title=Select\ FITS\ cube\ to\ compare --filename=$(initial)/`)
                else
                    kdialog = Sys.which("kdialog")
                    if kdialog !== nothing
                        out = readchomp(`$kdialog --getopenfilename $initial "FITS (*.fits *.fits.gz)"`)
                    end
                end
            elseif Sys.iswindows()
                ps = """
                Add-Type -AssemblyName System.Windows.Forms | Out-Null;
                \$f = New-Object System.Windows.Forms.OpenFileDialog;
                \$f.Filter = 'FITS (*.fits;*.fits.gz)|*.fits;*.fits.gz|All files (*.*)|*.*';
                \$f.InitialDirectory = '$initial';
                if (\$f.ShowDialog() -eq 'OK') { Write-Output \$f.FileName }
                """
                out = readchomp(`powershell -NoProfile -Command $ps`)
            end
        catch e
            @debug "Native file dialog failed; falling back to textbox." exception=e
            return ""
        end
        return strip(String(out))
    end

    # ------------------------------------------------------------------
    # Lightweight WCS compatibility check.
    # When both cubes declare a WCS on a given axis, CTYPE base / CRVAL /
    # CRPIX / CDELT must agree (relative tolerance 1e-6).
    # Axes where neither cube has WCS are accepted.
    # Axes where only one cube has WCS are rejected.
    # ------------------------------------------------------------------
    wcs_axes_compatible(a::AbstractVector, b::AbstractVector) = begin
        length(a) == length(b) || return false
        for (xa, xb) in zip(a, b)
            if !xa.available && !xb.available
                continue
            end
            xa.available == xb.available || return false
            xa.ctype_base == xb.ctype_base || return false
            isapprox(xa.crval, xb.crval; rtol = 1e-6, atol = 0.0) || return false
            isapprox(xa.cdelt, xb.cdelt; rtol = 1e-6, atol = 0.0) || return false
            isapprox(xa.crpix, xb.crpix; rtol = 0.0, atol = 1e-6) || return false
        end
        return true
    end

    # Looser check for visual reprojection: axes must be separable and have
    # matching physical meanings, but CRPIX/CDELT may differ because that is
    # exactly what the interpolation step reconciles.
    wcs_axes_reprojectable(a::AbstractVector, b::AbstractVector) = begin
        length(a) == length(b) || return false
        for (xa, xb) in zip(a, b)
            xa.available && xb.available || return false
            xa.ctype_base == xb.ctype_base || return false
        end
        return true
    end

    # ------------------------------------------------------------------
    # Align `raw_cmp` to the primary cube's grid, choosing the most
    # faithful strategy available (native → reproject → resample).
    # Returns (aligned_array, grid_note_string).
    # ------------------------------------------------------------------
    function prepare_compare_cube(raw_cmp, cmp_wcs)
        if size(raw_cmp) == siz && wcs_axes_compatible(wcs, cmp_wcs)
            return (Float32.(raw_cmp), "native grid")
        end
        if wcs_axes_reprojectable(wcs, cmp_wcs)
            return (reproject_cube_linear(raw_cmp, cmp_wcs, wcs, siz), "reprojected to primary WCS")
        end
        return (resample_cube_linear(raw_cmp, siz), "resampled to primary pixel grid")
    end

    # ------------------------------------------------------------------
    # Full pipeline: validate path → read FITS → align → activate.
    # Returns `true` on success, `false` on any recoverable error.
    # ------------------------------------------------------------------
    function load_compare_cube!(path_txt::AbstractString)
        cmp_path = resolve_compare_path(path_txt)
        if isempty(cmp_path)
            set_status!("Provide a second cube FITS path before enabling dual view.")
            return false
        end
        if !isfile(cmp_path)
            set_status!("Second cube not found: $(cmp_path)")
            return false
        end
        raw_cmp, cmp_header = try
            FITS(cmp_path) do f
                (read(f[1]), read_header(f[1]))
            end
        catch e
            msg = "Failed to read second cube: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
            return false
        end
        if ndims(raw_cmp) != 3
            set_status!("Second cube must be 3D, got ndims=$(ndims(raw_cmp)).")
            return false
        end
        cmp_wcs = try
            read_simple_wcs(cmp_header, 3)
        catch
            SimpleWCSAxis[]
        end
        cmp_prepared, grid_note = try
            prepare_compare_cube(raw_cmp, cmp_wcs)
        catch e
            msg = "Failed to align second cube: $(sprint(showerror, e))"
            set_status!(msg)
            @error msg exception=(e, catch_backtrace())
            return false
        end
        compare_data[]         = cmp_prepared
        compare_name[]         = String(replace(basename(cmp_path), r"\.fits(\.gz)?$" => ""))
        compare_path_current[] = cmp_path
        compare_visible[]      = true
        colsize!(img_grid, 2, Auto())
        colsize!(img_grid, 4, Auto())
        set_block_visible!(ax_cmp, true)
        set_block_visible!(img_colorbar_cmp, true)
        ax_cmp.xgridvisible[] = show_grid[]
        ax_cmp.ygridvisible[] = show_grid[]
        autolimits!(ax_cmp)
        hide_compare_loader!()
        compare_state_label.text[]  = "Comparison: $(compare_name[]) ($(grid_note))"
        compare_state_label.color[] = ui_success
        # Pull the spectrum line in immediately so the user sees overlay
        # without having to click anything.
        refresh_spectrum!()
        set_status!("Comparison cube loaded: $(cmp_path) ($(grid_note)).")
        return true
    end

    return (;
        show_compare_loader!,
        hide_compare_loader!,
        resolve_compare_path,
        pick_compare_path,
        load_compare_cube!,
    )
end
