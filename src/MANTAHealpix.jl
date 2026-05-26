# HEALPix Mollweide viewer with interactive zoom.
# API publique : `is_healpix_fits`, `read_healpix_map`, `mollweide_grid`,
# `manta_healpix(filepath; ...)`.
#
# Compatible avec les conventions de `manta(...)` (zoom right-drag, reset,
# colormap, vlims, save image, échelles lin/log10/ln).

using GLMakie, CairoMakie, Makie, Observables, FITSIO, LaTeXStrings
using Healpix

############################
# Détection / Lecture
############################

"""
    is_healpix_fits(path) -> Bool

Heuristique : un fichier HEALPix expose `PIXTYPE = 'HEALPIX'` dans le
header d'une extension BinTable. On lit les headers sans charger les
données.
"""
function is_healpix_fits(path::AbstractString)
    isfile(path) || return false
    try
        FITS(path) do f
            for hdu in f
                hdr = read_header(hdu)
                if haskey(hdr, "PIXTYPE")
                    val = uppercase(strip(string(hdr["PIXTYPE"])))
                    val == "HEALPIX" && return true
                end
            end
            return false
        end
    catch
        return false
    end
end

"""
    read_healpix_map(path; column=1) -> (HealpixMap, header_dict)

Lit la carte HEALPix (RING ou NESTED auto-détecté). `column` est le
numéro de colonne dans la BinTable (1 pour I_STOKES, etc.).
Retourne aussi le header de l'extension lue, utile pour récupérer
unités et noms.
"""
function read_healpix_map(path::AbstractString; column::Int=1)
    m = Healpix.readMapFromFITS(String(path), column, Float64)
    hdr = FITS(path) do f
        # Le header de la HDU 2 (BinTable HEALPix) contient les infos utiles.
        h = length(f) >= 2 ? read_header(f[2]) : read_header(f[1])
        Dict{String,Any}(string(k) => h[k] for k in keys(h))
    end
    return m, hdr
end

# ---- HEALPix projection / graticule helpers ----
include("views/HealpixProjection.jl")

function manta_healpix(
    pixels::AbstractArray;
    title::AbstractString = "RGB HEALPix",
    nx::Int = 1400,
    ny::Int = 700,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    show_graticule::Bool = true,
)
    rgb_pixels = as_rgb_pixels(pixels)
    img = mollweide_color_grid(rgb_pixels; nx=nx, ny=ny)
    pick_backend!(activate_gl)
    fig = Figure(size = _pick_fig_size(figsize))
    ax = Axis(
        fig[1, 1];
        title = make_main_title(title),
        aspect = DataAspect(),
        xticksvisible = false,
        yticksvisible = false,
        xticklabelsvisible = false,
        yticklabelsvisible = false,
        bottomspinevisible = false,
        topspinevisible = false,
        leftspinevisible = false,
        rightspinevisible = false,
    )
    image!(ax, (-2f0, 2f0), (-1f0, 1f0), permutedims(img))
    set_mollweide_view!(ax, -2.0, 2.0, -1.0, 1.0)
    graticule = draw_mollweide_graticule!(ax)
    set_graticule_visible!(graticule, show_graticule)
    ell_x = [2cos(t) for t in LinRange(0, 2π, 200)]
    ell_y = [sin(t) for t in LinRange(0, 2π, 200)]
    lines!(ax, ell_x, ell_y; color=:black, linewidth=0.8)
    keepalive!(fig)
    on(fig.scene.events.window_open) do is_open
        is_open || forget!(fig)
    end
    display_fig && display(fig)
    return fig
end

function manta_healpix_panels(
    panels::Vararg{Any,N};
    titles = nothing,
    cmaps = nothing,
    clims = nothing,
    nx::Int = 1400,
    ny::Int = 700,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    show_graticule::Bool = true,
) where {N}
    N >= 1 || throw(ArgumentError("Provide at least one HEALPix panel."))
    pick_backend!(activate_gl)
    fig_size = _pick_fig_size(figsize)
    # Layout responsive : on rétrécit la colorbar (et son row Fixed) sur les
    # petites fenêtres pour que la carte reste lisible — même logique que
    # les vues map / cube HEALPix.
    compact_layout = fig_size[1] <= 1500 || fig_size[2] <= 950
    cbar_row_h     = compact_layout ? 32 : 44
    cbar_height    = compact_layout ? 12 : 16
    fig = Figure(size = fig_size)
    rowgap!(fig.layout, -8)
    title_at(i) = titles === nothing ? "panel $(i)" : String(titles[i])
    cmap_at(i) = cmaps === nothing ? :inferno : cmaps[i]
    clim_at(i, vals) = clims === nothing ? clamped_extrema(vals) : clims[i]
    for (i, panel) in enumerate(panels)
        ax = Axis(
            fig[1, i];
            title = make_main_title(title_at(i)),
            aspect = DataAspect(),
            xticksvisible = false,
            yticksvisible = false,
            xticklabelsvisible = false,
            yticklabelsvisible = false,
            bottomspinevisible = false,
            topspinevisible = false,
            leftspinevisible = false,
            rightspinevisible = false,
        )
        if is_rgb_like(panel)
            img = mollweide_color_grid(as_rgb_pixels(panel); nx=nx, ny=ny)
            image!(ax, (-2f0, 2f0), (-1f0, 1f0), permutedims(img))
        else
            vals = _mollweide_scalar_grid(panel; nx=nx, ny=ny)
            plot_vals = permutedims(vals)
            hm = heatmap!(
                ax,
                LinRange(-2f0, 2f0, nx),
                LinRange(-1f0, 1f0, ny),
                plot_vals;
                colormap=cmap_at(i),
                colorrange=clim_at(i, vals),
                nan_color=:white,
            )
            Colorbar(
                fig[2, i],
                hm;
                vertical=false,
                height=cbar_height,
                tellwidth=false,
                halign=:center,
            )
            rowsize!(fig.layout, 1, Relative(1))
            rowsize!(fig.layout, 2, Fixed(cbar_row_h))
        end
        set_mollweide_view!(ax, -2.0, 2.0, -1.0, 1.0)
        graticule = draw_mollweide_graticule!(ax)
        set_graticule_visible!(graticule, show_graticule)
        ell_x = [2cos(t) for t in LinRange(0, 2π, 200)]
        ell_y = [sin(t) for t in LinRange(0, 2π, 200)]
        lines!(ax, ell_x, ell_y; color=:black, linewidth=0.8)
    end
    keepalive!(fig)
    on(fig.scene.events.window_open) do is_open
        is_open || forget!(fig)
    end
    display_fig && display(fig)
    return fig
end

"""
    detect_velocity_axis(filepath, ndim) -> (axis, v0, dv, vunit) | nothing

Scan les `CTYPE{i}` (i=1..ndim) de la HDU primaire pour identifier l'axe
vitesse/fréquence. Reconnaît `VRAD`, `VOPT`, `VELO`, `VELOCITY`, `FREQ`,
`FELO`. Si trouvé, lit `CRVAL/CDELT/CRPIX/CUNIT` du même axe et calcule
`v0 = CRVAL - (CRPIX - 1) * CDELT`, `dv = CDELT`. Conversion `m/s → km/s`.

Retourne `nothing` si aucun CTYPE vitesse n'est trouvé. La dim non
détectée est alors l'axe HEALPix.
"""
function detect_velocity_axis(filepath::AbstractString, ndim::Int)
    try
        FITS(String(filepath)) do f
            h = read_header(f[1])
            v_axis = 0
            ctype_found = ""
            for i in 1:ndim
                k = "CTYPE$(i)"
                haskey(h, k) || continue
                ct = uppercase(strip(String(h[k])))
                # On accepte les CTYPE typiques d'un axe spectral : vitesse
                # radio/optique, fréquence, longueur d'onde. On veut juste
                # identifier l'axe non-spatial du cube.
                if startswith(ct, "VRAD") || startswith(ct, "VOPT") ||
                   startswith(ct, "VELO") || startswith(ct, "FREQ") ||
                   startswith(ct, "FELO") || startswith(ct, "WAVE") ||
                   startswith(ct, "AWAV") || ct == "VELOCITY"
                    v_axis = i; ctype_found = ct; break
                end
            end
            v_axis == 0 && return nothing
            kCRVAL = "CRVAL$(v_axis)"
            kCDELT = "CDELT$(v_axis)"
            (haskey(h, kCRVAL) && haskey(h, kCDELT)) || return nothing
            crval = Float64(h[kCRVAL])
            cdelt = Float64(h[kCDELT])
            crpix = haskey(h, "CRPIX$(v_axis)") ? Float64(h["CRPIX$(v_axis)"]) : 1.0
            unit_raw = haskey(h, "CUNIT$(v_axis)") ?
                lowercase(strip(String(h["CUNIT$(v_axis)"]))) : ""
            v0 = crval - (crpix - 1) * cdelt
            dv = cdelt
            unit_norm = unit_raw
            if unit_raw in ("m/s", "m s-1", "m.s-1")
                v0 *= 1e-3; dv *= 1e-3; unit_norm = "km/s"
            elseif unit_raw in ("hz",)
                unit_norm = "Hz"
            elseif unit_raw in ("khz", "mhz", "ghz")
                unit_norm = unit_raw
            elseif isempty(unit_raw)
                # Heuristique : si CTYPE est une vitesse, on suppose km/s ;
                # si c'est une fréquence, on suppose Hz.
                unit_norm = startswith(ctype_found, "F") ? "Hz" : "km/s"
            end
            return (v_axis, v0, dv, unit_norm)
        end
    catch
        return nothing
    end
end

"""
    valid_healpix_npix(n) -> Int

Retourne `nside` si `n = 12·nside²`, sinon 0. Sert à détecter si une
dimension d'un tableau 2D est un nombre HEALPix valide.
"""
function valid_healpix_npix(n::Integer)
    n <= 0 && return 0
    if n % 12 == 0
        s2 = n ÷ 12
        s = isqrt(s2)
        s*s == s2 && (s & (s-1)) == 0 && return s   # nside puissance de 2
    end
    return 0
end

"""
    mollweide_xy_to_lonlat(x, y) -> (lon_deg, lat_deg) | nothing

Inverse de la projection Mollweide. Retourne `nothing` si (x,y) est hors
ellipse. Longitude ∈ (-180°, 180°], latitude ∈ [-90°, 90°].
"""
@inline function mollweide_xy_to_lonlat(x::Real, y::Real)
    (x^2 / 4 + y^2 > 1) && return nothing
    θaux = asin(y)
    sinφ = (2θaux + sin(2θaux)) / π
    abs(sinφ) > 1 && return nothing
    lat = asin(sinφ)
    lon = π * x / (2 * cos(θaux))
    abs(lon) > π && return nothing
    return (rad2deg(lon), rad2deg(lat))
end

############################
# Viewer interactif
############################

"""
    manta_healpix(filepath::String;
                  cmap=:inferno, vmin=nothing, vmax=nothing,
                  invert=false, scale=:lin, column=1,
                  nx=1400, ny=700,
                  figsize=nothing, save_dir=nothing,
                  activate_gl=true, display_fig=true)

Visualiseur interactif HEALPix en projection Mollweide.

- **Zoom** : maintenir clic-droit et glisser pour dessiner un rectangle ;
  le bouton "Reset zoom" rétablit la vue complète.
- **Hover/clic gauche** : affiche `(l, b)` galactiques et la valeur du
  pixel.
- **Échelle** : `:lin`, `:log10`, `:ln` (sélectionnable au runtime).
- **Contrast** : auto (quantiles 2/98 % en lin, 5/98 % en log) ou
  `vmin`/`vmax` manuels.

Retourne la `Figure` GLMakie.
"""
function manta_healpix(
    filepath::String;
    cmap::Symbol = :inferno,
    vmin = nothing,
    vmax = nothing,
    invert::Bool = false,
    scale::Symbol = :lin,
    column::Int = 1,
    nx::Int = 1400,
    ny::Int = 700,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    hist_mode::Symbol = :bars,
    hist_bins::Int = 64,
    hist_xlimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    hist_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
)
    ds = load_dataset(filepath; column = column)
    ds isa HealpixMapDataset || throw(ArgumentError(
        "MANTA: expected a HEALPix map in $(abspath(filepath)), got $(typeof(ds))."))
    return _view_healpix_map(ds;
        cmap = cmap, vmin = vmin, vmax = vmax, invert = invert,
        scale = scale, nx = nx, ny = ny, figsize = figsize,
        save_dir = save_dir, activate_gl = activate_gl,
        display_fig = display_fig,
        hist_mode = hist_mode, hist_bins = hist_bins,
        hist_xlimits = hist_xlimits, hist_ylimits = hist_ylimits)
end

"""
    manta_healpix_cube(filepath::String;
                       cmap=:inferno, vmin=nothing, vmax=nothing,
                       invert=false, scale=:lin,
                       v0=0.0, dv=1.0, vunit="km/s",
                       nx=1200, ny=600,
                       figsize=nothing, save_dir=nothing,
                       activate_gl=true, display_fig=true)

Visualiseur interactif d'un **cube HEALPix-PPV** stocké comme un tableau
2D `(npix, nv)` ou `(nv, npix)` dans un FITS classique. Affiche :

- en haut, la **carte Mollweide** du canal courant ;
- en bas, le **spectre** au pixel cliqué.

Contrôles :
- slider "Channel" → change de canal (réutilise l'index Mollweide
  précalculé, pas de recalcul de projection).
- right-drag → zoom rectangulaire sur la Mollweide.
- left-click → sélectionne un pixel HEALPix, met à jour le spectre.
- échelle, contraste manuel, colormap, invert colormap, save PNG.

`v0`, `dv`, `vunit` : axe vitesse `v(j) = v0 + (j-1)*dv` pour le spectre.
"""
function manta_healpix_cube(
    filepath::String;
    cmap::Symbol = :inferno,
    vmin = nothing,
    vmax = nothing,
    invert::Bool = false,
    scale::Symbol = :lin,
    v0::Real = 0.0,
    dv::Real = 1.0,
    vunit::AbstractString = "km/s",
    nx::Int = 1200,
    ny::Int = 600,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    hist_mode::Symbol = :bars,
    hist_bins::Int = 64,
    hist_xlimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    hist_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    spec_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    moment_threshold::Real = 0.0,
    moment_nsigma::Union{Nothing,Real} = nothing,
    moment_channels::Union{Nothing,AbstractVector{<:Integer}} = nothing,
)
    ds = load_dataset(filepath; v0 = v0, dv = dv, vunit = vunit)
    ds isa HealpixCubeDataset || throw(ArgumentError(
        "MANTA: expected a HEALPix PPV cube in $(abspath(filepath)), got $(typeof(ds))."))
    return _view_healpix_cube(ds;
        cmap = cmap, vmin = vmin, vmax = vmax, invert = invert,
        scale = scale, nx = nx, ny = ny, figsize = figsize,
        save_dir = save_dir, activate_gl = activate_gl,
        display_fig = display_fig,
        hist_mode = hist_mode, hist_bins = hist_bins,
        hist_xlimits = hist_xlimits, hist_ylimits = hist_ylimits,
        spec_ylimits = spec_ylimits,
        moment_threshold = moment_threshold,
        moment_nsigma = moment_nsigma,
        moment_channels = moment_channels)
end

"""
    _vunit_quantity_word(vunit) -> String

Classify a spectral CUNIT-style string into the matching quantity word
("velocity", "frequency", "wavelength"). HEALPix-PPV datasets don't carry
a CTYPE through the viewer pipeline, so we infer from the unit instead.
Defaults to "velocity" because the historical default was km/s.
"""
function _vunit_quantity_word(vunit::AbstractString)
    u = lowercase(strip(String(vunit)))
    if u in ("hz", "khz", "mhz", "ghz", "thz")
        return "frequency"
    elseif u in ("m", "nm", "um", "µm", "mm", "cm", "angstrom", "å", "a")
        return "wavelength"
    elseif u == "channel"
        return "value"
    else
        return "velocity"  # km/s, m/s and bare blank fall here
    end
end

function _view_healpix_cube(
    ds::HealpixCubeDataset;
    cmap::Symbol = :inferno,
    vmin = nothing,
    vmax = nothing,
    invert::Bool = false,
    scale::Symbol = :lin,
    nx::Int = 1200,
    ny::Int = 600,
    figsize::Union{Nothing,Tuple{Int,Int}} = nothing,
    save_dir::Union{Nothing,AbstractString} = nothing,
    activate_gl::Bool = true,
    display_fig::Bool = true,
    hist_mode::Symbol = :bars,
    hist_bins::Int = 64,
    hist_xlimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    hist_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    spec_ylimits::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
    moment_threshold::Real = 0.0,
    moment_nsigma::Union{Nothing,Real} = nothing,
    moment_channels::Union{Nothing,AbstractVector{<:Integer}} = nothing,
)
    cube = as_float32(ds.data)
    nside = ds.nside
    npix, nv = size(cube)
    v0_eff = ds.v0
    dv_eff = ds.dv
    vunit_eff = ds.vunit
    data_unit = ds.unit_label
    data_unit_tex = latexstring("\\text{", latex_safe(data_unit), "}")
    fname = ds.source_id
    spec_x = Float32.(v0_eff .+ (0:nv-1) .* dv_eff)
    # Pass the actual Δx through so M0 is "K·(spectral unit)" (e.g. K·km/s,
    # K·Hz) rather than the historical "K·channel" summation. Threshold and
    # channel window are forwarded from the public viewer kwargs so noisy or
    # continuum-subtracted cubes can be cleaned up without a code edit.
    moment_dx_unit = vunit_eff == "channel" ? 1.0 : abs(Float64(dv_eff))
    moment_vecs = moment_vectors(cube, spec_x;
                                 threshold = moment_threshold,
                                 nsigma = moment_nsigma,
                                 channels = moment_channels === nothing ? (1:nv) : moment_channels,
                                 dx = moment_dx_unit)
    # Choose a label that follows the spectral axis: velocity / frequency /
    # wavelength (heuristic on `vunit`, since HealpixCubeDataset only stores
    # the unit string, not a CTYPE).
    spec_word = _vunit_quantity_word(vunit_eff)
    moment_caption(order::Integer) = order == 0 ? "moment 0 [$(data_unit) " * vunit_eff * "]" :
                                     order == 1 ? "mean " * spec_word * " [" * vunit_eff * "]" :
                                                  spec_word * " dispersion [" * vunit_eff * "]"

    # ---------- Précalcul de l'index Mollweide (une fois) ----------
    res = Healpix.Resolution(nside)
    ipix_grid = mollweide_pixel_index(res, nx, ny)   # 0 = hors ellipse

    function projected_vector_image(vals)
        out = fill(NaN32, ny, nx)
        @inbounds for q in eachindex(ipix_grid)
            ip = ipix_grid[q]
            ip == 0 && continue
            v = vals[ip]
            out[q] = (isfinite(v) && v != Float32(Healpix.UNSEEN)) ? v : NaN32
        end
        out
    end

    frame_image(j::Int) = projected_vector_image(@view(cube[:, j]))
    moment_vector(order::Integer) = order == 0 ? moment_vecs[1] : order == 1 ? moment_vecs[2] : moment_vecs[3]
    moment_label(order::Integer) = order == 0 ? "moment 0" : order == 1 ? "moment 1" : "moment 2"
    # Public caption shown in titles: encodes the spectral quantity and unit.
    moment_long_label(order::Integer) = moment_caption(order)

    # ---------- État ----------
    cmap_name   = Observable(cmap)
    invert_cmap = Observable(invert)
    cm_obs = lift(cmap_name, invert_cmap) do name, inv
        base = to_cmap(name); inv ? reverse(base) : base
    end
    scale_mode = Observable(scale)
    chan_idx   = Observable(max(1, nv ÷ 2))
    show_moment = Observable(false)
    moment_order = Observable(0)
    gauss_on = Observable(false)
    sigma = Observable(1.5f0)

    img_raw = lift(chan_idx, show_moment, moment_order) do j, show_mom, ord
        show_mom ? projected_vector_image(moment_vector(ord)) : frame_image(j)
    end
    img_proc = lift(img_raw, gauss_on, sigma) do im, on, σ
        on ? nan_gaussian_filter(im, σ) : im
    end
    img_disp = lift(img_proc, scale_mode) do im, m_
        out = apply_scale(im, m_)
        out2 = similar(out, Float32)
        @inbounds for k in eachindex(out)
            x = out[k]; out2[k] = isfinite(x) ? Float32(x) : NaN32
        end
        out2
    end

    # Échelle de couleur globale, calculée dans l'espace transformé (cohérent
    # entre frames). On évalue les quantiles sur tout le cube pour le mode
    # actif. Hypothèse : les `clims_manual` sont dans le même espace que
    # l'image affichée (i.e. l'utilisateur tape les valeurs après log).
    use_manual = Observable(false)
    clims_manual = Observable((0f0, 1f0))
    function _vector_clims(vals, mode::Symbol)
        fin = Float32[]
        if mode === :lin
            @inbounds for v in vals
                (isfinite(v) && v != Float32(Healpix.UNSEEN)) && push!(fin, Float32(v))
            end
        else
            f = mode === :log10 ? log10 : log
            @inbounds for v in vals
                (isfinite(v) && v != Float32(Healpix.UNSEEN) && v > 0) && push!(fin, Float32(f(v)))
            end
        end
        isempty(fin) && return mode === :lin ? (0f0, 1f0) : (-1f0, 1f0)
        lo = Float32(quantile(fin, mode === :lin ? 0.01 : 0.05))
        hi = Float32(quantile(fin, 0.995))
        lo == hi && (lo = prevfloat(lo); hi = nextfloat(hi))
        return (lo, hi)
    end

    function _global_clims(mode::Symbol)
        if mode === :lin
            fin = Float32[]
            @inbounds for v in cube
                (isfinite(v) && v != Float32(Healpix.UNSEEN)) && push!(fin, v)
            end
            isempty(fin) && return (0f0, 1f0)
            return (Float32(quantile(fin, 0.01)), Float32(quantile(fin, 0.995)))
        else
            f = mode === :log10 ? log10 : log
            fin = Float32[]
            @inbounds for v in cube
                (isfinite(v) && v != Float32(Healpix.UNSEEN) && v > 0) && push!(fin, Float32(f(v)))
            end
            isempty(fin) && return (-1f0, 1f0)
            return (Float32(quantile(fin, 0.05)), Float32(quantile(fin, 0.995)))
        end
    end
    clims_auto = lift(scale_mode, show_moment, moment_order) do m_, show_mom, ord
        show_mom ? _vector_clims(moment_vector(ord), m_) : _global_clims(m_)
    end

    if vmin !== nothing && vmax !== nothing
        a, b = Float32(vmin), Float32(vmax)
        a == b && (a = prevfloat(a); b = nextfloat(b))
        clims_manual[] = (a, b); use_manual[] = true
    end
    clims_obs = lift(use_manual, clims_auto, clims_manual) do um, ca, cm
        um ? cm : ca
    end
    clims_safe = lift(clims_obs) do (lo, hi)
        (isfinite(lo) && isfinite(hi) && lo != hi) ? (lo, hi) : (0f0, 1f0)
    end

    contour_auto_levels = lift(img_disp) do im
        automatic_contour_levels(im; n = 7)
    end
    contour_use_manual = Observable(false)
    contour_manual_levels = Observable(Float32[])
    contour_manual_colors = Observable(String[])
    contour_levels_obs = lift(contour_use_manual, contour_manual_levels, contour_auto_levels) do use_man, manual, auto
        use_man && !isempty(manual) ? manual : auto
    end
    contour_default_color = RGBAf(0, 0, 0, 0.62)
    contour_colors_obs = lift(contour_levels_obs, contour_use_manual, contour_manual_colors) do levels, use_man, colors
        contour_color_values(use_man ? colors : String[], length(levels), contour_default_color)
    end
    show_contours = Observable(false)

    hist_mode_obs = Observable(normalize_histogram_mode(hist_mode))
    hist_bins_obs = Observable(clamp(hist_bins, 4, 512))
    hist_xlimits_manual = Observable(hist_xlimits !== nothing)
    hist_xlimits_manual_value = Observable(hist_xlimits === nothing ?
        (0f0, 1f0) :
        parse_histogram_xlimits(string(first(hist_xlimits)), string(last(hist_xlimits)))[3])
    hist_ylimits_manual = Observable(hist_ylimits !== nothing)
    hist_ylimits_manual_value = Observable(hist_ylimits === nothing ?
        (0f0, 1f0) :
        parse_histogram_ylimits(string(first(hist_ylimits)), string(last(hist_ylimits)))[3])
    hist_limits_obs = lift(hist_xlimits_manual, hist_xlimits_manual_value, clims_safe) do manual, xlim, clim
        manual ? xlim : clim
    end
    hist_pair_obs = lift(img_disp, hist_limits_obs, hist_bins_obs, hist_mode_obs) do im, lim, bins, mode
        histogram_profile(im; bins = bins, limits = lim, mode = mode)
    end
    hist_x_obs = lift(p -> p.x, hist_pair_obs)
    hist_y_obs = lift(p -> p.y, hist_pair_obs)
    hist_width_obs = lift(p -> p.width, hist_pair_obs)
    hist_bars_visible = lift(m -> m === :bars, hist_mode_obs)
    hist_kde_visible = lift(m -> m === :kde, hist_mode_obs)
    hist_ylabel_obs = lift(histogram_ylabel, hist_mode_obs)

    zoom_drag_active = Observable(false)
    zoom_drag_start  = Observable(Point2f(NaN32, NaN32))
    zoom_drag_end    = Observable(Point2f(NaN32, NaN32))
    show_graticule   = Observable(true)
    selection_mode = Observable(:point)
    region_shape = Observable(:box)
    region_drag_active = Observable(false)
    region_start = Observable(Point2f(NaN32, NaN32))
    region_end = Observable(Point2f(NaN32, NaN32))
    region_ipix = Observable(Int[])

    # Pixel sélectionné (initial : centre)
    sel_ipix  = Observable(0)
    sel_xy    = Observable(Point2f(NaN32, NaN32))
    sel_label = Observable(latexstring("\\text{click on map to select a pixel}"))

    spec_y_obs = Observable(zeros(Float32, nv))
    spec_ylimits_value = Observable(spec_ylimits === nothing ?
        (use_manual[] ? clims_manual[] : (0f0, 1f0)) :
        parse_spectrum_ylimits(string(first(spec_ylimits)), string(last(spec_ylimits)))[3])
    spec_ylimits_source = Observable(spec_ylimits === nothing ? (use_manual[] ? :contrast : :auto) : :manual)
    function update_spectrum!(ip::Int)
        if 1 ≤ ip ≤ npix
            region_ipix[] = Int[]
            sel_ipix[] = ip
            spec_y_obs[] = Float32.(@view cube[ip, :])
            θ, φ = Healpix.pix2angRing(res, ip)
            l_deg = rad2deg(φ); b_deg = 90 - rad2deg(θ)
            sel_label[] = latexstring(
                "\\text{pixel ", ip, "}\\;(l, b) = (",
                string(round(mod(l_deg, 360); digits=2)), "^\\circ, ",
                string(round(b_deg; digits=2)), "^\\circ)")
        end
    end

    function update_region_spectrum!(ipixels)
        ips = Int.(ipixels)
        region_ipix[] = ips
        spec_y_obs[] = healpix_region_mean_spectrum(cube, ips, nv)
        shape = region_shape[] === :circle ? "circle" : "box"
        j = clamp(chan_idx[], 1, nv)
        mean_val = healpix_region_mean(@view(cube[:, j]), ips)
        valstr = isfinite(mean_val) ? string(round(mean_val; digits=4)) : "NaN"
        sel_label[] = latexstring(
            "\\text{mean spectrum in ", shape, " region}\\;N=", length(ips),
            "\\;\\text{channel mean}=", valstr,
            "\\;\\mathrm{", latex_safe(data_unit), "}"
        )
    end

    ui_theme = default_ui_theme()
    ui_accent = ui_theme.accent
    ui_selection = ui_theme.selection
    ui_text_muted = ui_theme.text_muted

    # ---------- Figure ----------
    pick_backend!(activate_gl)
    fig_size = _pick_fig_size(figsize)
    # Layout responsive (cf. _view_healpix_map / CubeView). Sans ces seuils,
    # les Fixed() écrasent la carte ou poussent les contrôles hors fenêtre
    # quand l'utilisateur ouvre la vue PPV avec figsize ≤ 1500×950.
    compact_layout = fig_size[1] <= 1500 || fig_size[2] <= 950
    cbar_h_px      = compact_layout ? 38  : 52
    spec_h_px      = compact_layout ? 130 : 165
    hist_h_px      = compact_layout ? 70  : 100
    font_sz        = compact_layout ? 13  : 15
    fig = Figure(size = fig_size, backgroundcolor = ui_theme.background)
    main_grid = fig[1, 1] = GridLayout()

    # Carte
    map_grid = main_grid[1, 1] = GridLayout()
    colgap!(map_grid, -8)
    rowgap!(map_grid, -8)
    is_channel_axis = (vunit_eff == "channel")
    title_obs = lift(chan_idx, show_moment, moment_order) do j, show_mom, ord
        if show_mom
            return latexstring("\\text{", latex_safe(fname), "}\\;\\text{", latex_safe(moment_long_label(ord)), "}")
        end
        v = v0_eff + (j-1)*dv_eff
        if is_channel_axis
            latexstring("\\text{", latex_safe(fname), "}\\;\\text{ch}=", j)
        else
            latexstring("\\text{", latex_safe(fname), "}\\;\\text{ch}=", j,
                        ",\\;v=", string(round(v; digits=2)), "\\,\\mathrm{",
                        latex_safe(vunit_eff), "}")
        end
    end
    ax_img = Axis(map_grid[1, 1];
        title = title_obs,
        aspect = DataAspect(),
        xticksvisible = false, yticksvisible = false,
        xticklabelsvisible = false, yticklabelsvisible = false,
        bottomspinevisible = false, topspinevisible = false,
        leftspinevisible   = false, rightspinevisible = false)

    xs = LinRange(-2f0, 2f0, nx)
    ys = LinRange(-1f0, 1f0, ny)
    img_for_plot = lift(img_disp) do im; permutedims(im); end
    hm = heatmap!(ax_img, xs, ys, img_for_plot;
                  colormap=cm_obs, colorrange=clims_safe, nan_color=:white)
    contour!(ax_img, xs, ys, img_for_plot;
             levels=contour_levels_obs, color=contour_colors_obs, linewidth=1.1,
             visible=show_contours)
    full_map_bounds = (-2.0, 2.0, -1.0, 1.0)
    set_mollweide_view!(ax_img, full_map_bounds...)
    graticule = draw_mollweide_graticule!(ax_img)
    refresh_graticule_labels!(graticule, ax_img; bounds=full_map_bounds)

    # ellipse + zoom box + marker
    ell_x = [2cos(t) for t in LinRange(0, 2π, 200)]
    ell_y = [sin(t)  for t in LinRange(0, 2π, 200)]
    lines!(ax_img, ell_x, ell_y; color=:black, linewidth=0.8)

    zoom_box_segments = lift(zoom_drag_active, zoom_drag_start, zoom_drag_end) do active, p0, p1
        active || return Point2f[]
        (isfinite(p0[1]) && isfinite(p1[1])) || return Point2f[]
        x0,y0 = p0; x1,y1 = p1
        Point2f[Point2f(x0,y0),Point2f(x1,y0),Point2f(x1,y0),Point2f(x1,y1),
                Point2f(x1,y1),Point2f(x0,y1),Point2f(x0,y1),Point2f(x0,y0)]
    end
    linesegments!(ax_img, zoom_box_segments; color=(ui_selection,0.95),
                  linewidth=2.0, linestyle=:dash)
    region_segments = lift(region_start, region_end, region_shape, region_ipix, region_drag_active) do p0, p1, shape, ipixs, dragging
        (dragging || !isempty(ipixs)) ? projected_region_segments(p0, p1, shape) : Point2f[]
    end
    lines!(ax_img, region_segments; color=(ui_selection, 0.98), linewidth=2.3)
    marker_pts = lift(sel_xy) do p
        (isfinite(p[1]) && isfinite(p[2])) ? Point2f[p] : Point2f[]
    end
    scatter!(ax_img, marker_pts; color=ui_accent, markersize=12, marker=:cross)

    map_unit_label = lift(show_moment, moment_order) do show_mom, ord
        show_mom ? latexstring("\\text{", latex_safe(moment_label(ord)), "}") : data_unit_tex
    end
    Colorbar(
        map_grid[2, 1],
        hm;
        label=map_unit_label,
        vertical=false,
        height=compact_layout ? 14 : 18,
        tellwidth=false,
        halign=:center,
    )
    rowsize!(map_grid, 1, Relative(1))
    rowsize!(map_grid, 2, Fixed(cbar_h_px))

    # Spectre
    # Affiché dans le même espace que la carte (lin/log10/ln) → cohérence
    # avec la colorbar : le spectre est mis à l'échelle, et les bornes
    # `clims_manual` (entrées par l'utilisateur dans le même espace
    # transformé) lui sont appliquées en y-limits.
    spec_y_disp = lift(spec_y_obs, scale_mode) do y, m_
        out = apply_scale(y, m_)
        out2 = similar(out, Float32)
        @inbounds for k in eachindex(out)
            x = out[k]; out2[k] = isfinite(x) ? Float32(x) : NaN32
        end
        out2
    end
    ax_spec = Axis(main_grid[2, 1];
        title  = sel_label,
        xlabel = is_channel_axis ?
            L"\text{channel}" :
            latexstring("v\\;[\\mathrm{", latex_safe(vunit_eff), "}]"),
        ylabel = lift(m_ -> m_ === :lin   ? data_unit_tex :
                            m_ === :log10 ? latexstring("\\log_{10}\\,\\text{", latex_safe(data_unit), "}") :
                                            latexstring("\\ln\\,\\text{", latex_safe(data_unit), "}"), scale_mode))
    lines!(ax_spec, spec_x, spec_y_disp; color=:black, linewidth=1.5)
    # ligne verticale à v(chan_idx)
    chan_v = lift(chan_idx) do j; Float32(v0_eff + (j-1)*dv_eff); end
    vlines!(ax_spec, lift(v -> [v], chan_v); color=ui_accent, linewidth=1.2, linestyle=:dash)

    # ylimits du spectre : manuel, hérité du contraste initial, ou auto.
    function _refresh_spec_ylim!()
        if spec_ylimits_source[] === :manual || spec_ylimits_source[] === :contrast
            lo, hi = spec_ylimits_value[]
            ylims!(ax_spec, Float32(lo), Float32(hi))
        else
            ys = spec_y_disp[]
            fin = filter(isfinite, ys)
            if isempty(fin)
                autolimits!(ax_spec)
            else
                lo = Float32(minimum(fin)); hi = Float32(maximum(fin))
                lo == hi && (lo = prevfloat(lo); hi = nextfloat(hi))
                ylims!(ax_spec, lo, hi)
            end
        end
        xlims!(ax_spec, Float32(spec_x[1]), Float32(spec_x[end]))
    end

    function _refresh_hist_axes!()
        xlo, xhi = hist_limits_obs[]
        if hist_ylimits_manual[]
            ylo, yhi = hist_ylimits_manual_value[]
            limits!(ax_hist, Float32(xlo), Float32(xhi), Float32(ylo), Float32(yhi))
        else
            autolimits!(ax_hist)
            xlims!(ax_hist, Float32(xlo), Float32(xhi))
        end
    end

    ax_hist = Axis(
        main_grid[3, 1];
        title = L"\text{Visible channel histogram}",
        xlabel = data_unit_tex,
        ylabel = hist_ylabel_obs,
        # height is governed by `rowsize!(main_grid, 3, ...)` below — no
        # hard-coded value here (cf. CLAUDE.md / anti-patterns).
        xtickformat = _latex_tick_formatter,
        ytickformat = _latex_tick_formatter,
    )
    barplot!(ax_hist, hist_x_obs, hist_y_obs; width=hist_width_obs, color=(ui_accent, 0.44), strokecolor=ui_accent, strokewidth=0.3, visible=hist_bars_visible)
    lines!(ax_hist, hist_x_obs, hist_y_obs; color=ui_accent, linewidth=1.8, visible=hist_kde_visible)
    vlines!(ax_hist, lift(lim -> [first(lim), last(lim)], clims_safe);
            color=(ui_text_muted, 0.65), linewidth=1.0, linestyle=:dash)

    # Contrôles (card-based modal layout, mirrors CubeView / HealpixMapView)
    ctrl_row_h   = compact_layout ? (36, 168, 130) : (42, 190, 150)
    ctrl_gap     = compact_layout ? 6 : 10
    ctrl_total_h = sum(ctrl_row_h) + 2 * ctrl_gap
    card_pad     = compact_layout ? 9 : 12
    card_gap     = compact_layout ? 7 : 10

    ctrl_grid = main_grid[4, 1] = GridLayout(; alignmode = Outside())

    # -- Visibility helpers (verbatim from CubeView) --
    set_block_visible!(block, visible::Bool) = begin
        try; block.visible[] = visible;            catch; end
        try; block.scene.visible[] = visible;      catch; end
        try; block.blockscene.visible[] = visible; catch; end
        nothing
    end
    function set_layout_contents_visible!(layout, visible::Bool)
        for block in try; contents(layout); catch; Any[]; end
            set_block_visible!(block, visible)
            block isa GridLayout && set_layout_contents_visible!(block, visible)
        end
        nothing
    end

    # -- Card factory (mirrors CubeView / HealpixMapView) --
    function control_card!(parent, row, col, title::AbstractString; rows::Int = 4, cols::Int = 4)
        card = parent[row, col] = GridLayout(;
            alignmode = Outside(card_pad), tellwidth = false, tellheight = false)
        body_rows = rows + 1
        Box(card[1:body_rows, 1:cols]; color = ui_theme.panel, strokecolor = ui_theme.border,
            strokewidth = 1.0, cornerradius = 8, z = -6)
        Box(card[1, 1:cols]; color = ui_theme.panel_header, strokecolor = (:transparent, 0.0),
            strokewidth = 0.0, cornerradius = 8, z = -5)
        Label(card[1, 1:cols]; text = uppercase(title), halign = :left, tellwidth = false,
            fontsize = 13, font = :bold, color = ui_theme.accent_strong, padding = (10, 10, 6, 6))
        Box(card[body_rows, 1:cols]; color = :transparent, strokewidth = 0, z = -7)
        rowsize!(card, body_rows, Fixed(compact_layout ? 10 : 12))
        rowgap!(card, card_gap)
        colgap!(card, card_gap)
        return card
    end
    ctrl_lbl!(layout, pos, txt) = Label(layout[pos...]; text = txt, halign = :left,
        tellwidth = false, fontsize = font_sz, color = ui_theme.text_muted)

    # -- Mode bar (row 1, full width) --
    mode_bar = ctrl_grid[1, 1:3] = GridLayout(; alignmode = Outside(0))
    colgap!(mode_bar, compact_layout ? 6 : 10)
    mode_nav_btn      = Button(mode_bar[1, 1]; label = "Navigation", width = 130, height = 32)
    mode_analysis_btn = Button(mode_bar[1, 2]; label = "Analysis",   width = 112, height = 32)
    mode_export_btn   = Button(mode_bar[1, 3]; label = "Export",     width = 96,  height = 32)
    help_btn          = Button(mode_bar[1, 4]; label = "Help",       width = 74,  height = 32)
    foreach(c -> colsize!(mode_bar, c, Auto()), 1:4)
    foreach(w -> manta_style_button!(w, ui_theme; compact = compact_layout),
            (mode_nav_btn, mode_analysis_btn, mode_export_btn, help_btn))
    control_mode = Observable(:navigation)

    # ---- NAVIGATION cards ----

    # channel_card [nav col 1] — Channel slider + Scale
    channel_card = control_card!(ctrl_grid, 2, 1, "Channel"; rows = 4, cols = 6)
    ctrl_lbl!(channel_card, (2, 1), "Channel")
    chan_slider = Slider(channel_card[2, 2:5]; range = 1:nv, startvalue = chan_idx[])
    manta_style_slider!(chan_slider, ui_theme; compact = compact_layout)
    chan_label = Label(channel_card[2, 6];
        text = lift(j -> is_channel_axis ?
            latexstring("j=", j) :
            latexstring("j=", j, ",\\;v=",
                string(round(v0_eff + (j-1)*dv_eff; digits = 2)),
                "\\,\\mathrm{", latex_safe(vunit_eff), "}"), chan_idx),
        fontsize = font_sz, halign = :left, tellwidth = false)
    ctrl_lbl!(channel_card, (3, 1), "Scale")
    scale_menu = Menu(channel_card[3, 2:3]; options = ["lin", "log10", "ln"], prompt = String(scale))
    manta_style_menu!(scale_menu, ui_theme; compact = compact_layout)

    # display_card [nav col 2] — Colormap, Invert, Smoothing
    display_card = control_card!(ctrl_grid, 2, 2, "Display"; rows = 4, cols = 5)
    cmap_menu = Menu(display_card[2, 1:3]; options = ui_colormap_options(), prompt = String(cmap))
    manta_style_menu!(cmap_menu, ui_theme; compact = compact_layout)
    invert_chk = Checkbox(display_card[2, 4])
    invert_chk.checked[] = invert_cmap[]
    manta_style_checkbox!(invert_chk, ui_theme; compact = compact_layout)
    Label(display_card[2, 5]; text = "Invert", halign = :left, tellwidth = false,
        fontsize = font_sz, color = ui_theme.text_muted)
    gauss_chk = Checkbox(display_card[3, 1])
    manta_style_checkbox!(gauss_chk, ui_theme; compact = compact_layout)
    Label(display_card[3, 2]; text = "Smooth", halign = :left, tellwidth = false,
        fontsize = font_sz, color = ui_theme.text_muted)
    sigma_label = Label(display_card[3, 3];
        text = latexstring("\\sigma = 1.5\\,\\text{px}"),
        fontsize = font_sz, halign = :left, tellwidth = false)
    sigma_slider = Slider(display_card[3, 4:5]; range = LinRange(0, 10, 101), startvalue = 1.5)
    manta_style_slider!(sigma_slider, ui_theme; compact = compact_layout)

    # nav_view_card [nav col 3] — Graticule, Reset zoom
    nav_view_card = control_card!(ctrl_grid, 2, 3, "View"; rows = 3, cols = 4)
    graticule_chk = Checkbox(nav_view_card[2, 1])
    graticule_chk.checked[] = show_graticule[]
    manta_style_checkbox!(graticule_chk, ui_theme; compact = compact_layout)
    Label(nav_view_card[2, 2]; text = "Graticule", halign = :left, tellwidth = false,
        fontsize = font_sz, color = ui_theme.text_muted)
    reset_btn = Button(nav_view_card[2, 3:4]; label = "Reset zoom", height = 32)
    manta_style_button!(reset_btn, ui_theme; compact = compact_layout)

    # ---- ANALYSIS cards ----

    # contrast_card [analysis col 1] — Contrast controls
    contrast_card = control_card!(ctrl_grid, 2, 1, "Contrast"; rows = 4, cols = 5)
    clim_min_box = Textbox(contrast_card[2, 1:2]; placeholder = "min", height = 30)
    manta_style_textbox!(clim_min_box, ui_theme; compact = compact_layout)
    clim_max_box = Textbox(contrast_card[2, 3:4]; placeholder = "max", height = 30)
    manta_style_textbox!(clim_max_box, ui_theme; compact = compact_layout)
    apply_btn = Button(contrast_card[2, 5]; label = "Apply", height = 30)
    manta_style_button!(apply_btn, ui_theme; compact = compact_layout)
    auto_btn = Button(contrast_card[3, 1:2]; label = "Auto", height = 30)
    manta_style_button!(auto_btn, ui_theme; compact = compact_layout)
    p1_btn   = Button(contrast_card[3, 3]; label = "p1-p99", height = 30)
    manta_style_button!(p1_btn, ui_theme; compact = compact_layout)
    p5_btn   = Button(contrast_card[3, 4:5]; label = "p5-p95", height = 30)
    manta_style_button!(p5_btn, ui_theme; compact = compact_layout)

    # selection_card [analysis col 2] — Selection mode + Spectrum y-limits
    selection_card = control_card!(ctrl_grid, 2, 2, "Selection"; rows = 4, cols = 5)
    region_mode_menu = Menu(selection_card[2, 1:3]; options = ["point", "box", "circle"], prompt = "point")
    manta_style_menu!(region_mode_menu, ui_theme; compact = compact_layout)
    region_clear_btn = Button(selection_card[2, 4:5]; label = "Clear", height = 30)
    manta_style_button!(region_clear_btn, ui_theme; compact = compact_layout)
    region_count_label = Label(selection_card[3, 1:3]; text = "0 pix", halign = :left,
        tellwidth = false, fontsize = font_sz, color = ui_theme.text_muted)
    ctrl_lbl!(selection_card, (4, 1), "Spec y")
    spec_ymin_box = Textbox(selection_card[4, 2]; placeholder = "y min", height = 30)
    manta_style_textbox!(spec_ymin_box, ui_theme; compact = compact_layout)
    spec_ymax_box = Textbox(selection_card[4, 3]; placeholder = "y max", height = 30)
    manta_style_textbox!(spec_ymax_box, ui_theme; compact = compact_layout)
    spec_y_apply_btn = Button(selection_card[4, 4]; label = "Apply y", height = 30)
    manta_style_button!(spec_y_apply_btn, ui_theme; compact = compact_layout)
    spec_y_auto_btn  = Button(selection_card[4, 5]; label = "Auto y", height = 30)
    manta_style_button!(spec_y_auto_btn, ui_theme; compact = compact_layout)

    # contour_card [analysis col 3] — Contours
    contour_card = control_card!(ctrl_grid, 2, 3, "Contours"; rows = 3, cols = 5)
    contour_chk = Checkbox(contour_card[2, 1])
    contour_chk.checked[] = show_contours[]
    manta_style_checkbox!(contour_chk, ui_theme; compact = compact_layout)
    Label(contour_card[2, 2]; text = "Show", halign = :left, tellwidth = false,
        fontsize = font_sz, color = ui_theme.text_muted)
    contour_levels_box = Textbox(contour_card[2, 3:4]; placeholder = "auto or 1:red, 2:#00ffaa", height = 30)
    manta_style_textbox!(contour_levels_box, ui_theme; compact = compact_layout)
    contour_apply_btn = Button(contour_card[2, 5]; label = "Apply", height = 30)
    manta_style_button!(contour_apply_btn, ui_theme; compact = compact_layout)

    # ---- ANALYSIS bottom: Moment + Histogram ----
    analysis_bottom = ctrl_grid[3, 1:3] = GridLayout(; alignmode = Outside(0))
    colgap!(analysis_bottom, ctrl_gap)

    moment_card = control_card!(analysis_bottom, 1, 1, "Moment"; rows = 3, cols = 6)
    moment_menu = Menu(moment_card[2, 1:3];
        options = ["M0 integrated", "M1 mean", "M2 dispersion"], prompt = "M0 integrated")
    manta_style_menu!(moment_menu, ui_theme; compact = compact_layout)
    show_moment_btn = Button(moment_card[2, 4]; label = "Show", height = 30)
    manta_style_button!(show_moment_btn, ui_theme; compact = compact_layout)
    show_channel_btn = Button(moment_card[2, 5:6]; label = "Channel", height = 30)
    manta_style_button!(show_channel_btn, ui_theme; compact = compact_layout)
    save_moment_fits_btn = Button(moment_card[3, 1:4]; label = "Save moment FITS", height = 30)
    manta_style_button!(save_moment_fits_btn, ui_theme; compact = compact_layout)

    hist_card = control_card!(analysis_bottom, 1, 2, "Histogram"; rows = 3, cols = 7)
    hist_mode_menu = Menu(hist_card[2, 1]; options = ["bars", "kde"], prompt = String(hist_mode_obs[]))
    manta_style_menu!(hist_mode_menu, ui_theme; compact = compact_layout)
    hist_bins_box = Textbox(hist_card[2, 2]; placeholder = "bins", height = 30)
    manta_style_textbox!(hist_bins_box, ui_theme; compact = compact_layout)
    hist_xmin_box = Textbox(hist_card[2, 3]; placeholder = "x min", height = 30)
    manta_style_textbox!(hist_xmin_box, ui_theme; compact = compact_layout)
    hist_xmax_box = Textbox(hist_card[2, 4]; placeholder = "x max", height = 30)
    manta_style_textbox!(hist_xmax_box, ui_theme; compact = compact_layout)
    hist_apply_btn = Button(hist_card[2, 5]; label = "Apply x", height = 30)
    manta_style_button!(hist_apply_btn, ui_theme; compact = compact_layout)
    hist_auto_btn  = Button(hist_card[2, 6:7]; label = "Auto x", height = 30)
    manta_style_button!(hist_auto_btn, ui_theme; compact = compact_layout)
    hist_ymin_box = Textbox(hist_card[3, 1:2]; placeholder = "y min", height = 30)
    manta_style_textbox!(hist_ymin_box, ui_theme; compact = compact_layout)
    hist_ymax_box = Textbox(hist_card[3, 3:4]; placeholder = "y max", height = 30)
    manta_style_textbox!(hist_ymax_box, ui_theme; compact = compact_layout)
    hist_y_apply_btn = Button(hist_card[3, 5]; label = "Apply y", height = 30)
    manta_style_button!(hist_y_apply_btn, ui_theme; compact = compact_layout)
    hist_y_auto_btn  = Button(hist_card[3, 6:7]; label = "Auto y", height = 30)
    manta_style_button!(hist_y_auto_btn, ui_theme; compact = compact_layout)

    colsize!(analysis_bottom, 1, Relative(0.38))
    colsize!(analysis_bottom, 2, Relative(0.62))

    # ---- EXPORT cards ----
    output_card = control_card!(ctrl_grid, 2, 1, "Output"; rows = 3, cols = 3)
    save_btn = Button(output_card[2, 1:2]; label = "Save PNG", height = 32)
    manta_style_button!(save_btn, ui_theme; compact = compact_layout)

    # Grid sizing
    foreach(c -> colsize!(ctrl_grid, c, Relative(1 / 3)), 1:3)
    rowsize!(ctrl_grid, 1, Fixed(ctrl_row_h[1]))
    rowsize!(ctrl_grid, 2, Fixed(ctrl_row_h[2]))
    rowsize!(ctrl_grid, 3, Fixed(ctrl_row_h[3]))
    colgap!(ctrl_grid, ctrl_gap)
    rowgap!(ctrl_grid, ctrl_gap)

    # main_grid row sizing
    rowsize!(main_grid, 1, Relative(1))
    rowsize!(main_grid, 2, Fixed(spec_h_px))
    rowsize!(main_grid, 3, Fixed(hist_h_px))
    rowsize!(main_grid, 4, Fixed(ctrl_total_h))

    # ---- Mode switching ----
    nav_cards_hpc      = (channel_card, display_card, nav_view_card)
    analysis_cards_hpc = (contrast_card, selection_card, contour_card, analysis_bottom)
    export_cards_hpc   = (output_card,)

    function set_mode_button_active!(btn, active::Bool)
        btn.buttoncolor[]       = active ? ui_theme.accent        : ui_theme.surface
        btn.buttoncolor_hover[] = active ? ui_theme.accent_strong : ui_theme.surface_hover
        btn.labelcolor[]        = active ? :white                 : ui_theme.text
        btn.labelcolor_hover[]  = active ? :white                 : ui_theme.accent_strong
        nothing
    end
    function refresh_control_mode!()
        mode = control_mode[]
        is_nav = (mode === :navigation)
        is_ana = (mode === :analysis)
        is_exp = (mode === :export)
        for c in nav_cards_hpc;      set_layout_contents_visible!(c, is_nav); end
        for c in analysis_cards_hpc; set_layout_contents_visible!(c, is_ana); end
        for c in export_cards_hpc;   set_layout_contents_visible!(c, is_exp); end
        set_mode_button_active!(mode_nav_btn,      is_nav)
        set_mode_button_active!(mode_analysis_btn, is_ana)
        set_mode_button_active!(mode_export_btn,   is_exp)
        nothing
    end
    refresh_control_mode!()
    on_mode(btn, sym) = on(btn.clicks) do _
        control_mode[] = sym
        refresh_control_mode!()
    end
    on_mode(mode_nav_btn,      :navigation)
    on_mode(mode_analysis_btn, :analysis)
    on_mode(mode_export_btn,   :export)

    if use_manual[]
        a, b = clims_manual[]
        sa, sb = string(a), string(b)
        clim_min_box.displayed_string[] = sa; clim_min_box.stored_string[] = sa
        clim_max_box.displayed_string[] = sb; clim_max_box.stored_string[] = sb
    end

    set_box_text!(tb, s::AbstractString) = begin
        str = String(s)
        tb.displayed_string[] = str
        tb.stored_string[] = str
        nothing
    end
    set_box_text!(hist_bins_box, string(hist_bins_obs[]))
    if hist_xlimits_manual[]
        lo, hi = hist_xlimits_manual_value[]
        set_box_text!(hist_xmin_box, string(lo))
        set_box_text!(hist_xmax_box, string(hi))
    end
    if hist_ylimits_manual[]
        lo, hi = hist_ylimits_manual_value[]
        set_box_text!(hist_ymin_box, string(lo))
        set_box_text!(hist_ymax_box, string(hi))
    end
    if spec_ylimits_source[] !== :auto
        lo, hi = spec_ylimits_value[]
        set_box_text!(spec_ymin_box, string(lo))
        set_box_text!(spec_ymax_box, string(hi))
    end
    function clear_region!()
        region_ipix[] = Int[]
        region_start[] = Point2f(NaN32, NaN32)
        region_end[] = Point2f(NaN32, NaN32)
        region_drag_active[] = false
        region_count_label.text[] = "0 pix"
        nothing
    end
    function apply_region!(p0::Point2f, p1::Point2f)
        ips = projected_region_ipix(ipix_grid, p0[1], p0[2], p1[1], p1[2], region_shape[])
        region_count_label.text[] = "$(length(ips)) pix"
        update_region_spectrum!(ips)
        _refresh_spec_ylim!()
        nothing
    end
    function apply_percentile_clims!(lo::Real, hi::Real)
        clims = percentile_clims(img_disp[], lo, hi)
        clims_manual[] = clims
        use_manual[] = true
        set_box_text!(clim_min_box, string(first(clims)))
        set_box_text!(clim_max_box, string(last(clims)))
        if spec_ylimits_source[] === :contrast
            spec_ylimits_value[] = clims
            set_box_text!(spec_ymin_box, string(first(clims)))
            set_box_text!(spec_ymax_box, string(last(clims)))
            _refresh_spec_ylim!()
        end
        nothing
    end

    # ---------- Reactivity ----------
    on(chan_slider.value) do v
        chan_idx[] = Int(round(v))
        if !isempty(region_ipix[])
            update_region_spectrum!(region_ipix[])
            _refresh_spec_ylim!()
        end
    end
    on(scale_menu.selection) do sel
        sel === nothing && return
        new_mode = Symbol(sel)
        new_mode === scale_mode[] && return
        # Les clims_manual étaient exprimées dans l'ancien espace (lin/log10/ln).
        # Les invalider et vider les textboxes pour repartir en auto dans le
        # nouvel espace — sinon le spectre et la colorbar restent bloqués sur
        # des bornes incohérentes.
        if use_manual[]
            use_manual[] = false
        end
        clim_min_box.displayed_string[] = ""; clim_min_box.stored_string[] = ""
        clim_max_box.displayed_string[] = ""; clim_max_box.stored_string[] = ""
        scale_mode[] = new_mode
    end
    on(cmap_menu.selection) do sel
        sel === nothing && return
        cmap_name[] = Symbol(sel)
    end
    on(invert_chk.checked) do v; invert_cmap[] = v; end
    on(gauss_chk.checked) do v
        gauss_on[] = v
    end
    on(sigma_slider.value) do v
        sigma[] = Float32(v)
        sigma_label.text[] = latexstring("\\sigma = $(round(v; digits=2))\\,\\text{px}")
    end
    on(graticule_chk.checked) do v
        show_graticule[] = v
        set_graticule_visible!(graticule, v)
    end
    on(reset_btn.clicks) do _
        set_mollweide_view!(ax_img, full_map_bounds...)
        refresh_graticule_labels!(graticule, ax_img; bounds=full_map_bounds)
    end
    on(apply_btn.clicks) do _
        ok, manual, clims, _msg = parse_manual_clims(
            get_box_str(clim_min_box), get_box_str(clim_max_box);
            fallback = clims_manual[])
        ok || return
        if manual
            clims_manual[] = clims
            use_manual[]   = true
            if spec_ylimits_source[] === :contrast
                spec_ylimits_value[] = clims
                set_box_text!(spec_ymin_box, string(first(clims)))
                set_box_text!(spec_ymax_box, string(last(clims)))
            end
        else
            use_manual[]   = false
            if spec_ylimits_source[] === :contrast
                spec_ylimits_source[] = :auto
                set_box_text!(spec_ymin_box, "")
                set_box_text!(spec_ymax_box, "")
            end
        end
        _refresh_spec_ylim!()                # propage au spectre
    end
    on(auto_btn.clicks) do _
        use_manual[] = false
        set_box_text!(clim_min_box, "")
        set_box_text!(clim_max_box, "")
        if spec_ylimits_source[] === :contrast
            spec_ylimits_source[] = :auto
            set_box_text!(spec_ymin_box, "")
            set_box_text!(spec_ymax_box, "")
        end
        _refresh_spec_ylim!()
    end
    on(p1_btn.clicks) do _; apply_percentile_clims!(1, 99); end
    on(p5_btn.clicks) do _; apply_percentile_clims!(5, 95); end
    on(hist_mode_menu.selection) do sel
        sel === nothing && return
        hist_mode_obs[] = normalize_histogram_mode(sel)
    end
    on(hist_apply_btn.clicks) do _
        ok_bins, bins, _bins_msg = parse_histogram_bins(get_box_str(hist_bins_box); fallback = hist_bins_obs[])
        ok_x, manual_x, xlim, _x_msg = parse_histogram_xlimits(
            get_box_str(hist_xmin_box),
            get_box_str(hist_xmax_box);
            fallback = hist_xlimits_manual_value[],
        )
        ok_bins && ok_x || return
        hist_bins_obs[] = bins
        hist_xlimits_manual_value[] = xlim
        hist_xlimits_manual[] = manual_x
        set_box_text!(hist_bins_box, string(bins))
        set_box_text!(hist_xmin_box, manual_x ? string(first(xlim)) : "")
        set_box_text!(hist_xmax_box, manual_x ? string(last(xlim)) : "")
        _refresh_hist_axes!()
    end
    on(hist_auto_btn.clicks) do _
        hist_xlimits_manual[] = false
        set_box_text!(hist_xmin_box, "")
        set_box_text!(hist_xmax_box, "")
        _refresh_hist_axes!()
    end
    on(hist_y_auto_btn.clicks) do _
        hist_ylimits_manual[] = false
        set_box_text!(hist_ymin_box, "")
        set_box_text!(hist_ymax_box, "")
        _refresh_hist_axes!()
    end
    on(hist_y_apply_btn.clicks) do _
        ok_y, manual_y, ylim, _msg = parse_histogram_ylimits(
            get_box_str(hist_ymin_box),
            get_box_str(hist_ymax_box);
            fallback = hist_ylimits_manual_value[],
        )
        ok_y || return
        hist_ylimits_manual_value[] = ylim
        hist_ylimits_manual[] = manual_y
        set_box_text!(hist_ymin_box, manual_y ? string(first(ylim)) : "")
        set_box_text!(hist_ymax_box, manual_y ? string(last(ylim)) : "")
        _refresh_hist_axes!()
    end
    on(spec_y_apply_btn.clicks) do _
        ok, manual, ylim, _msg = parse_spectrum_ylimits(
            get_box_str(spec_ymin_box),
            get_box_str(spec_ymax_box);
            fallback = spec_ylimits_value[],
        )
        ok || return
        if manual
            spec_ylimits_value[] = ylim
            spec_ylimits_source[] = :manual
            set_box_text!(spec_ymin_box, string(first(ylim)))
            set_box_text!(spec_ymax_box, string(last(ylim)))
        else
            spec_ylimits_source[] = :auto
            set_box_text!(spec_ymin_box, "")
            set_box_text!(spec_ymax_box, "")
        end
        _refresh_spec_ylim!()
    end
    on(spec_y_auto_btn.clicks) do _
        spec_ylimits_source[] = :auto
        set_box_text!(spec_ymin_box, "")
        set_box_text!(spec_ymax_box, "")
        _refresh_spec_ylim!()
    end
    on(hist_limits_obs) do _
        _refresh_hist_axes!()
    end
    on(hist_y_obs) do _
        _refresh_hist_axes!()
    end
    on(region_mode_menu.selection) do sel
        sel === nothing && return
        mode = Symbol(String(sel))
        mode in (:point, :box, :circle) || return
        selection_mode[] = mode
        region_shape[] = mode === :circle ? :circle : :box
        if mode === :point
            clear_region!()
            sel_ipix[] > 0 && update_spectrum!(sel_ipix[])
            _refresh_spec_ylim!()
        end
    end
    on(region_clear_btn.clicks) do _
        clear_region!()
        sel_ipix[] > 0 && update_spectrum!(sel_ipix[])
        _refresh_spec_ylim!()
    end
    on(contour_chk.checked) do v
        show_contours[] = v
    end
    on(contour_apply_btn.clicks) do _
        ok, use_man, levels, colors, _msg = parse_contour_specs(
            get_box_str(contour_levels_box);
            fallback_levels=contour_manual_levels[],
            fallback_colors=contour_manual_colors[],
        )
        ok || return
        contour_use_manual[] = use_man
        contour_manual_levels[] = levels
        contour_manual_colors[] = colors
        set_box_text!(contour_levels_box, use_man ? format_contour_specs(levels, colors) : "")
        show_contours[] = true
        contour_chk.checked[] = true
    end
    on(moment_menu.selection) do sel
        sel === nothing && return
        label = String(sel)
        moment_order[] = startswith(label, "M1") ? 1 : startswith(label, "M2") ? 2 : 0
    end
    on(show_moment_btn.clicks) do _
        show_moment[] = true
        use_manual[] = false
        set_box_text!(clim_min_box, "")
        set_box_text!(clim_max_box, "")
    end
    on(show_channel_btn.clicks) do _
        show_moment[] = false
        use_manual[] = false
        set_box_text!(clim_min_box, "")
        set_box_text!(clim_max_box, "")
    end
    on(scale_mode)        do _; _refresh_spec_ylim!(); end
    on(spec_y_disp)       do _; _refresh_spec_ylim!(); end
    on(use_manual)        do _; _refresh_spec_ylim!(); end
    on(clims_manual)      do clims
        if spec_ylimits_source[] === :contrast
            spec_ylimits_value[] = clims
        end
        _refresh_spec_ylim!()
    end

    # zoom right-drag + click left → select pixel
    on(events(ax_img).mousebutton) do ev
        if ev.button == Mouse.right && ev.action == Mouse.press
            p = mouseposition(ax_img); any(isnan, p) && return
            zoom_drag_start[] = Point2f(p[1], p[2])
            zoom_drag_end[]   = Point2f(p[1], p[2])
            zoom_drag_active[] = true
        elseif ev.button == Mouse.right && ev.action == Mouse.release
            zoom_drag_active[] || return
            p = mouseposition(ax_img); !any(isnan, p) && (zoom_drag_end[] = Point2f(p[1], p[2]))
            p0 = zoom_drag_start[]; p1 = zoom_drag_end[]
            zoom_drag_active[] = false
            zoom_drag_start[] = Point2f(NaN32, NaN32); zoom_drag_end[] = Point2f(NaN32, NaN32)
            (isfinite(p0[1]) && isfinite(p1[1])) || return
            xmin,xmax = minmax(p0[1], p1[1]); ymin,ymax = minmax(p0[2], p1[2])
            (abs(xmax-xmin) < 1e-3 || abs(ymax-ymin) < 1e-3) && return
            zoom_bounds = (Float64(xmin), Float64(xmax), Float64(ymin), Float64(ymax))
            set_mollweide_view!(ax_img, zoom_bounds...)
            refresh_graticule_labels!(graticule, ax_img; bounds=zoom_bounds)
        elseif ev.button == Mouse.left && ev.action == Mouse.press && selection_mode[] != :point
            p = mouseposition(ax_img); any(isnan, p) && return
            mollweide_xy_to_lonlat(p[1], p[2]) === nothing && return
            region_start[] = Point2f(p[1], p[2])
            region_end[] = Point2f(p[1], p[2])
            region_drag_active[] = true
            region_ipix[] = Int[]
            region_count_label.text[] = "0 pix"
        elseif ev.button == Mouse.left && ev.action == Mouse.release && region_drag_active[]
            p = mouseposition(ax_img)
            !any(isnan, p) && (region_end[] = Point2f(p[1], p[2]))
            p0 = region_start[]; p1 = region_end[]
            region_drag_active[] = false
            if isfinite(p0[1]) && isfinite(p1[1])
                apply_region!(p0, p1)
            else
                clear_region!()
            end
        elseif ev.button == Mouse.left && ev.action == Mouse.press
            p = mouseposition(ax_img); any(isnan, p) && return
            ll = mollweide_xy_to_lonlat(p[1], p[2]); ll === nothing && return
            l_deg, b_deg = ll
            θhp = deg2rad(90 - b_deg); φhp = deg2rad(mod(l_deg, 360))
            ip = Healpix.ang2pixRing(res, θhp, φhp)
            sel_xy[] = Point2f(p[1], p[2])
            clear_region!()
            update_spectrum!(ip)
        end
    end
    on(events(ax_img).mouseposition) do p
        if zoom_drag_active[] && !any(isnan, p)
            zoom_drag_end[] = Point2f(p[1], p[2])
        elseif region_drag_active[] && !any(isnan, p)
            region_end[] = Point2f(p[1], p[2])
        end
    end

    # save PNG
    save_root = save_dir === nothing ? begin
        d = joinpath(homedir(), "Desktop"); isdir(d) ? d : pwd()
    end : (isdir(save_dir) ? String(save_dir) : (mkpath(save_dir); String(save_dir)))
    on(save_btn.clicks) do _
        out = joinpath(save_root, "$(fname)_ch$(chan_idx[]).png")
        try CairoMakie.save(String(out), fig; backend=CairoMakie); @info "Saved" out
        catch e; @error "Failed to save" exception=(e, catch_backtrace()) end
    end
    on(save_moment_fits_btn.clicks) do _
        label = replace(moment_label(moment_order[]), " " => "")
        out = joinpath(save_root, "$(fname)_$(label)_healpix.fits")
        try
            FITS(String(out), "w") do f
                write(f, Float32.(moment_vector(moment_order[])))
            end
            @info "Saved moment FITS" out
        catch e
            @error "Failed to save moment FITS" exception=(e, catch_backtrace())
        end
    end

    # init
    update_spectrum!(max(1, npix ÷ 2))     # spectre par défaut au pixel central
    _refresh_spec_ylim!()
    _refresh_hist_axes!()

    # Espacement vertical : éloigne la ligne de contrôles des xticks du
    # spectre pour éviter le chevauchement (ex: "j=41, v=80km/s" qui se
    # superposait au tick "80").
    try
        rowgap!(main_grid, 2, 22)
        rowgap!(main_grid, 1, 6)
    catch
        # rowgap! échoue si l'index est hors limites — silencieux.
    end

    # ---------- Keyboard shortcuts (HEALPix PPV cube) ----------
    _set_status_hpc!(msg::AbstractString) =
        (sel_label[] = latexstring("\\text{", latex_safe(msg), "}"); nothing)
    _trigger_btn_hpc!(btn) = (btn.clicks[] = btn.clicks[] + 1)
    function _set_channel_hpc!(n::Integer)
        n_clamped = clamp(Int(n), 1, nv)
        n_clamped == chan_idx[] && return
        chan_slider.value[] = n_clamped
        _set_status_hpc!("Channel $(n_clamped) / $(nv).")
    end
    function _toggle_contours_hpc!()
        new_val = !show_contours[]
        contour_chk.checked[] = new_val
        _set_status_hpc!(new_val ? "Contours enabled." : "Contours hidden.")
    end
    function _cycle_log_scale_hpc!()
        next = scale_mode[] === :lin   ? :log10 :
               scale_mode[] === :log10 ? :ln    : :lin
        scale_menu.selection[] = String(next)
        _set_status_hpc!("Image scale: $(String(next)).")
    end
    shortcuts_hpc = ShortcutBinding[
        ShortcutBinding(Keyboard.i,         () -> (invert_cmap[] = !invert_cmap[]);
                        description = "invert cmap"),
        ShortcutBinding(Keyboard.page_up,   () -> _set_channel_hpc!(chan_idx[] - 1);
                        description = "prev channel"),
        ShortcutBinding(Keyboard.page_down, () -> _set_channel_hpc!(chan_idx[] + 1);
                        description = "next channel"),
        ShortcutBinding(Keyboard.home,      () -> _set_channel_hpc!(1);
                        description = "first channel"),
        ShortcutBinding(Keyboard.a,         () -> _trigger_btn_hpc!(auto_btn);
                        description = "auto contrast"),
        ShortcutBinding(Keyboard._1,        () -> apply_percentile_clims!(1, 99);
                        description = "p1-p99"),
        ShortcutBinding(Keyboard._5,        () -> apply_percentile_clims!(5, 95);
                        description = "p5-p95"),
        ShortcutBinding(Keyboard.r,         () -> _trigger_btn_hpc!(reset_btn);
                        description = "reset zoom"),
        ShortcutBinding(Keyboard.s,         () -> _trigger_btn_hpc!(save_btn);
                        description = "save image"),
        ShortcutBinding(Keyboard.c,         () -> _toggle_contours_hpc!();
                        description = "contours"),
        ShortcutBinding(Keyboard.l,         () -> _cycle_log_scale_hpc!();
                        description = "cycle scale"),
    ]
    # Help: Shift+/ and the Help button open a dedicated Makie figure
    # listing every documented binding; status bar keeps the one-liner.
    function _open_help_hpc!()
        try
            open_shortcut_help_window(shortcuts_hpc;
                title = "MANTA — HEALPix cube shortcuts", theme = ui_theme)
        catch e
            @warn "Could not open shortcut help window" exception = (e, catch_backtrace())
        end
        _set_status_hpc!(shortcut_help_message(shortcuts_hpc))
    end
    push!(shortcuts_hpc,
          ShortcutBinding(Keyboard.slash,
                          _open_help_hpc!;
                          description = "this help",
                          modifier = :shift))
    on(help_btn.clicks) do _
        _open_help_hpc!()
    end
    register_shortcuts!(fig, shortcuts_hpc;
        textboxes = (clim_min_box, clim_max_box, contour_levels_box,
                     hist_bins_box, hist_xmin_box, hist_xmax_box,
                     hist_ymin_box, hist_ymax_box,
                     spec_ymin_box, spec_ymax_box),
        is_blocked = () -> zoom_drag_active[] || region_drag_active[],
    )

    keepalive!(fig)
    on(fig.scene.events.window_open) do is_open
        is_open || forget!(fig)
    end
    display_fig && display(fig)
    return fig
end
