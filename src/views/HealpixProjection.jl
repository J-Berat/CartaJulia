# path: src/views/HealpixProjection.jl
#
# HEALPix Mollweide projection, region, and graticule helpers shared by the
# file-backed compatibility viewers and dataset-backed HEALPix views.

############################
# Projection Mollweide
############################

"""
    mollweide_grid(map; nx=1200, ny=600) -> Matrix{Float32}

Reproject `map::HealpixMap` onto a `(ny, nx)` Mollweide grid centred on
`(l, b) = (0, 0)`. Pixels outside the ellipse → `NaN32`. Convention:
- `x ∈ [-2, 2]`, `y ∈ [-1, 1]`
- longitude increases to the left following the astronomical convention;
  same orientation as `Healpix.jl` (lon = φ HEALPix).
"""
function mollweide_grid(m::Healpix.HealpixMap; nx::Int=1200, ny::Int=600)
    return mollweide_apply_index(mollweide_pixel_index(m.resolution, nx, ny), m)
end

"""
    mollweide_apply_index(idx, m::HealpixMap) -> Matrix{Float32}

Gather scalar HEALPix values onto a precomputed Mollweide pixel-index grid
(see [`mollweide_pixel_index`](@ref)). Pixels with `idx == 0` are NaN32.
`Healpix.UNSEEN` and non-finite values are also mapped to NaN32. Use this
when reprojecting many frames sharing the same `(nside, nx, ny)`: the
projection is done once and only the cheap gather step runs per frame.
"""
function mollweide_apply_index(idx::AbstractMatrix{<:Integer}, m::Healpix.HealpixMap)
    ny, nx = size(idx)
    img = fill(NaN32, ny, nx)
    pix = m.pixels
    npix = length(pix)
    @inbounds for j in 1:ny, i in 1:nx
        ip = Int(idx[j, i])
        (ip < 1 || ip > npix) && continue
        v = pix[ip]
        img[j, i] = (isfinite(v) && v != Healpix.UNSEEN) ? Float32(v) : NaN32
    end
    img
end

"""
    mollweide_color_grid(pixels; nx=1200, ny=600) -> Matrix{RGBAf}

Reproject a HEALPix RGB/RGBA pixel vector on a Mollweide grid. Pixels outside
the projection ellipse are transparent.
"""
function mollweide_color_grid(pixels::AbstractVector{<:Colorant}; nx::Int=1200, ny::Int=600)
    nside = valid_healpix_npix(length(pixels))
    nside > 0 || throw(ArgumentError("RGB HEALPix vector length must be 12*nside^2."))
    res = Healpix.Resolution(nside)
    return mollweide_apply_index_color(mollweide_pixel_index(res, nx, ny), pixels)
end

"""
    mollweide_apply_index_color(idx, pixels) -> Matrix{RGBAf}

Gather colorant HEALPix values onto a precomputed Mollweide pixel-index
grid. Out-of-ellipse cells (`idx == 0`) are fully transparent. Companion
to [`mollweide_apply_index`](@ref) — see its docstring for the projection
caching motivation.
"""
function mollweide_apply_index_color(idx::AbstractMatrix{<:Integer},
                                     pixels::AbstractVector{<:Colorant})
    ny, nx = size(idx)
    img = fill(RGBAf(1, 1, 1, 0), ny, nx)
    npix = length(pixels)
    @inbounds for j in 1:ny, i in 1:nx
        ip = Int(idx[j, i])
        (ip < 1 || ip > npix) && continue
        img[j, i] = RGBAf(pixels[ip])
    end
    img
end

function _mollweide_scalar_grid(vals::AbstractVector; nx::Int=1200, ny::Int=600)
    nside = valid_healpix_npix(length(vals))
    nside > 0 || throw(ArgumentError("HEALPix vector length must be 12*nside^2."))
    res = Healpix.Resolution(nside)
    return _mollweide_apply_index_scalar(mollweide_pixel_index(res, nx, ny), vals)
end

"""
    _mollweide_apply_index_scalar(idx, vals) -> Matrix{Float32}

Gather scalar HEALPix values (plain vector, not a `HealpixMap`) onto a
precomputed Mollweide pixel-index grid. Out-of-ellipse cells are NaN32,
as are non-finite source values.
"""
function _mollweide_apply_index_scalar(idx::AbstractMatrix{<:Integer},
                                       vals::AbstractVector)
    ny, nx = size(idx)
    img = fill(NaN32, ny, nx)
    npix = length(vals)
    @inbounds for j in 1:ny, i in 1:nx
        ip = Int(idx[j, i])
        (ip < 1 || ip > npix) && continue
        v = Float32(vals[ip])
        img[j, i] = isfinite(v) ? v : NaN32
    end
    img
end

"""
    mollweide_pixel_index(res, nx, ny) -> Matrix{Int32}

Pre-compute the HEALPix (RING) pixel index at every point of the Mollweide
grid. 0 = pixel outside the ellipse. Allows regenerating a new frame in
O(npx) without recomputing the projection.
"""
function mollweide_pixel_index(res::Healpix.Resolution, nx::Int, ny::Int)
    idx = zeros(Int32, ny, nx)
    @inbounds for j in 1:ny, i in 1:nx
        x = 2 * (2 * (i - 0.5) / nx - 1)
        y = 2 * (j - 0.5) / ny - 1
        (x^2 / 4 + y^2 > 1) && continue
        θaux = asin(y)
        sinφ = (2θaux + sin(2θaux)) / π
        abs(sinφ) > 1 && continue
        lat = asin(sinφ)
        lon = π * x / (2 * cos(θaux))
        abs(lon) > π && continue
        θhp = π/2 - lat
        φhp = lon < 0 ? lon + 2π : lon
        idx[j, i] = Int32(Healpix.ang2pixRing(res, θhp, φhp))
    end
    idx
end

function projected_region_segments(p0::Point2f, p1::Point2f, shape::Symbol)
    if !(isfinite(p0[1]) && isfinite(p0[2]) && isfinite(p1[1]) && isfinite(p1[2]))
        return Point2f[]
    end
    x0, y0 = p0
    x1, y1 = p1
    if shape === :circle
        r = hypot(x1 - x0, y1 - y0)
        r < 1f-5 && return Point2f[]
        return Point2f[Point2f(x0 + r * cos(t), y0 + r * sin(t)) for t in LinRange(0, 2π, 97)]
    else
        return Point2f[
            Point2f(x0, y0), Point2f(x1, y0),
            Point2f(x1, y1), Point2f(x0, y1),
            Point2f(x0, y0),
        ]
    end
end

function projected_region_ipix(
    ipix_grid::AbstractMatrix{<:Integer},
    x0::Real,
    y0::Real,
    x1::Real,
    y1::Real,
    shape::Symbol,
)
    ny, nx = size(ipix_grid)
    if nx < 1 || ny < 1
        return Int[]
    end
    xmin, xmax = minmax(Float64(x0), Float64(x1))
    ymin, ymax = minmax(Float64(y0), Float64(y1))
    if shape !== :circle && (abs(xmax - xmin) < 1e-5 || abs(ymax - ymin) < 1e-5)
        return Int[]
    end
    if shape === :circle
        cx, cy = Float64(x0), Float64(y0)
        r = hypot(Float64(x1) - cx, Float64(y1) - cy)
        r < 1e-5 && return Int[]
        xmin, xmax = cx - r, cx + r
        ymin, ymax = cy - r, cy + r
        rr = r * r
        inside = (x, y) -> (x - cx)^2 + (y - cy)^2 <= rr
    else
        inside = (x, y) -> xmin <= x <= xmax && ymin <= y <= ymax
    end
    ix0 = clamp(Int(floor((xmin + 2.0) / 4.0 * nx + 1)), 1, nx)
    ix1 = clamp(Int(ceil((xmax + 2.0) / 4.0 * nx + 1)), 1, nx)
    iy0 = clamp(Int(floor((ymin + 1.0) / 2.0 * ny + 1)), 1, ny)
    iy1 = clamp(Int(ceil((ymax + 1.0) / 2.0 * ny + 1)), 1, ny)
    seen = Set{Int}()
    @inbounds for j in iy0:iy1, i in ix0:ix1
        x = 2 * (2 * (i - 0.5) / nx - 1)
        y = 2 * (j - 0.5) / ny - 1
        inside(x, y) || continue
        ip = Int(ipix_grid[j, i])
        ip > 0 && push!(seen, ip)
    end
    return sort!(collect(seen))
end

function healpix_region_mean(vals, ipixels)
    isempty(ipixels) && return Float32(NaN)
    acc = 0.0
    cnt = 0
    @inbounds for ip in ipixels
        if 1 <= ip <= length(vals)
            v = vals[ip]
            fv = Float32(v)
            if isfinite(fv) && fv != Float32(Healpix.UNSEEN)
                acc += Float64(fv)
                cnt += 1
            end
        end
    end
    return cnt == 0 ? Float32(NaN) : Float32(acc / cnt)
end

function healpix_region_mean_spectrum(cube::AbstractMatrix, ipixels, nv::Int)
    y = fill(Float32(NaN), nv)
    isempty(ipixels) && return y
    @inbounds for j in 1:nv
        acc = 0.0
        cnt = 0
        for ip in ipixels
            if 1 <= ip <= size(cube, 1)
                v = cube[ip, j]
                fv = Float32(v)
                if isfinite(fv) && fv != Float32(Healpix.UNSEEN)
                    acc += Float64(fv)
                    cnt += 1
                end
            end
        end
        y[j] = cnt == 0 ? Float32(NaN) : Float32(acc / cnt)
    end
    return y
end

@inline function mollweide_lonlat_to_xy(lon_deg::Real, lat_deg::Real)
    lat = deg2rad(clamp(Float64(lat_deg), -90.0, 90.0))
    lon = deg2rad(clamp(Float64(lon_deg), -180.0, 180.0))
    target = π * sin(lat)

    θaux = lat
    for _ in 1:14
        f = 2θaux + sin(2θaux) - target
        fp = 2 + 2cos(2θaux)
        abs(fp) < 1e-12 && break
        θaux = clamp(θaux - f / fp, -π/2, π/2)
    end

    x = 2 * lon * cos(θaux) / π
    y = sin(θaux)
    (x^2 / 4 + y^2 > 1 + 1e-5) && return nothing
    return Point2f(Float32(x), Float32(y))
end

_angle_label(v::Real) = begin
    r = round(Int, v)
    r == 0 ? "0°" : "$(r)°"
end

_latex_tick(v::Real) = begin
    x = abs(Float64(v)) < 1e-10 ? 0.0 : Float64(v)
    r = round(x)
    s = abs(x - r) < 1e-8 ? string(Int(r)) : string(round(x; digits=2))
    latexstring("\\mathrm{", latex_safe(s), "}")
end
_latex_tick_formatter(vals) = [_latex_tick(v) for v in vals]

function draw_mollweide_graticule!(
    ax;
    lon_values = -150:30:150,
    lat_values = -60:30:60,
    line_color = RGBAf(1, 1, 1, 0.30),
    label_color = RGBAf(0, 0, 0, 0.85),
    linewidth::Real = 0.9,
    fontsize::Real = 12,
)
    line_plots = Any[]
    for lon in lon_values
        pts = Point2f[]
        for lat in LinRange(-89.0, 89.0, 240)
            p = mollweide_lonlat_to_xy(lon, lat)
            p === nothing || push!(pts, p)
        end
        length(pts) > 1 && push!(
            line_plots,
            lines!(ax, pts; color=line_color, linewidth=linewidth, linestyle=:dot),
        )
    end

    for lat in lat_values
        pts = Point2f[]
        for lon in LinRange(-180.0, 180.0, 360)
            p = mollweide_lonlat_to_xy(lon, lat)
            p === nothing || push!(pts, p)
        end
        length(pts) > 1 && push!(
            line_plots,
            lines!(ax, pts; color=line_color, linewidth=linewidth, linestyle=:dot),
        )
    end

    lon_bottom_pos = Observable(Point2f[])
    lon_bottom_txt = Observable(String[])
    lon_top_pos = Observable(Point2f[])
    lon_top_txt = Observable(String[])
    lat_left_pos = Observable(Point2f[])
    lat_left_txt = Observable(String[])
    lat_right_pos = Observable(Point2f[])
    lat_right_txt = Observable(String[])

    label_plots = Any[
        text!(ax, lon_bottom_pos; text=lon_bottom_txt, color=label_color,
              fontsize=fontsize, align=(:center, :top)),
        text!(ax, lon_top_pos; text=lon_top_txt, color=label_color,
              fontsize=fontsize, align=(:center, :bottom)),
        text!(ax, lat_left_pos; text=lat_left_txt, color=label_color,
              fontsize=fontsize, align=(:right, :center)),
        text!(ax, lat_right_pos; text=lat_right_txt, color=label_color,
              fontsize=fontsize, align=(:left, :center)),
    ]

    graticule = (
        lines = line_plots,
        labels = label_plots,
        lon_values = collect(lon_values),
        lat_values = collect(lat_values),
        lon_bottom_pos = lon_bottom_pos,
        lon_bottom_txt = lon_bottom_txt,
        lon_top_pos = lon_top_pos,
        lon_top_txt = lon_top_txt,
        lat_left_pos = lat_left_pos,
        lat_left_txt = lat_left_txt,
        lat_right_pos = lat_right_pos,
        lat_right_txt = lat_right_txt,
    )
    refresh_graticule_labels!(graticule, ax)
    return graticule
end

function _expanded_graticule_bounds(xlo, xhi, ylo, yhi)
    dx = max(Float64(xhi) - Float64(xlo), 1e-6)
    dy = max(Float64(yhi) - Float64(ylo), 1e-6)
    xpad = 0.075 * dx
    ypad = 0.080 * dy
    return (Float64(xlo) - xpad, Float64(xhi) + xpad,
            Float64(ylo) - ypad, Float64(yhi) + ypad)
end

function set_mollweide_view!(ax, xlo, xhi, ylo, yhi)
    xmin, xmax, ymin, ymax = _expanded_graticule_bounds(xlo, xhi, ylo, yhi)
    limits!(ax, xmin, xmax, ymin, ymax)
end

function _axis_bounds(ax)
    r = ax.finallimits[]
    xlo = Float64(r.origin[1])
    ylo = Float64(r.origin[2])
    return (xlo, xlo + Float64(r.widths[1]), ylo, ylo + Float64(r.widths[2]))
end

function _visible_meridian_points(lon, xlo, xhi, ylo, yhi)
    pts = Point2f[]
    for lat in LinRange(-89.0, 89.0, 360)
        p = mollweide_lonlat_to_xy(lon, lat)
        if p !== nothing && xlo ≤ p[1] ≤ xhi && ylo ≤ p[2] ≤ yhi
            push!(pts, p)
        end
    end
    return pts
end

function _visible_parallel_points(lat, xlo, xhi, ylo, yhi)
    pts = Point2f[]
    for lon in LinRange(-180.0, 180.0, 720)
        p = mollweide_lonlat_to_xy(lon, lat)
        if p !== nothing && xlo ≤ p[1] ≤ xhi && ylo ≤ p[2] ≤ yhi
            push!(pts, p)
        end
    end
    return pts
end

function refresh_graticule_labels!(graticule, ax; bounds = nothing)
    xlo, xhi, ylo, yhi = bounds === nothing ? _axis_bounds(ax) : bounds
    dx = max(xhi - xlo, 1e-6)
    dy = max(yhi - ylo, 1e-6)
    xoff = Float32(0.025 * dx)
    yoff = Float32(0.030 * dy)

    lon_bottom_pos = Point2f[]
    lon_bottom_txt = String[]
    lon_top_pos = Point2f[]
    lon_top_txt = String[]
    for lon in graticule.lon_values
        pts = _visible_meridian_points(lon, xlo, xhi, ylo, yhi)
        isempty(pts) && continue
        pbot = pts[argmin(getindex.(pts, 2))]
        ptop = pts[argmax(getindex.(pts, 2))]
        label = _angle_label(lon)
        push!(lon_bottom_pos, Point2f(pbot[1], pbot[2] - yoff))
        push!(lon_bottom_txt, label)
        push!(lon_top_pos, Point2f(ptop[1], ptop[2] + yoff))
        push!(lon_top_txt, label)
    end

    lat_left_pos = Point2f[]
    lat_left_txt = String[]
    lat_right_pos = Point2f[]
    lat_right_txt = String[]
    for lat in graticule.lat_values
        pts = _visible_parallel_points(lat, xlo, xhi, ylo, yhi)
        isempty(pts) && continue
        pleft = pts[argmin(getindex.(pts, 1))]
        pright = pts[argmax(getindex.(pts, 1))]
        label = _angle_label(lat)
        push!(lat_left_pos, Point2f(pleft[1] - xoff, pleft[2]))
        push!(lat_left_txt, label)
        push!(lat_right_pos, Point2f(pright[1] + xoff, pright[2]))
        push!(lat_right_txt, label)
    end

    graticule.lon_bottom_pos[] = lon_bottom_pos
    graticule.lon_bottom_txt[] = lon_bottom_txt
    graticule.lon_top_pos[] = lon_top_pos
    graticule.lon_top_txt[] = lon_top_txt
    graticule.lat_left_pos[] = lat_left_pos
    graticule.lat_left_txt[] = lat_left_txt
    graticule.lat_right_pos[] = lat_right_pos
    graticule.lat_right_txt[] = lat_right_txt
    return graticule
end

function set_graticule_visible!(graticule, visible::Bool)
    foreach(p -> p.visible[] = visible, graticule.lines)
    foreach(p -> p.visible[] = visible, graticule.labels)
    return graticule
end
