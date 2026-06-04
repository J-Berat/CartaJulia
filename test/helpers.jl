# path: test/helpers.jl
# Testsets covering Helpers.jl: scaling, mapping, products, latex, io, UI,
# validation, WCS, settings, power spectrum, slice utilities, path parsing,
# FITS export headers, keyboard shortcuts, shortcut window, structured errors,
# progress/cancellation, downsampling, undo/redo, plugins, backend selection.

@testset "helpers: scaling" begin
    A = Float32.([1, 10, 100, 0, -1])
    lin = MANTA.apply_scale(A, :lin)
    log10v = MANTA.apply_scale(A, :log10)
    lnv = MANTA.apply_scale(A, :ln)

    @test eltype(lin) == Float32
    @test eltype(log10v) == Float32
    @test eltype(lnv) == Float32

    @test lin[1:3] == A[1:3]
    @test isapprox(log10v[1], 0f0; atol=1e-6)
    @test isapprox(log10v[2], 1f0; atol=1e-6)
    @test isfinite(lnv[1])
    @test !isfinite(lnv[4]) && !isfinite(lnv[5])

    mn, mx = MANTA.clamped_extrema(Float32.([1, 2, 3]))
    @test mn == 1f0 && mx == 3f0

    mn2, mx2 = MANTA.clamped_extrema(Float32.([5, 5, 5]))
    @test mn2 < 5.0f0 && mx2 > 5.0f0

    mn3, mx3 = MANTA.clamped_extrema(Float32.([NaN32, NaN32]))
    @test mn3 == 0f0 && mx3 == 1f0

    mn4, mx4 = MANTA.clamped_extrema(Float32.([]))
    @test mn4 == 0f0 && mx4 == 1f0

    p1, p99 = MANTA.percentile_clims(Float32.(1:100), 1, 99)
    @test p1 >= 1f0 && p99 <= 100f0 && p1 < p99

    hx, hy = MANTA.histogram_counts(Float32.(1:10); bins = 5)
    @test length(hx) == 5
    @test length(hy) == 5
    @test sum(hy) == 10f0
    hp = MANTA.histogram_profile(Float32.(1:10); bins = 5, mode = :bars)
    @test length(hp.x) == 5
    @test hp.mode === :bars
    @test sum(hp.y) == 10f0
    kde = MANTA.histogram_profile(Float32.(1:10); bins = 5, mode = :kde)
    @test kde.mode === :kde
    @test length(kde.y) == 5
    @test all(isfinite, kde.y)
    ok_bins, bins, _ = MANTA.parse_histogram_bins("128"; fallback = 64)
    @test ok_bins && bins == 128
    ok_bins2, bins2, _ = MANTA.parse_histogram_bins("9999"; fallback = 64)
    @test ok_bins2 && bins2 == 512
    ok_x, manual_x, xlim, _ = MANTA.parse_histogram_xlimits("10", "1")
    @test ok_x && manual_x && xlim == (1f0, 10f0)
    ok_x_auto, manual_x_auto, _, _ = MANTA.parse_histogram_xlimits("", "")
    @test ok_x_auto && !manual_x_auto
    ok_hy, manual_hy, hylim, _ = MANTA.parse_histogram_ylimits("42", "10")
    @test ok_hy && manual_hy && hylim == (10f0, 42f0)
    ok_sy_auto, manual_sy_auto, _, _ = MANTA.parse_spectrum_ylimits("", "")
    @test ok_sy_auto && !manual_sy_auto

    recipe = MANTA.manta_recipe("cube.fits";
        cmap = :magma,
        invert = true,
        state = Dict("axis" => 3, "index" => 12),
    )
    @test occursin("MANTA.manta(\"cube.fits\";", recipe)
    @test occursin("cmap = :magma", recipe)
    @test occursin("\"axis\" => 3", recipe)

    smoothed = MANTA.nan_gaussian_filter(Float32[NaN 1 1; NaN 1 1; NaN NaN NaN], 1.0)
    @test size(smoothed) == (3, 3)
    @test isfinite(smoothed[2, 2])
    @test isnan(MANTA.nan_gaussian_filter(fill(NaN32, 3, 3), 1.0)[2, 2])

    levels = MANTA.automatic_contour_levels(Float32.(1:100); n = 6)
    @test length(levels) == 6
    @test issorted(levels)
end

@testset "helpers: asinh/sqrt stretches" begin
    A = Float32.([1, 10, 100, 0, -1])

    # asinh keeps the sign and is defined on all of ℝ (0 and negatives finite),
    # unlike log/ln. Default softening a = 1 → asinh(x).
    as = MANTA.apply_scale(A, :asinh)
    @test eltype(as) == Float32
    @test all(isfinite, as)
    @test isapprox(as[1], Float32(asinh(1.0));   atol = 1e-6)
    @test isapprox(as[2], Float32(asinh(10.0));  atol = 1e-5)
    @test as[4] == 0f0
    @test as[5] < 0f0 && isapprox(as[5], -as[1]; atol = 1e-6)

    # Softening: asinh(x / a). Larger a → smaller magnitude for the same x.
    as2 = MANTA.apply_scale(A, :asinh; asinh_softening = 2.0)
    @test isapprox(as2[2], Float32(asinh(10.0 / 2.0)); atol = 1e-5)
    @test abs(as2[2]) < abs(as[2])
    # Non-positive / non-finite softening collapses to the default (a = 1).
    @test MANTA.apply_scale(A, :asinh; asinh_softening = -3.0) == as
    @test MANTA.apply_scale(A, :asinh; asinh_softening = NaN) == as

    # sqrt mirrors the log/ln rule: values ≤ 0 become NaN (0 included).
    sq = MANTA.apply_scale(A, :sqrt)
    @test eltype(sq) == Float32
    @test isapprox(sq[2], Float32(sqrt(10.0)); atol = 1e-5)
    @test sq[3] == 10f0
    @test isnan(sq[4]) && isnan(sq[5])

    # apply_scale_display never returns non-finite (NaN → 0) for both modes.
    @test all(isfinite, MANTA.apply_scale_display(A, :sqrt))
    @test all(isfinite, MANTA.apply_scale_display(A, :asinh; asinh_softening = 3.0))

    # Centralised mode metadata (single source of truth for menus & cycling).
    @test MANTA.scale_menu_options() == ["lin", "log10", "ln", "asinh", "sqrt"]
    @test MANTA.cycle_scale_mode(:lin)   === :log10
    @test MANTA.cycle_scale_mode(:ln)    === :asinh
    @test MANTA.cycle_scale_mode(:asinh) === :sqrt
    @test MANTA.cycle_scale_mode(:sqrt)  === :lin       # wraps around
    @test MANTA.cycle_scale_mode(:bogus) === :lin       # unknown restarts cycle

    # LaTeX label helper decorates the unit with the active stretch.
    @test occursin("sqrt",  String(MANTA.scale_label_tex(:sqrt, "Jy")))
    @test occursin("asinh", String(MANTA.scale_label_tex(:asinh, "Jy")))
    @test occursin("log",   String(MANTA.scale_label_tex(:log10, "Jy")))
    @test occursin("Jy",    String(MANTA.scale_label_tex(:lin, "Jy")))
end

@testset "helpers: wcs sky graticule (pure math)" begin
    # CAR projection: lon = lon0 + (i-1)·CDELT1, lat = lat0 + (j-1)·CDELT2.
    hdr_car = Dict{String,Any}(
        "CTYPE1" => "GLON-CAR", "CUNIT1" => "deg",
        "CRVAL1" => 30.0, "CRPIX1" => 1.0, "CDELT1" => 1.0,
        "CTYPE2" => "GLAT-CAR", "CUNIT2" => "deg",
        "CRVAL2" => 0.0,  "CRPIX2" => 1.0, "CDELT2" => 1.0,
    )
    wt = MANTA.read_wcs_transform(hdr_car, 2)
    lon, lat = MANTA.wcs_sky_grids(wt, 4, 3)
    @test size(lon) == (4, 3) && size(lat) == (4, 3)
    @test all(isfinite, lon) && all(isfinite, lat)
    @test isapprox(lon[1, 1], 30.0; atol = 1e-9)
    @test isapprox(lon[4, 1], 33.0; atol = 1e-9)   # depends on i only
    @test isapprox(lat[1, 1], 0.0;  atol = 1e-9)
    @test isapprox(lat[1, 3], 2.0;  atol = 1e-9)   # depends on j only

    lon_levels, lat_levels = MANTA.wcs_graticule_levels(lon, lat; n = 6)
    @test !isempty(lon_levels) && !isempty(lat_levels)
    @test issorted(lon_levels) && issorted(lat_levels)
    @test all(l -> 30.0 - 1e-9 <= l <= 33.0 + 1e-9, lon_levels)
    @test all(l -> 0.0  - 1e-9 <= l <= 2.0  + 1e-9, lat_levels)

    # "nice" step selection and its degenerate guards.
    @test MANTA.nice_graticule_step(3.0, 6) == 0.5
    @test MANTA.nice_graticule_step(0.0) == 1.0
    @test MANTA.nice_graticule_step(NaN) == 1.0

    # No sky pair → all-NaN grids, empty levels, and a no-op overlay handle.
    lon0, lat0 = MANTA.wcs_sky_grids(MANTA.SimpleWCSAxis[], 3, 3)
    @test all(isnan, lon0) && all(isnan, lat0)
    @test MANTA.wcs_graticule_levels(lon0, lat0) == (Float64[], Float64[])
    # draw_wcs_graticule! never touches `ax` when there are no levels, so the
    # no-op path is safe to exercise headlessly with a dummy axis argument.
    grat = MANTA.draw_wcs_graticule!(nothing, MANTA.SimpleWCSAxis[], 3, 3)
    @test isempty(grat.plots)
    @test MANTA.set_wcs_graticule_visible!(grat, true) === grat   # no error
end

@testset "helpers: startup defaults" begin
    # Typed coercers apply a value when present & well-typed, else keep current.
    d = Dict{Symbol,Any}(
        :cmap => "magma", :invert => true, :scale => "asinh",
        :asinh_softening => 2.5, :hist_bins => 256, :nx => 1600,
        :figsize => [1400, 900], :save_dir => "/tmp/out",
    )
    @test MANTA._manta_default_symbol(d, :cmap, :viridis) === :magma
    @test MANTA._manta_default_bool(d, :invert, false) === true
    @test MANTA._manta_default_symbol(d, :scale, :lin) === :asinh
    @test MANTA._manta_default_real(d, :asinh_softening, 1.0) == 2.5
    @test MANTA._manta_default_int(d, :hist_bins, 64) === 256
    @test MANTA._manta_default_figsize(d, :figsize, nothing) === (1400, 900)
    @test MANTA._manta_default_string(d, :save_dir, nothing) == "/tmp/out"
    # Absent key → current is returned unchanged.
    @test MANTA._manta_default_symbol(d, :missing, :keep) === :keep
    @test MANTA._manta_default_int(d, :missing, 7) === 7
    # Wrong-typed entry degrades gracefully (keeps current).
    bad = Dict{Symbol,Any}(:hist_bins => "lots", :figsize => [1, 2, 3])
    @test MANTA._manta_default_int(bad, :hist_bins, 64) === 64
    @test MANTA._manta_default_figsize(bad, :figsize, nothing) === nothing

    # Round-trip through a TOML file via the MANTA_DEFAULTS override.
    saved = get(ENV, "MANTA_DEFAULTS", nothing)
    try
        dir = mktempdir()
        path = joinpath(dir, "defaults.toml")
        write(path, "cmap = \"inferno\"\nscale = \"sqrt\"\nhist_bins = 128\n")
        ENV["MANTA_DEFAULTS"] = path
        @test MANTA.manta_defaults_path() == path
        loaded = MANTA.load_manta_defaults()
        @test loaded[:cmap] == "inferno"
        @test loaded[:scale] == "sqrt"
        @test loaded[:hist_bins] == 128
        @test MANTA._manta_default_symbol(loaded, :scale, :lin) === :sqrt

        # Missing file → empty dict (no error).
        ENV["MANTA_DEFAULTS"] = joinpath(dir, "does_not_exist.toml")
        @test isempty(MANTA.load_manta_defaults())

        # Malformed file → empty dict (warns, never fatal).
        bad_path = joinpath(dir, "bad.toml")
        write(bad_path, "this is = = not valid toml [[[")
        ENV["MANTA_DEFAULTS"] = bad_path
        @test isempty(MANTA.load_manta_defaults())
    finally
        if saved === nothing
            delete!(ENV, "MANTA_DEFAULTS")
        else
            ENV["MANTA_DEFAULTS"] = saved
        end
    end
end

@testset "helpers: mapping" begin
    # bijection uv <-> ijk depending on the axis
    for axis in 1:3
        i, j, k = 3, 2, 1
        u, v = MANTA.ijk_to_uv(i, j, k, axis)
        ii, jj, kk = MANTA.uv_to_ijk(u, v, axis, axis == 1 ? i : axis == 2 ? j : k)
        @test (ii, jj, kk) == (i, j, k)
    end

    # get_slice dims and type
    data = Array{Float32}(undef, 7, 5, 4)
    fill!(data, 1f0)
    s1 = MANTA.get_slice(data, 1, 2)
    s2 = MANTA.get_slice(data, 2, 3)
    s3 = MANTA.get_slice(data, 3, 1)
    @test size(s1) == (size(data, 2), size(data, 3))
    @test size(s2) == (size(data, 1), size(data, 3))
    @test size(s3) == (size(data, 1), size(data, 2))
    @test eltype(s1) == Float32 && eltype(s2) == Float32 && eltype(s3) == Float32

    box_uv = MANTA.region_uv_indices(10, 10, 2, 3, 4, 5, :box)
    @test (3, 2) in box_uv
    @test (5, 4) in box_uv

    circle_uv = MANTA.region_uv_indices(10, 10, 5, 5, 7, 5, :circle)
    @test (5, 5) in circle_uv
    @test (5, 7) in circle_uv
    @test (1, 1) ∉ circle_uv

    cube = reshape(Float32.(1:24), 2, 3, 4)
    spec = MANTA.mean_region_spectrum(cube, 3, [(1, 1), (2, 1)])
    @test length(spec) == 4
    @test spec[1] == mean(Float32[cube[1, 1, 1], cube[2, 1, 1]])
end

@testset "helpers: products" begin
    # M0 is now a true integral: Σ y_i Δx_i. For x = [10, 20, 30, 40] the
    # auto-inferred channel width is 10, so the +y entries (2, 3) contribute
    # 2·10 + 3·10 = 50 (in [y · x-units]). M1/M2 are weighted averages so
    # the Δx factor cancels out — they are unchanged vs. the legacy sum form.
    @test MANTA.moments(Float32[-1, 0, 2, 3]; x = Float32[10, 20, 30, 40]) == (50.0, 36.0, sqrt(24.0))
    # Explicit scalar dx = 1 reproduces the legacy "sum" semantics.
    @test MANTA.moments(Float32[-1, 0, 2, 3]; x = Float32[10, 20, 30, 40], dx = 1.0) ==
          (5.0, 36.0, sqrt(24.0))
    @test all(isnan, MANTA.moments(Float32[-1, 0]; x = Float32[1, 2]))

    # Explicit (nsigma, sigma) overrides the MAD estimate: thr = 2·1 = 2.
    # Only y_i > 2 survive → 3 + 4 = 7 (Δx = 1).
    let y = Float32[1, 2, 3, 4], x = Float32[1, 2, 3, 4]
        m0_clip, _, _ = MANTA.moments(y; x = x, nsigma = 2.0, sigma = 1.0, dx = 1.0)
        @test m0_clip == 7.0
    end

    # Robust σ estimator: MAD-based, with the 1.4826 Gaussian factor.
    # For y = [-3, -1, 0, 1, 3]: median = 0, MAD = 1, σ ≈ 1.4826.
    @test isapprox(MANTA._robust_sigma(Float64[-3, -1, 0, 1, 3]), 1.4826; atol = 1e-6)
    @test isnan(MANTA._robust_sigma(Float64[]))

    # Explicit channel window restricts the integration support.
    let y = Float32[10, 10, 10, 10], x = Float32[1, 2, 3, 4]
        m0_win, _, _ = MANTA.moments(y; x = x, channels = 2:3, dx = 1.0)
        @test m0_win == 20.0    # 10 + 10
    end

    a = Float32[2 4; 6 8]
    b = Float32[1 2; 0 4]
    @test MANTA.dual_view_product(a, b, :A) == a
    @test MANTA.dual_view_product(a, b, :B) == b
    @test MANTA.dual_view_product(a, b, :diff) == Float32[1 2; 6 4]
    ratio = MANTA.dual_view_product(a, b, :ratio)
    @test ratio[1, 1] == 2f0
    @test isnan(ratio[2, 1])
    resid = MANTA.dual_view_product(a, b, :residuals)
    @test isapprox(mean(vec(resid)), 0f0; atol = 1f-6)

    small = reshape(Float32[1, 3, 5, 7], 2, 2)
    up = MANTA.resample_matrix_linear(small, (3, 3))
    @test size(up) == (3, 3)
    @test up[1, 1] == small[1, 1]
    @test up[3, 3] == small[2, 2]
    @test up[2, 2] ≈ 4f0

    coarse_cube = reshape(Float32.(1:8), 2, 2, 2)
    fine_cube = MANTA.resample_cube_linear(coarse_cube, (3, 3, 3))
    @test size(fine_cube) == (3, 3, 3)
    @test fine_cube[1, 1, 1] == coarse_cube[1, 1, 1]
    @test fine_cube[3, 3, 3] == coarse_cube[2, 2, 2]
    mixed = MANTA.dual_view_product(zeros(Float32, 3, 3), small, :B)
    @test size(mixed) == (3, 3)
    @test mixed[2, 2] ≈ 4f0

    cube = Array{Float32}(undef, 2, 2, 3)
    cube[:, :, 1] .= 1f0
    cube[:, :, 2] .= 2f0
    cube[:, :, 3] .= 3f0
    m0 = MANTA.moment_map(cube, 3, 0; coords = Float32[10, 20, 30])
    m1 = MANTA.moment_map(cube, 3, 1; coords = Float32[10, 20, 30])
    M0, M1, M2 = MANTA.moments_map(cube, Float32[10, 20, 30])
    # coords have Δv = 10 → M0 is multiplied by Δv vs the legacy sum form.
    @test all(==(60f0), m0)
    @test M0 == m0
    @test M1 == m1
    @test all(isfinite, M2)
    @test all(isapprox.(m1, Ref(Float32((10 + 40 + 90) / 6)); atol = 1f-5))
    # Δv = 1 reproduces legacy sums.
    m0_legacy = MANTA.moment_map(cube, 3, 0; coords = Float32[10, 20, 30], dx = 1.0)
    @test all(==(6f0), m0_legacy)
    mv0, mv1, mv2 = MANTA.moment_vectors(Float32[0 2 3; -1 0 4], Float32[1, 2, 3])
    @test mv0 == Float32[5, 4]
    @test isfinite(mv1[1]) && isfinite(mv2[1])
    @test MANTA.filtered_cube_by_slice(cube, 3, 0) == cube
end

@testset "helpers: latex" begin
    s = MANTA.make_info_tex(1, 2, 3, 4, 5, 6f0)
    t1 = MANTA.make_slice_title("fname", 3, 10)
    t2 = MANTA.make_spec_title(1, 2, 3)

    @test s isa LaTeXString
    @test t1 isa LaTeXString
    @test t2 isa LaTeXString

    # No LaTeX line breaks: forbid "\\ " and "\\\n"
    raw_s  = String(s)
    raw_t1 = String(t1)
    raw_t2 = String(t2)
    for raw in (raw_s, raw_t1, raw_t2)
        @test !occursin("\\\\ ", raw)
        @test !occursin("\\\\\\n", raw)
    end

    # Expect inline LaTeX (e.g., \\, for thin space)
    @test occursin("\\,", raw_s) || occursin("\\,", raw_t1) || occursin("\\,", raw_t2)
    @test occursin("intensity", lowercase(raw_s))
end

@testset "helpers: io" begin
    # to_cmap
    cm = MANTA.to_cmap(:viridis)
    @test length(cm) > 0
    @test cm[1] isa ColorTypes.Colorant
    @test MANTA.to_cmap(:gray) == MANTA.to_cmap(:grayC)
    @test all(name -> length(MANTA.to_cmap(name)) > 0, MANTA.ui_colormap_options())
    @test MANTA.ui_colormap_options() == collect(MANTA.MANTA_COLORMAP_OPTIONS)
    @test all(in(MANTA.ui_colormap_options()), ["viridis", "inferno", "gray", "coolwarm"])

    # get_box_str via mock (no Makie Textbox available)
    struct MockTB
        stored_string::Observable{String}
    end
    tb = MockTB(Observable("   hello world   "))
    @test MANTA.get_box_str(tb) == "hello world"

    struct MockDisplayTB
        displayed_string::Observable{String}
    end
    tb2 = MockDisplayTB(Observable("   fallback value   "))
    @test MANTA.get_box_str(tb2) == "fallback value"
end

@testset "RGB helpers and direct viewers" begin
    r = Float32[-1 0; 1 2]
    g = Float32[0 1; 2 3]
    b = Float32[3 2; 1 0]
    rgb = MANTA.rgb_image(r, g, b)
    @test size(rgb) == (2, 2)
    @test eltype(rgb) <: ColorTypes.Colorant

    stack_last = zeros(Float32, 2, 3, 3)
    stack_last[:, :, 1] .= 1
    img_last = MANTA.as_rgb_image(stack_last)
    @test size(img_last) == (2, 3)

    stack_first = zeros(Float32, 3, 2, 3)
    stack_first[2, :, :] .= 1
    img_first = MANTA.as_rgb_image(stack_first)
    @test size(img_first) == (2, 3)

    hpix_rgb = [RGBf(i / 12, 0.25, 1 - i / 12) for i in 1:12]
    @test length(MANTA.as_rgb_pixels(hpix_rgb)) == 12
    @test length(MANTA.as_rgb_pixels(Float32.(reshape(1:36, 12, 3)) ./ 36)) == 12

    fig_rgb = MANTA.manta(rgb; activate_gl=false, display_fig=false, figsize=(500, 400))
    @test fig_rgb isa Makie.Figure
    MANTA.forget!(fig_rgb)

    fig_panels = MANTA.manta_panels(rgb, r; activate_gl=false, display_fig=false, figsize=(700, 400))
    @test fig_panels isa Makie.Figure
    MANTA.forget!(fig_panels)

    fig_hpix_rgb = MANTA.manta_healpix(hpix_rgb; activate_gl=false, display_fig=false, nx=60, ny=30, figsize=(500, 320))
    @test fig_hpix_rgb isa Makie.Figure
    MANTA.forget!(fig_hpix_rgb)

    fig_hpix_panels = MANTA.manta_healpix_panels(hpix_rgb, Float32.(1:12); activate_gl=false, display_fig=false, nx=60, ny=30, figsize=(700, 320))
    @test fig_hpix_panels isa Makie.Figure
    MANTA.forget!(fig_hpix_panels)
end

@testset "helpers: ui" begin
    # explicit override
    @test MANTA._pick_fig_size((111, 222)) == (111, 222)

    # default when no explicit size is provided: contract is now
    #   - returns a (w, h) tuple of `Int`,
    #   - is at least `MANTA._MIN_FIG_SIZE` on both axes,
    #   - falls back to `MANTA._DEFAULT_FIG_SIZE` when no screen can be
    #     detected (headless CI, no DISPLAY, etc.).
    sz = MANTA._pick_fig_size(nothing)
    @test sz isa Tuple{Int,Int}
    @test sz[1] >= MANTA._MIN_FIG_SIZE[1]
    @test sz[2] >= MANTA._MIN_FIG_SIZE[2]

    # Force-headless code path: clear the cache, blank the env overrides and
    # blank DISPLAY so the GLFW branch is skipped → must yield the default.
    MANTA._SCREEN_SIZE_CACHE[] = nothing
    MANTA._SCREEN_SIZE_PROBED[] = false
    saved_w = get(ENV, "MANTA_SCREEN_W", nothing)
    saved_h = get(ENV, "MANTA_SCREEN_H", nothing)
    saved_display = get(ENV, "DISPLAY", nothing)
    saved_wayland = get(ENV, "WAYLAND_DISPLAY", nothing)
    ENV["MANTA_SCREEN_W"] = ""
    ENV["MANTA_SCREEN_H"] = ""
    if Sys.islinux()
        ENV["DISPLAY"] = ""
        ENV["WAYLAND_DISPLAY"] = ""
    end
    try
        if Sys.islinux()
            @test MANTA._pick_fig_size(nothing) == MANTA._DEFAULT_FIG_SIZE
        end
    finally
        saved_w === nothing ? delete!(ENV, "MANTA_SCREEN_W") : (ENV["MANTA_SCREEN_W"] = saved_w)
        saved_h === nothing ? delete!(ENV, "MANTA_SCREEN_H") : (ENV["MANTA_SCREEN_H"] = saved_h)
        saved_display === nothing ? delete!(ENV, "DISPLAY") : (ENV["DISPLAY"] = saved_display)
        saved_wayland === nothing ? delete!(ENV, "WAYLAND_DISPLAY") : (ENV["WAYLAND_DISPLAY"] = saved_wayland)
        MANTA._SCREEN_SIZE_CACHE[] = nothing
        MANTA._SCREEN_SIZE_PROBED[] = false
    end
end

@testset "helpers: parse_textbox" begin
    # Good parse → (true, value, hint_ok)
    ok, val, msg = MANTA.parse_textbox("  42 "; type = Int, fallback = 0)
    @test ok && val == 42 && isempty(msg)

    # Good parse with explicit hint_ok
    ok2, val2, msg2 = MANTA.parse_textbox("3.14"; type = Float64, fallback = 0.0,
        hint_ok = "applied")
    @test ok2 && isapprox(val2, 3.14; atol = 1e-12) && msg2 == "applied"

    # Empty input → (true, fallback, hint_empty)
    ok3, val3, msg3 = MANTA.parse_textbox(""; type = Int, fallback = 7,
        hint_empty = "unchanged")
    @test ok3 && val3 == 7 && msg3 == "unchanged"

    # Whitespace-only also counts as empty
    ok4, val4, _ = MANTA.parse_textbox("   "; type = Float32, fallback = 1f0)
    @test ok4 && val4 == 1f0

    # Bad parse → (false, fallback, hint_fail)
    ok5, val5, msg5 = MANTA.parse_textbox("nope"; type = Float32, fallback = -1f0,
        hint_fail = "bad")
    @test !ok5 && val5 == -1f0 && msg5 == "bad"

    # Float32 round-trip
    ok6, val6, _ = MANTA.parse_textbox("2.5"; type = Float32, fallback = 0f0)
    @test ok6 && val6 == 2.5f0
end

@testset "ui: manta_flag_textbox!" begin
    # Red-border validity cue. Construct a real Makie Textbox headlessly
    # (no backend activated) and toggle its border palette.
    theme = MANTA.default_ui_theme()
    fig = Makie.Figure()
    tb = Makie.Textbox(fig[1, 1])

    # Rejected input → all three border states painted with the error red.
    ret = MANTA.manta_flag_textbox!(tb, false, theme)
    @test ret === tb
    @test RGBf(tb.bordercolor[])         == theme.error
    @test RGBf(tb.bordercolor_hover[])   == theme.error
    @test RGBf(tb.bordercolor_focused[]) == theme.error

    # Accepted input → the normal themed palette (matches manta_style_textbox!).
    MANTA.manta_flag_textbox!(tb, true, theme)
    @test RGBf(tb.bordercolor[])         == theme.border
    @test RGBf(tb.bordercolor_hover[])   == theme.accent_dim
    @test RGBf(tb.bordercolor_focused[]) == theme.accent
end

@testset "ui: cube status footer tone" begin
    theme = MANTA.default_ui_theme()

    @test MANTA._cube_status_tone("Saved image to /tmp/out.png.") === :success
    @test MANTA._cube_status_color("GIF exported to /tmp/out.gif.", theme) == theme.success

    @test MANTA._cube_status_tone("Failed to save PNG: permission denied") === :error
    @test MANTA._cube_status_color("Histogram bins must be an integer.", theme) == theme.error

    @test MANTA._cube_status_tone("Draw a box with left drag on the image.") === :action
    @test MANTA._cube_status_color("Colormap set to magma.", theme) == theme.accent

    @test MANTA._cube_status_tone("x = 12, y = 42, value = 0.7") === :neutral
    @test MANTA._cube_status_color("x = 12, y = 42, value = 0.7", theme) == theme.text_muted
end

@testset "helpers: validation" begin
    ok, use_manual, clims, msg = MANTA.parse_manual_clims("1.5", "2.5")
    @test ok && use_manual
    @test clims == (1.5f0, 2.5f0)
    @test !isempty(msg)

    ok2, use_manual2, _, _ = MANTA.parse_manual_clims("", "")
    @test ok2 && !use_manual2

    ok3, use_manual3, _, _ = MANTA.parse_manual_clims("9", "2")
    @test ok3 && use_manual3

    ok4, _, _, _ = MANTA.parse_manual_clims("a", "2")
    @test !ok4

    gok, frames, fps, _ = MANTA.parse_gif_request("1", "5", "2", "10", 10)
    @test gok
    @test frames == [1, 3, 5]
    @test fps == 10

    gok2, frames2, _, _ = MANTA.parse_gif_request("5", "1", "2", "12", 10; pingpong = true)
    @test gok2
    @test frames2 == [1, 3, 5, 3]

    gok3, _, _, _ = MANTA.parse_gif_request("1", "5", "0", "12", 10)
    @test !gok3

    cok, cmanual, clevels, _ = MANTA.parse_contour_levels("1, 2  3")
    @test cok && cmanual
    @test clevels == Float32[1, 2, 3]

    cok2, cmanual2, _, _ = MANTA.parse_contour_levels("")
    @test cok2 && !cmanual2

    cok3, _, _, _ = MANTA.parse_contour_levels("1, nope")
    @test !cok3

    sok, smanual, slevels, scolors, _ = MANTA.parse_contour_specs("3:blue, 1:red, 2:#00ffaa")
    @test sok && smanual
    @test slevels == Float32[1, 2, 3]
    @test scolors == ["red", "#00ffaa", "blue"]
    @test MANTA.format_contour_specs(slevels, scolors) == "1:red, 2:#00ffaa, 3:blue"

    color_values = MANTA.contour_color_values(scolors, length(slevels), RGBAf(0, 0, 0, 1))
    @test length(color_values) == 3

    bad_color, _, _, _, _ = MANTA.parse_contour_specs("1:not_a_color")
    @test !bad_color
end

@testset "helpers: simple wcs" begin
    header = Dict{String,Any}(
        "CTYPE1" => "RA---TAN",
        "CUNIT1" => "deg",
        "CRVAL1" => 120.0,
        "CRPIX1" => 1.0,
        "CDELT1" => -0.5,
        "CTYPE2" => "DEC--TAN",
        "CUNIT2" => "deg",
        "CRVAL2" => -30.0,
        "CRPIX2" => 2.0,
        "CDELT2" => 0.25,
    )
    wcs = MANTA.read_simple_wcs(header, 3)
    @test MANTA.has_wcs(wcs, 1)
    @test MANTA.has_wcs(wcs, 2)
    @test !MANTA.has_wcs(wcs, 3)
    @test MANTA.world_coord(wcs, 1, 3) == 119.0
    @test occursin("RA", String(MANTA.wcs_axis_label(wcs, 1)))
    @test occursin("RA---TAN", MANTA.format_world_coord(wcs, 1, 1))
    @test MANTA.data_unit_label(Dict{String,Any}("BUNIT" => "K")) == "K"
    @test MANTA.data_unit_label(Dict{String,Any}("BUNIT" => "   ")) == "value"
    @test MANTA.data_unit_label(nothing) == "value"

    # New: CTYPE classification (base / projection / kind / spectral_quantity).
    @test wcs[1].ctype_base == "RA"
    @test wcs[1].projection == "TAN"
    @test wcs[1].kind === :ra
    @test wcs[2].kind === :dec
    @test wcs[2].projection == "TAN"
    @test wcs[1].spectral_quantity === :other

    unnamed_wcs = MANTA.read_simple_wcs(Dict{String,Any}("CRVAL1" => 0.0, "CDELT1" => 1.0), 1)
    @test MANTA.has_wcs(unnamed_wcs, 1)
    unnamed_label = String(MANTA.wcs_axis_label(unnamed_wcs, 1; fallback = "axis1 [pix]"))
    @test occursin("axis1", unnamed_label)
    @test occursin("[pix]", unnamed_label)
    @test !occursin("world", lowercase(unnamed_label))
end

@testset "helpers: wcs transform" begin
    # 2D sky header with CD matrix (rotated frame) + TAN projection.
    hdr_cd = Dict{String,Any}(
        "CTYPE1" => "RA---TAN", "CUNIT1" => "deg",
        "CRVAL1" => 10.0, "CRPIX1" => 2.0, "CDELT1" => -0.01,
        "CTYPE2" => "DEC--TAN", "CUNIT2" => "deg",
        "CRVAL2" => 20.0, "CRPIX2" => 2.0, "CDELT2" => 0.01,
        # 90° rotation embedded in CD:
        "CD1_1" => 0.0,   "CD1_2" => -0.01,
        "CD2_1" => 0.01,  "CD2_2" => 0.0,
    )
    wt = MANTA.read_wcs_transform(hdr_cd, 2)
    @test wt isa MANTA.WCSTransform
    @test length(wt) == 2
    @test wt.cd !== nothing
    @test !wt.has_pc
    @test MANTA.sky_dims(wt) == (1, 2)
    @test MANTA.spectral_dim(wt) == 0
    # WCSTransform must be index/iterate compatible with a plain WCS vector.
    @test wt[1].kind === :ra
    @test eachindex(wt) == 1:2

    # pixel_scale on a longitude axis applies cos(lat). For CRVAL2 = 20°,
    # cos(20°) ≈ 0.9397.
    sky_scale_ra = MANTA.pixel_scale(wt, 1)
    @test isapprox(sky_scale_ra, 0.01 * cosd(20.0); rtol = 1e-9)
    # The latitude axis is untouched.
    @test isapprox(MANTA.pixel_scale(wt, 2), 0.01; rtol = 1e-9)

    # sky_world_coords at CRPIX returns CRVAL exactly (TAN).
    coords0 = MANTA.sky_world_coords(wt, 2.0, 2.0)
    @test coords0 !== nothing
    @test isapprox(coords0[1], 10.0; atol = 1e-9)
    @test isapprox(coords0[2], 20.0; atol = 1e-9)

    # PC-only header → reconstructs CD from PC × CDELT.
    hdr_pc = Dict{String,Any}(
        "CTYPE1" => "RA---SIN", "CUNIT1" => "deg",
        "CRVAL1" => 0.0, "CRPIX1" => 1.0, "CDELT1" => -0.1,
        "CTYPE2" => "DEC--SIN", "CUNIT2" => "deg",
        "CRVAL2" => 0.0, "CRPIX2" => 1.0, "CDELT2" => 0.1,
        "PC1_1" => 1.0, "PC2_2" => 1.0,
    )
    wt_pc = MANTA.read_wcs_transform(hdr_pc, 2)
    @test wt_pc.has_pc
    @test wt_pc.cd !== nothing
    @test isapprox(wt_pc.cd[1, 1], -0.1; atol = 1e-12)
    @test wt_pc[1].projection == "SIN"
    @test wt_pc[1].kind === :ra
    # SIN: at the pole pixel, returns origin.
    sc = MANTA.sky_world_coords(wt_pc, 1.0, 1.0)
    @test sc !== nothing
    @test isapprox(sc[1], 0.0; atol = 1e-9)
    @test isapprox(sc[2], 0.0; atol = 1e-9)

    # Pure-linear (no CD, no PC) still produces a transform with cd === nothing.
    hdr_lin = Dict{String,Any}(
        "CTYPE1" => "RA---TAN", "CUNIT1" => "deg",
        "CRVAL1" => 0.0, "CRPIX1" => 1.0, "CDELT1" => 0.5,
        "CTYPE2" => "DEC--TAN", "CUNIT2" => "deg",
        "CRVAL2" => 0.0, "CRPIX2" => 1.0, "CDELT2" => 0.5,
    )
    wt_lin = MANTA.read_wcs_transform(hdr_lin, 2)
    @test wt_lin.cd === nothing
    @test !wt_lin.has_pc
    @test isapprox(MANTA.pixel_scale(wt_lin, 1), 0.5; rtol = 1e-9)   # cos(0) = 1

    # CAR projection short-circuits to (lon0 + xi, lat0 + eta).
    hdr_car = Dict{String,Any}(
        "CTYPE1" => "GLON-CAR", "CUNIT1" => "deg",
        "CRVAL1" => 30.0, "CRPIX1" => 1.0, "CDELT1" => 1.0,
        "CTYPE2" => "GLAT-CAR", "CUNIT2" => "deg",
        "CRVAL2" => 0.0,  "CRPIX2" => 1.0, "CDELT2" => 1.0,
    )
    wt_car = MANTA.read_wcs_transform(hdr_car, 2)
    @test wt_car[1].kind === :glon
    @test wt_car[2].kind === :glat
    car = MANTA.sky_world_coords(wt_car, 3.0, 2.0)
    @test car !== nothing
    @test isapprox(car[1], 30.0 + 2.0; atol = 1e-9)
    @test isapprox(car[2], 0.0 + 1.0; atol = 1e-9)

    # Spectral classification: VRAD / FREQ / WAVE → :spectral with the right
    # quantity. wcs_axis_label and spectral_quantity_word reflect the kind.
    hdr_spec = Dict{String,Any}(
        "CTYPE3" => "FREQ", "CUNIT3" => "Hz",
        "CRVAL3" => 1.0e11, "CRPIX3" => 1.0, "CDELT3" => 1.0e6,
    )
    wt_spec = MANTA.read_wcs_transform(hdr_spec, 3)
    @test MANTA.spectral_dim(wt_spec) == 3
    @test MANTA.spectral_quantity(wt_spec, 3) === :frequency
    @test MANTA.spectral_quantity_word(:frequency) == "frequency"
    @test MANTA.spectral_quantity_word(:velocity) == "velocity"
    @test MANTA.spectral_quantity_word(:wavelength) == "wavelength"
    @test MANTA.spectral_quantity_word(:other) == "value"
    @test occursin("frequency", lowercase(String(MANTA.wcs_axis_label(wt_spec, 3))))

    # VRAD and WAVE classification.
    @test MANTA.SimpleWCSAxis("VRAD",   "m/s", 0, 1, 1, true).spectral_quantity === :velocity
    @test MANTA.SimpleWCSAxis("WAVE",   "m",   0, 1, 1, true).spectral_quantity === :wavelength
    @test MANTA.SimpleWCSAxis("STOKES", "",    0, 1, 1, true).kind === :stokes
end

@testset "helpers: cursor readout" begin
    # Pixel coordinates are rounded to the nearest integer FITS pixel.
    s = MANTA.format_cursor_readout(["x" => 12.4, "y" => 7.8], 3.5, "Jy/beam")
    @test occursin("x=12", s)
    @test occursin("y=8", s)
    @test occursin("3.5", s)
    @test occursin("Jy/beam", s)

    # Blank unit is omitted; the value still shows.
    s_nounit = MANTA.format_cursor_readout(["x" => 1, "y" => 1], 2.0, "   ")
    @test occursin("2.0", s_nounit)
    @test !occursin("  2.0 ", s_nounit)  # no trailing unit separator

    # NaN / nothing / missing all render as "NaN" (or are skipped for nothing).
    @test occursin("NaN", MANTA.format_cursor_readout(["x" => 1], NaN, "K"))
    @test occursin("NaN", MANTA.format_cursor_readout(["x" => 1], missing, "K"))
    @test !occursin("NaN", MANTA.format_cursor_readout(["x" => 1], nothing, "K"))

    # World-coordinate strings (from format_world_coord) are appended verbatim,
    # empty ones skipped. Reuse the WCS header built above is overkill — build
    # a tiny RA/DEC pair here.
    hdr = Dict{String,Any}(
        "CTYPE1" => "RA---TAN", "CUNIT1" => "deg",
        "CRVAL1" => 120.0, "CRPIX1" => 1.0, "CDELT1" => -0.5,
        "CTYPE2" => "DEC--TAN", "CUNIT2" => "deg",
        "CRVAL2" => -30.0, "CRPIX2" => 2.0, "CDELT2" => 0.25,
    )
    w = MANTA.read_simple_wcs(hdr, 2)
    world = [MANTA.format_world_coord(w, 1, 3), MANTA.format_world_coord(w, 2, 2)]
    full = MANTA.format_cursor_readout(["x" => 3, "y" => 2], 9.0, "K", world)
    @test occursin("RA---TAN", full)
    @test occursin("DEC--TAN", full)
    @test occursin("K", full)
    # An empty world entry is dropped, not rendered as a stray separator.
    @test !occursin("  =", MANTA.format_cursor_readout(["x" => 1], 1.0, "K", ["", "RA=1"]))

    # Empty everything → empty string (no crash, no stray separators).
    @test MANTA.format_cursor_readout([]) == ""

    # format_world_value formats a *given* world value (no pixel→world step)
    # and stays consistent with format_world_coord ∘ world_coord.
    @test MANTA.format_world_value(w, 1, 119.0) == MANTA.format_world_coord(w, 1, 3)
    @test occursin("RA---TAN", MANTA.format_world_value(w, 1, 119.0))
    @test occursin("deg", MANTA.format_world_value(w, 1, 119.0))
    # No WCS on the axis → "pixN=value" fallback.
    @test occursin("pix3=", MANTA.format_world_value(MANTA.SimpleWCSAxis[], 3, 7.0))
end

@testset "helpers: cursor world strings (exact CD path)" begin
    # Rotated TAN frame with a full CD matrix: cursor_world_strings must use
    # sky_world_coords (CD + deprojection), not the per-axis linear CDELT.
    hdr = Dict{String,Any}(
        "CTYPE1" => "RA---TAN", "CUNIT1" => "deg",
        "CRVAL1" => 150.0, "CRPIX1" => 64.0,
        "CTYPE2" => "DEC--TAN", "CUNIT2" => "deg",
        "CRVAL2" => 2.0, "CRPIX2" => 64.0,
        # ~30° rotation, 0.001 deg/pix scale.
        "CD1_1" => -0.001 * cosd(30), "CD1_2" =>  0.001 * sind(30),
        "CD2_1" =>  0.001 * sind(30), "CD2_2" =>  0.001 * cosd(30),
    )
    wvec  = MANTA.read_simple_wcs(hdr, 2)
    xform = MANTA.read_wcs_transform(hdr, 2)
    @test MANTA.sky_dims(xform) == (1, 2)
    @test xform.cd !== nothing

    px, py = 100.0, 30.0   # well off the reference pixel so rotation matters

    # Exact path matches sky_world_coords formatted through format_world_value.
    sc = MANTA.sky_world_coords(xform, px, py)
    @test sc !== nothing
    exact = MANTA.cursor_world_strings(wvec, xform, [1, 2], [px, py])
    @test exact == [MANTA.format_world_value(wvec, 1, sc[1]),
                    MANTA.format_world_value(wvec, 2, sc[2])]

    # Linear fallback (xform = nothing) differs from the exact CD/projection
    # result at this off-center pixel — proves the CD path actually fires.
    linear = MANTA.cursor_world_strings(wvec, nothing, [1, 2], [px, py])
    @test linear == [MANTA.format_world_coord(wvec, 1, px),
                     MANTA.format_world_coord(wvec, 2, py)]
    @test exact != linear

    # Display order is preserved (x first, then y) — swap dims/pix and check.
    swapped = MANTA.cursor_world_strings(wvec, xform, [2, 1], [py, px])
    @test occursin("DEC--TAN", swapped[1])
    @test occursin("RA---TAN", swapped[2])

    # A spectral 3rd axis stays linear even when the sky pair is exact.
    hdr3 = merge(hdr, Dict{String,Any}(
        "CTYPE3" => "FREQ", "CUNIT3" => "Hz",
        "CRVAL3" => 1.0e11, "CRPIX3" => 1.0, "CDELT3" => 1.0e6))
    wvec3  = MANTA.read_simple_wcs(hdr3, 3)
    xform3 = MANTA.read_wcs_transform(hdr3, 3)
    cube_world = MANTA.cursor_world_strings(wvec3, xform3, [2, 1, 3], [py, px, 5])
    @test length(cube_world) == 3
    @test occursin("FREQ", cube_world[3])
    @test occursin("1.00004e11", cube_world[3])  # 1e11 + (5-1)*1e6

    # No-WCS axes are skipped entirely; empty when nothing carries WCS.
    @test MANTA.cursor_world_strings(MANTA.SimpleWCSAxis[], nothing, [1], [3.0]) == String[]

    # Length mismatch is a hard error (programmer bug, not silent).
    @test_throws ArgumentError MANTA.cursor_world_strings(wvec, xform, [1, 2], [1.0])
end

@testset "helpers: settings io" begin
    mktempdir() do tmp
        settings_path = joinpath(tmp, "viewer_settings.toml")
        payload = Dict{String,Any}(
            "axis" => 2,
            "img_scale" => "log10",
            "colormap" => "viridis",
            "use_manual_clims" => true,
            "clim_min" => 0.1,
            "clim_max" => 42.0,
        )
        MANTA.save_viewer_settings(settings_path, payload)
        @test isfile(settings_path)

        restored = MANTA.load_viewer_settings(settings_path)
        @test restored["axis"] == 2
        @test restored["img_scale"] == "log10"
        @test restored["use_manual_clims"] == true
    end

    # Spectrum y-axis policy schema: a string source + Float32 bounds must
    # survive the TOML round-trip (regression guard for the persisted
    # spec_ylimits_source / spec_ymin / spec_ymax keys).
    mktempdir() do tmp
        settings_path = joinpath(tmp, "spec_ylimits.toml")
        payload = Dict{String,Any}(
            "spec_ylimits_source" => "manual",
            "spec_ymin" => 2.5f0,
            "spec_ymax" => 9.0f0,
        )
        MANTA.save_viewer_settings(settings_path, payload)
        restored = MANTA.load_viewer_settings(settings_path)
        @test restored["spec_ylimits_source"] == "manual"
        @test isapprox(restored["spec_ymin"], 2.5; atol = 1e-6)
        @test isapprox(restored["spec_ymax"], 9.0; atol = 1e-6)
    end
end

@testset "helpers: power spectrum" begin
    # --- Kronecker delta: flat raw |F|² (no window, no demean, no padding).
    @testset "delta is flat" begin
        N = 16
        A = zeros(Float64, N, N)
        A[N ÷ 2 + 1, N ÷ 2 + 1] = 1.0
        res = MANTA.power_spectrum_2d(A; window = :none, demean = false, pad_pow2 = false)
        # All bins identical for a delta input.
        @test maximum(res.P2d) ≈ minimum(res.P2d) atol = 1e-9
        # MASTER-light divisor is ⟨1²⟩ = 1 here, so |F|² should be 1 everywhere.
        @test all(p -> isapprox(p, 1.0; atol = 1e-9), res.P2d)
        @test res.f_sky == 1.0
        @test res.window === :none
        @test !res.padded
    end

    # --- Pure cosine peaks at the expected radial bin.
    @testset "sinusoid peaks at expected k" begin
        N = 64
        m = 5
        A = [cos(2π * m * (j - 1) / N) for i in 1:N, j in 1:N]
        res = MANTA.power_spectrum_2d(A; window = :none, demean = false, pad_pow2 = false)
        radii, prof = MANTA.power_spectrum_1d_radial(res.P2d)
        # Peak should be at the bin matching the spatial frequency m.
        @test argmax(prof) - 1 == m
        # Other bins (excluding immediate neighbours) are much smaller.
        peak_val = prof[m + 1]
        for b in 1:length(prof)
            (b == m || b == m + 1 || b == m + 2) && continue
            @test prof[b] < peak_val * 1e-3
        end
    end

    # Deterministic pseudo-random surface (LCG-shaped sums of trig modes) so the
    # test suite does not need Random/MersenneTwister as a stdlib dependency.
    deterministic_field(N::Int, M::Int) = Float64[
        sin(0.123 * i + 0.371 * j) +
        0.5 * cos(0.7 * i - 0.3 * j) +
        0.25 * sin(0.05 * i * j)
        for i in 1:N, j in 1:M
    ]

    # --- Parseval (window=:none, no padding, no demean): sum(|F|²) = N²·sum(|A|²).
    @testset "parseval no-window no-pad" begin
        A = deterministic_field(8, 8)
        res = MANTA.power_spectrum_2d(A; window = :none, demean = false, pad_pow2 = false)
        # power_spectrum_2d divides by ⟨W²⟩ = 1 here, so it stores |F|² directly.
        # Parseval for non-unitary FFT used by FFTW: sum(|F|²) = N² · sum(|A|²).
        @test isapprox(sum(res.P2d), length(A) * sum(abs2, A); rtol = 1e-9)
    end

    # --- Padding to next power of 2 produces correct effective size.
    @testset "pad to next pow2" begin
        A = deterministic_field(7, 11)
        res = MANTA.power_spectrum_2d(A; window = :none, demean = false, pad_pow2 = true)
        @test res.ny_eff == 8 && res.nx_eff == 16
        @test res.padded
        @test size(res.P2d) == (8, 16)
        # Already-pow2 input does not grow.
        B = deterministic_field(8, 8)
        res2 = MANTA.power_spectrum_2d(B; window = :none, demean = false, pad_pow2 = true)
        @test !res2.padded
        @test (res2.ny_eff, res2.nx_eff) == (8, 8)
    end

    # --- NaN apodization runs and reports a coherent f_sky.
    @testset "NaN apodization" begin
        N = 32
        A = deterministic_field(N, N)
        # Carve a NaN strip on the left.
        A[:, 1:6] .= NaN
        res = MANTA.power_spectrum_2d(A; window = :hann,
                                       apodize_nan = true, nan_taper = 3)
        @test res.apodized
        @test isapprox(res.f_sky, (N * (N - 6)) / (N * N); atol = 1e-9)
        @test all(isfinite, res.P2d)
        @test res.w_norm > 0
    end

    # --- Hamming window is applied (different result from :none).
    @testset "window kinds differ" begin
        A = deterministic_field(16, 16)
        r_none    = MANTA.power_spectrum_2d(A; window = :none, demean = false)
        r_hann    = MANTA.power_spectrum_2d(A; window = :hann, demean = false)
        r_hamming = MANTA.power_spectrum_2d(A; window = :hamming, demean = false)
        @test r_none.P2d != r_hann.P2d
        @test r_hann.P2d != r_hamming.P2d
    end

    # --- Log-log slope fit recovers an injected power law.
    @testset "fit_loglog_slope" begin
        k = collect(1.0:1.0:50.0)
        # P = 10 * k^(-2.7)
        p = 10.0 .* (k .^ -2.7)
        slope, intercept, n = MANTA.fit_loglog_slope(k, p; kmin = 2.0, kmax = 40.0)
        @test isapprox(slope, -2.7; atol = 1e-9)
        @test isapprox(intercept, log10(10.0); atol = 1e-9)
        @test n == count(ki -> 2.0 <= ki <= 40.0, k)

        # Empty band -> NaN, n = 0.
        s2, i2, n2 = MANTA.fit_loglog_slope(k, p; kmin = 1e9, kmax = 1e10)
        @test isnan(s2) && isnan(i2) && n2 == 0

        # Drops non-positive p (log undefined).
        kn = [1.0, 2.0, 4.0, 8.0]
        pn = [1.0, 0.0, 1 / 16, 1 / 64]   # zero is dropped
        s3, _, n3 = MANTA.fit_loglog_slope(kn, pn)
        @test n3 == 3
        @test isapprox(s3, -2.0; atol = 1e-9)
    end
end

# ----------------------------------------------------------------------------
# Refactor regression tests (Phase 1)
# ----------------------------------------------------------------------------

@testset "helpers: as_float32 and get_slice variants" begin
    # No-op on already-Float32 dense arrays.
    A32 = rand(Float32, 4, 5)
    @test MANTA.as_float32(A32) === A32

    # Conversion path is allocated, but only when needed.
    A64 = rand(Float64, 4, 5)
    A32_from_64 = MANTA.as_float32(A64)
    @test eltype(A32_from_64) === Float32
    @test size(A32_from_64) == size(A64)

    # get_slice_view returns a view; get_slice_copy returns an independent
    # buffer.
    cube = reshape(collect(Float32, 1:24), 2, 3, 4)
    sv1 = MANTA.get_slice_view(cube, 1, 1)
    sv2 = MANTA.get_slice_view(cube, 2, 1)
    sv3 = MANTA.get_slice_view(cube, 3, 1)
    @test size(sv1) == (3, 4)
    @test size(sv2) == (2, 4)
    @test size(sv3) == (2, 3)
    @test sv1 isa SubArray
    @test sv2 isa SubArray
    @test sv3 isa SubArray

    sc1 = MANTA.get_slice_copy(cube, 1, 1)
    @test sc1 == sv1
    sc1[1, 1] = -99f0
    @test cube[1, 1, 1] != -99f0   # the cube was not mutated by editing the copy.

    @test_throws ArgumentError MANTA.get_slice_view(cube, 4, 1)
    @test_throws BoundsError MANTA.get_slice_view(cube, 1, 99)
end

@testset "helpers: parse_path_spec" begin
    @test first(MANTA.parse_path_spec("foo.fits")) === :fits
    @test first(MANTA.parse_path_spec("foo.fit")) === :fits
    @test first(MANTA.parse_path_spec("foo.FITS.GZ")) === :fits
    @test first(MANTA.parse_path_spec("foo.h5")) === :hdf5
    @test first(MANTA.parse_path_spec("foo.hdf5")) === :hdf5

    # path:address syntax
    kind, p, addr = MANTA.parse_path_spec("file.h5:/group/ds")
    @test kind === :hdf5
    @test p == "file.h5"
    @test addr == "/group/ds"

    # Windows-drive letter is NOT treated as an HDF5 spec.
    @test first(MANTA.parse_path_spec("C:/data/foo.fits")) === :fits

    # Unknown extension.
    @test first(MANTA.parse_path_spec("notes.txt")) === :unknown
end

@testset "helpers: FITS export headers" begin
    # Build a reference cube header with two sky axes and a spectral axis.
    src_keys = String[
        "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3",
        "CTYPE1", "CRPIX1", "CRVAL1", "CDELT1", "CUNIT1",
        "CTYPE2", "CRPIX2", "CRVAL2", "CDELT2", "CUNIT2",
        "CTYPE3", "CRPIX3", "CRVAL3", "CDELT3", "CUNIT3",
        "BUNIT", "OBJECT", "TELESCOP", "SPECSYS", "RESTFRQ",
        "PC1_2", "PC2_1", "PV2_1",
        "HISTORY",
    ]
    src_vals = Any[
        -32, 3, 8, 6, 4,
        "RA---TAN", 4.5, 150.0, -0.01, "deg",
        "DEC--TAN", 3.5,  10.0,  0.01, "deg",
        "VRAD",    1.0,   2.0e5, 1.0e3, "m/s",
        "K", "Source X", "ALMA", "LSRK", 1.0e11,
        0.0, 0.0, 0.1,
        nothing,
    ]
    src_comms = String[
        "", "", "", "", "",
        "axis1", "", "", "", "",
        "axis2", "", "", "", "",
        "axis3", "", "", "", "",
        "data unit", "", "", "", "",
        "", "", "",
        "original cube",
    ]
    src_hdr = FITSIO.FITSHeader(src_keys, src_vals, src_comms)

    # --- slice header: drop axis 3, keep BUNIT/OBJECT, add HISTORY ---
    sh = MANTA.fits_header_for_slice(src_hdr, 3, 12; source_id = "cube_x")
    @test sh isa FITSIO.FITSHeader
    @test haskey(sh, "CTYPE1") && haskey(sh, "CTYPE2")
    @test !haskey(sh, "CTYPE3")
    @test sh["CTYPE1"] == "RA---TAN"
    @test sh["CTYPE2"] == "DEC--TAN"
    @test sh["BUNIT"] == "K"
    @test sh["OBJECT"] == "Source X"
    # HISTORY is stored as comment text; check at least one MANTA history line.
    history_idx = findall(==("HISTORY"), sh.keys)
    @test !isempty(history_idx)
    @test any(occursin("MANTA slice axis=3 index=12 source=cube_x", sh.comments[i])
              for i in history_idx)

    # --- slice header: dropping axis 1 should renumber axis 2 -> 1 and 3 -> 2
    sh1 = MANTA.fits_header_for_slice(src_hdr, 1, 4)
    @test sh1["CTYPE1"] == "DEC--TAN"
    @test sh1["CTYPE2"] == "VRAD"
    @test sh1["CUNIT2"] == "m/s"

    # --- moment header: order 0 keeps BUNIT, order 1/2 use CUNIT_axis ---
    m0 = MANTA.fits_header_for_moment(src_hdr, 3, 0)
    @test m0["BUNIT"] == "K"
    m1 = MANTA.fits_header_for_moment(src_hdr, 3, 1)
    @test m1["BUNIT"] == "m/s"
    m2 = MANTA.fits_header_for_moment(src_hdr, 3, 2)
    @test m2["BUNIT"] == "m/s"
    # COMMENT card explaining BUNIT must be present.
    @test any(==("COMMENT"), m1.keys)

    # Missing CUNIT path produces "?" placeholder (Composer + commentaire).
    src_keys2 = copy(src_keys)
    src_vals2 = copy(src_vals)
    src_comms2 = copy(src_comms)
    # Wipe CUNIT3.
    cu_idx = findfirst(==("CUNIT3"), src_keys2)
    @assert cu_idx !== nothing
    deleteat!(src_keys2, cu_idx)
    deleteat!(src_vals2, cu_idx)
    deleteat!(src_comms2, cu_idx)
    src_hdr_nocunit = FITSIO.FITSHeader(src_keys2, src_vals2, src_comms2)
    m1_q = MANTA.fits_header_for_moment(src_hdr_nocunit, 3, 1)
    @test m1_q["BUNIT"] == "?"

    # --- region spectrum: only axis 3 WCS survives, renumbered to axis 1 ---
    rs = MANTA.fits_header_for_region_spectrum(src_hdr, 3, 42)
    @test rs["CTYPE1"] == "VRAD"
    @test rs["CUNIT1"] == "m/s"
    @test rs["CRVAL1"] ≈ 2.0e5
    @test !haskey(rs, "CTYPE2")
    @test !haskey(rs, "CTYPE3")
    @test rs["SPECSYS"] == "LSRK"
    @test rs["RESTFRQ"] ≈ 1.0e11

    # --- filtered cube: full WCS kept, MANTA history added ---
    fc = MANTA.fits_header_for_filtered_cube(src_hdr, 3, 1.5)
    @test fc["CTYPE3"] == "VRAD"
    @test fc["BUNIT"] == "K"
    @test any(occursin("MANTA filtered axis=3 sigma=1.5", fc.comments[i])
              for i in findall(==("HISTORY"), fc.keys))

    # --- nothing input passes through ---
    @test MANTA.fits_header_for_slice(nothing, 3, 1) === nothing
    @test MANTA.fits_header_for_moment(nothing, 3, 0) === nothing
    @test MANTA.fits_header_for_region_spectrum(nothing, 3, 1) === nothing
    @test MANTA.fits_header_for_filtered_cube(nothing, 3, 1.0) === nothing

    # --- display formatter for the header pop-out window ---
    display_lines = MANTA.fits_header_display_lines(src_hdr)
    @test occursin("CTYPE1", display_lines[6])
    @test occursin("'RA---TAN'", display_lines[6])
    @test occursin("axis1", display_lines[6])
    @test any(occursin("HISTORY", line) && occursin("original cube", line)
              for line in display_lines)
    @test MANTA.fits_header_display_lines(nothing) == ["(no FITS header available)"]

    # --- end-to-end: write a real FITS file and read the header back ---
    mktempdir() do dir
        out = joinpath(dir, "slice.fits")
        FITSIO.FITS(out, "w") do f
            FITSIO.write(f, rand(Float32, 8, 6); header = sh)
        end
        FITSIO.FITS(out, "r") do f
            rh = FITSIO.read_header(f[1])
            # FITSIO may pad string values with trailing spaces.
            @test strip(String(rh["CTYPE1"])) == "RA---TAN"
            @test strip(String(rh["CTYPE2"])) == "DEC--TAN"
            @test strip(String(rh["BUNIT"])) == "K"
            @test strip(String(rh["OBJECT"])) == "Source X"
            @test !haskey(rh, "CTYPE3")
        end
    end
end

# ----------------------------------------------------------------------------
# Keyboard shortcuts helper (headless)
# ----------------------------------------------------------------------------
@testset "helpers: keyboard shortcuts" begin
    # --- ShortcutBinding constructor: validation + field access ---
    b = MANTA.ShortcutBinding(Makie.Keyboard.r, () -> nothing;
                              description = "test", modifier = :none)
    @test b.key === Makie.Keyboard.r
    @test b.description == "test"
    @test b.modifier === :none
    # Unknown modifier must error to guard against typos at the call site.
    @test_throws Exception MANTA.ShortcutBinding(Makie.Keyboard.r, () -> nothing;
                                                  modifier = :super)

    # --- Pretty-printing: format_shortcut + format_shortcut_help ---
    @test MANTA.format_shortcut(b) == "R"
    bshift = MANTA.ShortcutBinding(Makie.Keyboard.slash, () -> nothing;
                                    description = "help", modifier = :shift)
    @test MANTA.format_shortcut(bshift) == "Shift + /"
    bctrl = MANTA.ShortcutBinding(Makie.Keyboard.s, () -> nothing;
                                   description = "save", modifier = :ctrl)
    @test MANTA.format_shortcut(bctrl) == "Ctrl + S"
    @test MANTA.format_shortcut(MANTA.ShortcutBinding(Makie.Keyboard.left,
                                                       () -> nothing)) == "←"

    # An empty description is silently skipped in the help string.
    b_nodesc = MANTA.ShortcutBinding(Makie.Keyboard.up, () -> nothing)
    help_str = MANTA.format_shortcut_help([b, b_nodesc])
    @test occursin("R test", help_str)
    @test !occursin("↑", help_str)
    @test MANTA.shortcut_help_message([b, bshift]) == "Shortcuts — R test · Shift + / help"
    @test MANTA.shortcut_help_message([b_nodesc]) == "Shortcuts — none"

    # --- Firing a press event invokes the registered action ---
    fig = Figure(size = (320, 200))
    n = Ref(0)
    b_inc = MANTA.ShortcutBinding(Makie.Keyboard.r, () -> (n[] += 1);
                                   description = "inc")
    MANTA.register_shortcuts!(fig, [b_inc])
    events(fig).keyboardbutton[] = Makie.KeyEvent(Makie.Keyboard.r, Makie.Keyboard.press)
    @test n[] == 1
    # Release events must NOT refire the binding.
    events(fig).keyboardbutton[] = Makie.KeyEvent(Makie.Keyboard.r, Makie.Keyboard.release)
    @test n[] == 1
    # Repeated press still triggers (each notify is a fresh dispatch).
    events(fig).keyboardbutton[] = Makie.KeyEvent(Makie.Keyboard.r, Makie.Keyboard.press)
    @test n[] == 2
    # Unrelated keys are ignored.
    events(fig).keyboardbutton[] = Makie.KeyEvent(Makie.Keyboard.s, Makie.Keyboard.press)
    @test n[] == 2

    # --- is_blocked predicate gates dispatch ---
    fig2 = Figure(size = (320, 200))
    blocked = Ref(true)
    m = Ref(0)
    b_block = MANTA.ShortcutBinding(Makie.Keyboard.r, () -> (m[] += 1))
    MANTA.register_shortcuts!(fig2, [b_block]; is_blocked = () -> blocked[])
    events(fig2).keyboardbutton[] = Makie.KeyEvent(Makie.Keyboard.r, Makie.Keyboard.press)
    @test m[] == 0
    blocked[] = false
    events(fig2).keyboardbutton[] = Makie.KeyEvent(Makie.Keyboard.r, Makie.Keyboard.press)
    @test m[] == 1

    # --- Action-level exception does not kill the figure handler ---
    fig3 = Figure(size = (320, 200))
    survived = Ref(0)
    b_bad  = MANTA.ShortcutBinding(Makie.Keyboard.r, () -> error("boom"))
    b_good = MANTA.ShortcutBinding(Makie.Keyboard.s,
                                    () -> (survived[] += 1))
    MANTA.register_shortcuts!(fig3, [b_bad, b_good])
    @test_logs (:warn, "Shortcut handler error") events(fig3).keyboardbutton[] = Makie.KeyEvent(Makie.Keyboard.r, Makie.Keyboard.press)
    events(fig3).keyboardbutton[] = Makie.KeyEvent(Makie.Keyboard.s, Makie.Keyboard.press)
    @test survived[] == 1
end

# ----------------------------------------------------------------------------
# Centered zoom shortcuts (headless)
# ----------------------------------------------------------------------------
@testset "helpers: zoom shortcuts" begin
    # --- zoom_limits: pure math, center-preserving, uniform scaling ---
    @test MANTA.zoom_limits((0.0, 10.0, 0.0, 20.0), 0.8) == (1.0, 9.0, 2.0, 18.0)
    @test MANTA.zoom_limits((0.0, 10.0, 0.0, 20.0), 1.25) == (-1.25, 11.25, -2.5, 22.5)
    # Center is preserved for any factor.
    let (a, b, c, d) = MANTA.zoom_limits((2.0, 6.0, -4.0, 4.0), 0.5)
        @test (a + b) / 2 ≈ 4.0
        @test (c + d) / 2 ≈ 0.0
        @test (b - a) ≈ 2.0   # 4-wide span × 0.5
        @test (d - c) ≈ 4.0   # 8-wide span × 0.5
    end
    # Degenerate / invalid inputs are returned unchanged (never NaN/inverted).
    @test MANTA.zoom_limits((0.0, 0.0, 0.0, 10.0), 0.8) == (0.0, 0.0, 0.0, 10.0)
    @test MANTA.zoom_limits((0.0, 10.0, 0.0, 10.0), 0.0) == (0.0, 10.0, 0.0, 10.0)
    @test MANTA.zoom_limits((0.0, 10.0, 0.0, 10.0), -1.0) == (0.0, 10.0, 0.0, 10.0)
    let bad = MANTA.zoom_limits((NaN, 10.0, 0.0, 10.0), 0.8)
        @test bad === (NaN, 10.0, 0.0, 10.0)
    end

    # --- key labels for the new bindings render as +/− ---
    @test MANTA.format_shortcut(MANTA.ShortcutBinding(Makie.Keyboard.equal, () -> nothing)) == "+"
    @test MANTA.format_shortcut(MANTA.ShortcutBinding(Makie.Keyboard.minus, () -> nothing)) == "−"
    @test MANTA.format_shortcut(MANTA.ShortcutBinding(Makie.Keyboard.kp_add, () -> nothing)) == "+"
    @test MANTA.format_shortcut(MANTA.ShortcutBinding(Makie.Keyboard.kp_subtract, () -> nothing)) == "−"

    # --- zoom_shortcut_bindings: shape + help (numpad twins are undocumented) ---
    msgs = String[]
    binds = MANTA.zoom_shortcut_bindings(nothing; on_change = m -> push!(msgs, m))
    @test length(binds) == 4
    # Only the main-row +/- carry a description, so help lists each once.
    @test MANTA.format_shortcut_help(binds) == "+ zoom in · − zoom out"

    # --- zoom_axis!: reads finallimits, applies a centered limits! ---
    fig = Figure(size = (320, 200))
    ax  = Axis(fig[1, 1])
    ax.finallimits[] = Makie.Rect2f(0.0f0, 0.0f0, 10.0f0, 20.0f0)  # x∈[0,10], y∈[0,20]
    MANTA.zoom_axis!(ax, 0.5)
    tl = ax.targetlimits[]
    @test isapprox(tl.origin[1], 2.5; atol = 1e-4)
    @test isapprox(tl.origin[2], 5.0; atol = 1e-4)
    @test isapprox(tl.widths[1], 5.0; atol = 1e-4)
    @test isapprox(tl.widths[2], 10.0; atol = 1e-4)

    # --- firing the `=`/`+` binding zooms in and echoes a status string ---
    fig2 = Figure(size = (320, 200))
    ax2  = Axis(fig2[1, 1])
    ax2.finallimits[] = Makie.Rect2f(0.0f0, 0.0f0, 100.0f0, 100.0f0)
    status = Ref("")
    z = MANTA.zoom_shortcut_bindings(ax2; on_change = s -> (status[] = s))
    MANTA.register_shortcuts!(fig2, z)
    events(fig2).keyboardbutton[] = Makie.KeyEvent(Makie.Keyboard.equal, Makie.Keyboard.press)
    @test status[] == "Zoomed in."
    @test ax2.targetlimits[].widths[1] < 100.0f0   # span shrank → zoomed in
    events(fig2).keyboardbutton[] = Makie.KeyEvent(Makie.Keyboard.minus, Makie.Keyboard.press)
    @test status[] == "Zoomed out."
end

# ----------------------------------------------------------------------------
# Shortcut help window (headless)
# ----------------------------------------------------------------------------
@testset "helpers: shortcut help window" begin
    # Typical multi-binding set, including an undocumented binding that
    # the window is supposed to skip.
    bindings = MANTA.ShortcutBinding[
        MANTA.ShortcutBinding(Makie.Keyboard.r, () -> nothing;
                              description = "reset zoom"),
        MANTA.ShortcutBinding(Makie.Keyboard.s, () -> nothing;
                              description = "save image"),
        MANTA.ShortcutBinding(Makie.Keyboard.slash, () -> nothing;
                              description = "this help", modifier = :shift),
        # Undocumented binding (empty description) must NOT crash the
        # window and must not appear in the rendered list.
        MANTA.ShortcutBinding(Makie.Keyboard.tab, () -> nothing),
    ]

    fig = MANTA.open_shortcut_help_window(bindings;
        title = "Test shortcuts",
        activate_gl = false, display_fig = false)
    @test fig isa Figure

    # Empty bindings: window still renders with a placeholder message.
    fig_empty = MANTA.open_shortcut_help_window(MANTA.ShortcutBinding[];
        title = "Empty", activate_gl = false, display_fig = false)
    @test fig_empty isa Figure

    # Custom figure size override is honoured.
    fig_sized = MANTA.open_shortcut_help_window(bindings;
        figsize = (420, 300),
        activate_gl = false, display_fig = false)
    @test fig_sized isa Figure
    @test size(fig_sized.scene) == (420, 300)

    # Default size scales with the documented-binding count: more bindings
    # should yield at least as tall a window as fewer bindings.
    one_binding = MANTA.ShortcutBinding[
        MANTA.ShortcutBinding(Makie.Keyboard.r, () -> nothing;
                              description = "only one"),
    ]
    fig_small = MANTA.open_shortcut_help_window(one_binding;
        activate_gl = false, display_fig = false)
    fig_big   = MANTA.open_shortcut_help_window(bindings;
        activate_gl = false, display_fig = false)
    @test size(fig_big.scene)[2] >= size(fig_small.scene)[2]
end

# ----------------------------------------------------------------------------
# Structured errors (Errors.jl)
# ----------------------------------------------------------------------------
@testset "helpers: structured errors" begin
    # File not found → actionable message with path + hint.
    err_str = sprint(showerror,
        MANTA.FileNotFoundError("/no/such.fits", "Vérifie le chemin."))
    @test occursin("/no/such.fits", err_str)
    @test occursin("Vérifie", err_str)

    # require_file roundtrip
    @test_throws MANTA.FileNotFoundError MANTA.require_file("/definitely/missing")
    mktemp() do path, io
        write(io, "X"); close(io)
        @test MANTA.require_file(path) == String(path)
    end

    # invalid_kwarg → carries kwarg name + value + hint.
    e = try
        MANTA.invalid_kwarg(:hdu, -3; hint = "Doit être ≥ 0.")
    catch err
        err
    end
    @test e isa MANTA.InvalidArgumentError
    @test e.name === :hdu
    @test e.value == -3
    @test occursin("hdu", sprint(showerror, e))

    # invalid_hdu → carries requested + available indices.
    e = try
        MANTA.invalid_hdu("/foo.fits", 5, 2)
    catch err
        err
    end
    @test e isa MANTA.HDUSelectionError
    @test e.requested == 5 && e.available == 2
    @test occursin("HDU #5", sprint(showerror, e))

    # MANTAError is the abstract root: catch should work uniformly.
    caught = try
        throw(MANTA.DatasetShapeError("dimensions inattendues",
                                       "Vérifie ton fichier."))
    catch err
        err isa MANTA.MANTAError
    end
    @test caught
end

# ----------------------------------------------------------------------------
# Progress + cancellation
# ----------------------------------------------------------------------------
@testset "helpers: progress + cancellation" begin
    # CancelToken: flip + idempotency
    tok = MANTA.CancelToken()
    @test !MANTA.is_cancelled(tok)
    MANTA.cancel!(tok)
    @test MANTA.is_cancelled(tok)
    MANTA.cancel!(tok)              # idempotent
    @test MANTA.is_cancelled(tok)
    @test !MANTA.is_cancelled(nothing)   # nothing-friendly

    # ProgressTracker: tick! advances counter and observable.
    p = MANTA.ProgressTracker(total = 4, label = "scan")
    @test p.progress[] == 0.0
    @test p.status[] == "scan"
    MANTA.tick!(p; status = "1/4")
    @test p.progress[] == 0.25
    @test p.status[] == "1/4"
    MANTA.tick!(p); MANTA.tick!(p); MANTA.tick!(p)
    @test p.progress[] == 1.0

    # set_progress! clamps to [0, 1].
    p2 = MANTA.ProgressTracker(total = 0)
    MANTA.set_progress!(p2, -0.1)
    @test p2.progress[] == 0.0
    MANTA.set_progress!(p2, 1.5; status = "overshoot")
    @test p2.progress[] == 1.0
    @test p2.status[] == "overshoot"

    # finish! pins to 1.
    p3 = MANTA.ProgressTracker(total = 10)
    MANTA.finish!(p3; status = "done")
    @test p3.progress[] == 1.0
    @test p3.status[] == "done"

    # nothing-friendly overloads do not crash.
    @test MANTA.tick!(nothing) === nothing
    @test MANTA.set_progress!(nothing, 0.5) === nothing
    @test MANTA.finish!(nothing) === nothing

    # with_progress runs body + finishes tracker on normal return.
    val = MANTA.with_progress("compute", total = 3) do prog, tok
        @test prog isa MANTA.ProgressTracker
        @test tok  isa MANTA.CancelToken
        for k in 1:3
            MANTA.is_cancelled(tok) && return nothing
            MANTA.tick!(prog)
        end
        return 42
    end
    @test val == 42
end

# ----------------------------------------------------------------------------
# Downsampling for display
# ----------------------------------------------------------------------------
@testset "helpers: downsampling" begin
    # No downsampling required when the array fits.
    A = rand(Float32, 100, 80)
    out, s = MANTA.auto_downsample(A; max_pixels = 4096)
    @test s == 1
    @test size(out) == size(A)
    @test eltype(out) == Float32

    # Large array: integer stride that brings both dims ≤ max_pixels.
    B = ones(Float32, 8000, 4000)
    out2, s2 = MANTA.auto_downsample(B; max_pixels = 4096)
    @test s2 >= 2
    @test maximum(size(out2)) <= 4096

    # Block-mean preserves the constant value of a constant array.
    C = fill(7f0, 200, 100)
    outc, _ = MANTA.auto_downsample(C; max_pixels = 50)
    @test all(isapprox.(outc, 7f0; atol = 1f-6))

    # Block-mean ignores NaNs (one NaN in a block ≠ NaN output).
    D = ones(Float32, 4, 4); D[1, 1] = NaN32
    outd = MANTA.downsample_block_mean(D, 4)
    @test size(outd) == (1, 1)
    @test isfinite(outd[1, 1])
    @test outd[1, 1] ≈ 1f0

    # downsample_factor edge cases.
    @test MANTA.downsample_factor((100, 80), 4096) == 1
    @test MANTA.downsample_factor((8000, 4000), 4096) == 2
    @test_throws ArgumentError MANTA.downsample_factor((100,), 0)

    # Block centers stay in the original pixel coordinate system.
    @test MANTA.downsample_centers(5, 2) == Float32[1.5, 3.5, 5.0]
    @test MANTA.downsample_centers(3, 1) == Float32[1.0, 2.0, 3.0]

    # 3D variant: only downsamples the spatial dims, preserves nz.
    E = rand(Float32, 1000, 1000, 5)
    outE, sE = MANTA.auto_downsample(E; max_pixels = 200)
    @test sE >= 5
    @test size(outE, 3) == 5
end

# ----------------------------------------------------------------------------
# Undo / redo stack
# ----------------------------------------------------------------------------
@testset "helpers: undo/redo" begin
    s = MANTA.UndoRedoStack{NamedTuple{(:v,),Tuple{Int}}}(capacity = 4)
    @test isempty(s)
    @test !s.can_undo[]
    @test !s.can_redo[]

    MANTA.register_state!(s, (v = 1,))
    MANTA.register_state!(s, (v = 2,))
    MANTA.register_state!(s, (v = 3,))
    @test length(s) == 3
    @test MANTA.current(s) == (v = 3,)
    @test s.can_undo[]
    @test !s.can_redo[]

    snap = MANTA.undo!(s)
    @test snap == (v = 2,)
    @test s.can_redo[]
    snap = MANTA.undo!(s)
    @test snap == (v = 1,)
    @test !s.can_undo[]

    snap = MANTA.redo!(s)
    @test snap == (v = 2,)
    @test s.can_undo[]
    @test s.can_redo[]

    # New entry after undo discards the redo tail.
    MANTA.register_state!(s, (v = 99,))
    @test MANTA.current(s) == (v = 99,)
    @test !s.can_redo[]

    # Deduplication: same value twice → no new entry.
    n_before = length(s)
    MANTA.register_state!(s, (v = 99,))
    @test length(s) == n_before

    # Capacity: oldest snapshots dropped from the bottom.
    s2 = MANTA.UndoRedoStack{Int}(capacity = 3)
    for k in 1:5
        MANTA.register_state!(s2, k)
    end
    @test length(s2) == 3
    @test MANTA.current(s2) == 5
    snaps = [MANTA.undo!(s2) for _ in 1:2]
    @test snaps == [4, 3]
    @test MANTA.undo!(s2) === nothing   # already at bottom

    # with_suppression: register_state! is a no-op inside.
    s3 = MANTA.UndoRedoStack(0)
    MANTA.register_state!(s3, 1)
    MANTA.with_suppression(s3) do
        MANTA.register_state!(s3, 42)
    end
    @test MANTA.current(s3) == 1
end

# ----------------------------------------------------------------------------
# Plugin extension surface
# ----------------------------------------------------------------------------
@testset "helpers: plugins" begin
    MANTA.clear_plugins!()
    @test isempty(MANTA.list_plugins(:loader))

    # Loader plugin: matcher + loader.
    matcher = path -> endswith(path, ".dummy")
    loader  = (path; kwargs...) -> :loaded
    MANTA.register_plugin!(:loader, (matcher, loader))
    @test length(MANTA.list_plugins(:loader)) == 1
    @test MANTA.plugin_load("/tmp/foo.dummy") === :loaded
    @test MANTA.plugin_load("/tmp/foo.bar") === nothing

    # Dataset-view plugin: matched on (type, name).
    MANTA.register_plugin!(:dataset_view,
        (MANTA.ImageDataset, :flatten, (ds; kwargs...) -> :ok))
    img = MANTA.load_dataset(ones(Float32, 4, 4))
    @test MANTA.plugin_view(img, :flatten) === :ok
    @test MANTA.plugin_view(img, :other)   === nothing

    # Postprocess plugin: catches user errors and keeps going.
    seen = Ref(0)
    bad  = (fig, ds, opts) -> error("boom")
    good = (fig, ds, opts) -> (seen[] += 1; nothing)
    MANTA.register_plugin!(:postprocess, bad)
    MANTA.register_plugin!(:postprocess, good)
    @test_logs (:warn, r"plugin postprocess") MANTA.run_postprocess!(nothing, img)
    @test seen[] == 1

    # Unregister: identity-based, idempotent.
    n = MANTA.unregister_plugin!(:postprocess, good)
    @test n == 1
    MANTA.run_postprocess!(nothing, img)   # `seen` should not advance
    @test seen[] == 1

    MANTA.clear_plugins!()
end

# ----------------------------------------------------------------------------
# Backend helpers
# ----------------------------------------------------------------------------
@testset "helpers: backend selection" begin
    # is_headless_env: env-var override.
    withenv("MANTA_HEADLESS" => "1") do
        @test MANTA.is_headless_env() == true
    end
    withenv("MANTA_HEADLESS" => "") do
        # We don't assert the negative case across CI variants — we only
        # require that the helper returns a Bool.
        @test MANTA.is_headless_env() isa Bool
    end

    # pick_backend!(false) → :CairoMakie. pick_backend!(true) may downgrade
    # if no GL is available; either symbol is acceptable.
    @test MANTA.pick_backend!(false) === :CairoMakie
    sym = MANTA.pick_backend!(true)
    @test sym in (:GLMakie, :CairoMakie)

    # with_export_backend always runs the body; restores prev backend on exit.
    MANTA.pick_backend!(false)
    result = MANTA.with_export_backend() do
        42
    end
    @test result == 42
end

# ----------------------------------------------------------------------------
# Drag-and-drop helpers
# ----------------------------------------------------------------------------
@testset "helpers: drag-and-drop" begin
    empty!(MANTA._DROP_ARMED)

    fig = Figure(size = (120, 90))
    @test MANTA.enable_file_drop!(fig; activate_gl = false, display_fig = false) === fig
    @test haskey(MANTA._DROP_ARMED, fig.scene)

    n_armed = length(MANTA._DROP_ARMED)
    @test MANTA.enable_file_drop!(fig; activate_gl = false, display_fig = false) === fig
    @test length(MANTA._DROP_ARMED) == n_armed
end
