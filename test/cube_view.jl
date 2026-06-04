# path: test/cube_view.jl
# Testsets covering cube/viewer integration: smoke + error handling, dataset
# dispatch, the vector viewer (helpers + headless), and view_cube field respect.

@testset "cube overview geometry" begin
    g = MANTA._cube_overview_geometry((4, 5, 6), 3, 4, 2, 3, 1)
    @test g.active_axis == 3
    @test g.active_progress_value ≈ 3 / 5
    @test g.marker_points == Point2f[Point2f(1 / 3, 3), Point2f(0.5, 2), Point2f(3 / 5, 1)]
    @test g.active_progress == Point2f[Point2f(0, 1), Point2f(3 / 5, 1)]

    one_wide = MANTA._cube_overview_geometry((1, 8, 8), 1, 1, 1, 4, 4)
    @test one_wide.active_progress_value == 0.5f0
end

@testset "cube 3D orientation geometry" begin
    R = MANTA._cube3d_axis_angle_rotation((0, 0, 1), 90)
    p = MANTA._cube3d_apply_rotation(R, Point3f(1, 0, 0))
    @test p[1] ≈ 0 atol = 1f-6
    @test p[2] ≈ 1 atol = 1f-6
    @test p[3] ≈ 0 atol = 1f-6

    R2 = MANTA._cube3d_compose_rotation(MANTA._cube3d_identity_rotation(), (1, 0, 0), 180)
    @test size(R2) == (3, 3)
    @test_throws ArgumentError MANTA._cube3d_axis_angle_rotation((0, 0, 0), 12)

    g3 = MANTA._cube3d_view_geometry((4, 5, 6), 3, 4, 2, 3, 1, MANTA._cube3d_identity_rotation())
    @test length(g3.box_segments) == 24
    @test length(g3.slice_loop) == 5
    @test length(g3.selected_point) == 1
    @test length(g3.axis_segments) == 6

    cube = Float32[i + 10j + 100k for i in 1:3, j in 1:4, k in 1:5]
    pmean = MANTA._cube_rotated_projection(cube, size(cube), 3, (0, 0, 1), 0, :mean)
    @test pmean ≈ dropdims(mean(cube; dims = 3); dims = 3)
    psum = MANTA._cube_rotated_projection(cube, size(cube), 1, (1, 0, 0), 0, :sum)
    @test psum ≈ dropdims(sum(cube; dims = 1); dims = 1)
    @test_throws ArgumentError MANTA._cube_rotated_projection(cube, size(cube), 3, (0, 0, 1), 0, :bad)
end

@testset "cube UI display checkboxes propagate to render state" begin
    invert_cmap = Observable(false)

    enabled = MANTA._apply_invert_colormap_toggle!(true, invert_cmap)
    @test enabled
    @test invert_cmap[]

    enabled = MANTA._apply_invert_colormap_toggle!(false, invert_cmap)
    @test !enabled
    @test !invert_cmap[]

    gauss_on = Observable(false)
    layout_mode = Observable(:base)
    refresh_count = Ref(0)
    ps_count = Ref(0)
    statuses = String[]

    enabled = MANTA._apply_gaussian_smoothing_toggle!(
        true,
        gauss_on,
        () -> (refresh_count[] += 1),
        layout_mode,
        () -> (ps_count[] += 1),
        msg -> push!(statuses, msg),
    )

    @test enabled
    @test gauss_on[]
    @test refresh_count[] == 1
    @test ps_count[] == 0
    @test last(statuses) == "Gaussian smoothing enabled."

    layout_mode[] = :power_spectrum
    enabled = MANTA._apply_gaussian_smoothing_toggle!(
        false,
        gauss_on,
        () -> (refresh_count[] += 1),
        layout_mode,
        () -> (ps_count[] += 1),
        msg -> push!(statuses, msg),
    )

    @test !enabled
    @test !gauss_on[]
    @test refresh_count[] == 2
    @test ps_count[] == 1
    @test last(statuses) == "Gaussian smoothing disabled."
end

@testset "integration: manta smoke and errors" begin
    mktempdir() do tmp
        cube_path = joinpath(tmp, "cube3d.fits")
        cube = reshape(Float32.(1:60), 3, 4, 5)
        FITS(cube_path, "w") do f
            write(f, cube)
        end

        fig = MANTA.manta(
            cube_path;
            activate_gl = false,
            display_fig = false,
            save_dir = tmp,
            settings_path = joinpath(tmp, "state.toml"),
            figsize = (800, 500),
        )
        @test fig isa Makie.Figure
        MANTA.forget!(fig)

        rgb_path = joinpath(tmp, "rgb_stack.fits")
        rgb_stack = zeros(Float32, 2, 3, 3)
        rgb_stack[:, :, 1] .= 1
        FITS(rgb_path, "w") do f
            write(f, rgb_stack)
        end
        fig_rgb_path = MANTA.manta(
            rgb_path;
            rgb = true,
            activate_gl = false,
            display_fig = false,
            figsize = (500, 360),
        )
        @test fig_rgb_path isa Makie.Figure
        MANTA.forget!(fig_rgb_path)

        hpix_ppv_path = joinpath(tmp, "healpix_ppv.fits")
        hpix_ppv = reshape(Float32.(1:48), 12, 4)
        FITS(hpix_ppv_path, "w") do f
            write(f, hpix_ppv)
        end
        fig_hpix = MANTA.manta_healpix_cube(
            hpix_ppv_path;
            activate_gl = false,
            display_fig = false,
            save_dir = tmp,
            nx = 100,
            ny = 50,
            figsize = (800, 560),
        )
        @test fig_hpix isa Makie.Figure
        MANTA.forget!(fig_hpix)

        fig_hpix_via_manta = MANTA.manta(
            hpix_ppv_path;
            activate_gl = false,
            display_fig = false,
            save_dir = tmp,
            nx = 100,
            ny = 50,
            figsize = (800, 560),
        )
        @test fig_hpix_via_manta isa Makie.Figure
        MANTA.forget!(fig_hpix_via_manta)

        hpix_map_path = joinpath(tmp, "healpix_map.fits")
        hpix_map = HealpixMap{Float64,RingOrder,Vector{Float64}}(collect(1.0:12.0))
        Healpix.saveToFITS(hpix_map, hpix_map_path; unit = "K")
        fig_map = MANTA.manta_healpix(
            hpix_map_path;
            activate_gl = false,
            display_fig = false,
            save_dir = tmp,
            nx = 100,
            ny = 50,
            figsize = (800, 560),
        )
        @test fig_map isa Makie.Figure
        MANTA.forget!(fig_map)

        fig_map_via_manta = MANTA.manta(
            hpix_map_path;
            activate_gl = false,
            display_fig = false,
            save_dir = tmp,
            nx = 100,
            ny = 50,
            figsize = (800, 560),
        )
        @test fig_map_via_manta isa Makie.Figure
        MANTA.forget!(fig_map_via_manta)

        missing = joinpath(tmp, "does_not_exist.fits")
        @test_throws MANTA.FileNotFoundError MANTA.manta(missing; activate_gl = false, display_fig = false)

        image2d_path = joinpath(tmp, "image2d.fits")
        FITS(image2d_path, "w") do f
            write(f, reshape(Float32.(1:12), 3, 4))
        end
        fig_2d = MANTA.manta(
            image2d_path;
            activate_gl = false,
            display_fig = false,
            save_dir = tmp,
            figsize = (700, 450),
        )
        @test fig_2d isa Makie.Figure
        MANTA.forget!(fig_2d)

        fig_2d_direct = MANTA.manta(
            reshape(Float32.(1:12), 3, 4);
            activate_gl = false,
            display_fig = false,
            figsize = (700, 450),
        )
        @test fig_2d_direct isa Makie.Figure
        MANTA.forget!(fig_2d_direct)

        bad_path = joinpath(tmp, "cube4d.fits")
        FITS(bad_path, "w") do f
            write(f, reshape(Float32.(1:24), 2, 3, 2, 2))
        end
        err = try
            MANTA.manta(bad_path; activate_gl = false, display_fig = false)
            nothing
        catch e
            e
        end
        @test err isa MANTA.DatasetShapeError
        @test occursin("4 dimensions", sprint(showerror, err))
    end
end

@testset "integration: spectrum y-limits inline state round-trip" begin
    # Persisted spectrum y-axis policy must replay through apply_inline_state!
    # → _restore_spec_ylimits! without breaking viewer construction. We capture
    # logs so a throw inside the restore (which _view_cube swallows with a
    # warning) still fails the test.
    cube = reshape(Float32.(1:60), 3, 4, 5)

    for src in ("manual", "contrast", "auto")
        logger = Test.TestLogger(min_level = Base.CoreLogging.BelowMinLevel)
        fig = Base.CoreLogging.with_logger(logger) do
            MANTA.manta(cube;
                state = Dict{String,Any}(
                    "spec_ylimits_source" => src,
                    "spec_ymin" => 2.0,
                    "spec_ymax" => 8.0,
                ),
                activate_gl = false,
                display_fig = false,
                figsize = (700, 450),
            )
        end
        @test fig isa Makie.Figure
        @test !any(r -> occursin("Failed to apply inline", string(r.message)),
                   logger.logs)
        MANTA.forget!(fig)
    end
end

@testset "dispatch: manta on datasets" begin
    # Cube dispatch (in-memory). The full interactive viewer now runs even
    # without a backing FITS file. Headless flags keep it CI-safe: no GL
    # context, no window display, just figure construction + return.
    cube_ds = MANTA.load_dataset(rand(Float32, 6, 6, 4))
    @test cube_ds isa MANTA.CubeDataset
    fig_cube = MANTA.manta(cube_ds; activate_gl = false, display_fig = false)
    @test fig_cube isa Makie.Figure

    # Same outcome via the user-facing 3D-array entry point — routes through
    # load_dataset → CubeDataset → _view_cube.
    fig_arr = MANTA.manta(rand(Float32, 6, 6, 4);
                          activate_gl = false, display_fig = false)
    @test fig_arr isa Makie.Figure

    # `view_cube` is the exported alias for `_view_cube` and accepts the same
    # CubeDataset directly.
    fig_vc = MANTA.view_cube(cube_ds; activate_gl = false, display_fig = false)
    @test fig_vc isa Makie.Figure
    @test MANTA.view_cube === MANTA._view_cube

    hpix_map = HealpixMap{Float64,RingOrder,Vector{Float64}}(collect(1.0:12.0))
    hpix_map_ds = MANTA.load_dataset(hpix_map)
    @test hpix_map_ds isa MANTA.HealpixMapDataset
    fig_hpix_map = MANTA.manta(hpix_map_ds;
                               activate_gl = false, display_fig = false,
                               nx = 60, ny = 30, figsize = (500, 320))
    @test fig_hpix_map isa Makie.Figure
    MANTA.forget!(fig_hpix_map)

    hpix_cube_ds = MANTA.HealpixCubeDataset(reshape(Float32.(1:48), 12, 4);
                                            nside = 1,
                                            source_id = "synthetic_hpix_cube")
    fig_hpix_cube = MANTA.manta(hpix_cube_ds;
                                activate_gl = false, display_fig = false,
                                nx = 60, ny = 30, figsize = (500, 360))
    @test fig_hpix_cube isa Makie.Figure
    MANTA.forget!(fig_hpix_cube)

    # VectorDataset now has a dedicated 1D viewer.
    vds = MANTA.load_dataset(rand(Float32, 16))
    @test vds isa MANTA.VectorDataset
    fig_vec = MANTA.manta(vds; activate_gl = false, display_fig = false,
                          figsize = (700, 450))
    @test fig_vec isa Makie.Figure
    MANTA.forget!(fig_vec)
end

@testset "dispatch: vector viewer" begin
    # ---- Pure helpers (no Makie state required) ----
    @testset "selection helpers" begin
        xs = Float64.(1:10)
        # Full range when sel === nothing.
        r = MANTA._vector_selection_indices(xs, nothing)
        @test r === 1:10
        # Inclusive bounds on a monotonic axis.
        r = MANTA._vector_selection_indices(xs, (3.0, 7.0))
        @test r == 3:7
        # Out-of-range high bound clamps to length.
        r = MANTA._vector_selection_indices(xs, (5.0, 100.0))
        @test r == 5:10
        # Empty intersection → empty range, NOT an error.
        r = MANTA._vector_selection_indices(xs, (100.0, 200.0))
        @test isempty(r)
        # Empty input vector → empty range, never throws.
        r = MANTA._vector_selection_indices(Float64[], (0.0, 1.0))
        @test isempty(r)
        # Non-monotonic fallback: pick the first/last matching index.
        xs2 = [1.0, 5.0, 2.0, 8.0, 3.0]
        r = MANTA._vector_selection_indices(xs2, (2.0, 5.0))
        @test r == 2:5
    end

    @testset "stats helpers" begin
        s = MANTA._vector_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        @test s.n == 5
        @test s.n_finite == 5
        @test s.n_nan == 0
        @test s.min == 1.0
        @test s.max == 5.0
        @test s.mean ≈ 3.0
        @test s.median == 3.0
        # std uses the corrected (n-1) estimator.
        @test s.std ≈ sqrt(2.5)

        s2 = MANTA._vector_stats([1.0, NaN, 3.0])
        @test s2.n == 3
        @test s2.n_finite == 2
        @test s2.n_nan == 1
        @test s2.min == 1.0
        @test s2.max == 3.0

        # All-NaN vector: finite stats degrade to NaN, no exception.
        s3 = MANTA._vector_stats([NaN, NaN])
        @test s3.n_finite == 0
        @test isnan(s3.mean)
        @test isnan(s3.std)
        @test isnan(s3.median)
    end

    @testset "parse range" begin
        ok, manual, bounds, _ = MANTA._parse_vector_range("", ""; fallback = (0.0, 1.0))
        @test ok && !manual
        @test bounds == (0.0, 1.0)

        ok, manual, bounds, _ = MANTA._parse_vector_range("1", ""; fallback = (0.0, 1.0))
        @test !ok && !manual

        ok, manual, bounds, _ = MANTA._parse_vector_range("abc", "5"; fallback = (0.0, 1.0))
        @test !ok

        ok, manual, bounds, _ = MANTA._parse_vector_range("5", "1"; fallback = (0.0, 1.0))
        @test ok && manual
        # Swapped automatically.
        @test bounds == (1.0, 5.0)

        ok, manual, bounds, _ = MANTA._parse_vector_range("2", "2"; fallback = (0.0, 1.0))
        @test ok && manual
        @test bounds[1] < 2.0 < bounds[2]
    end

    # ---- Figure creation (headless) ----
    @testset "view from VectorDataset" begin
        ys = Float32.(sin.(range(0.0, 4π; length = 64)))
        vds = MANTA.VectorDataset(ys;
            axis_label = "index",
            unit_label = "K",
            source_id  = "synthetic_vec",
        )
        fig = MANTA.manta(vds;
            activate_gl = false, display_fig = false,
            figsize = (640, 420))
        @test fig isa Makie.Figure
        MANTA.forget!(fig)

        # AbstractVector bridge: bypass explicit dataset construction.
        fig2 = MANTA.manta(collect(1.0:32.0);
            activate_gl = false, display_fig = false,
            figsize = (640, 420))
        @test fig2 isa Makie.Figure
        MANTA.forget!(fig2)

        # Log scales must not crash on a positive vector.
        fig3 = MANTA.manta(collect(1.0:32.0);
            xscale = :lin, yscale = :log10,
            activate_gl = false, display_fig = false,
            figsize = (640, 420))
        @test fig3 isa Makie.Figure
        MANTA.forget!(fig3)
    end

    @testset "view from 1D FITS file" begin
        mktempdir() do tmp
            path = joinpath(tmp, "vec1d.fits")
            FITS(path, "w") do f
                keys = String["BUNIT", "CTYPE1", "CRVAL1", "CRPIX1", "CDELT1", "CUNIT1"]
                vals = Any["Jy",    "FREQ",   1.0e9,    1.0,       1.0e6,    "Hz"]
                comms = String["",   "",       "",       "",        "",       ""]
                hdr = FITSIO.FITSHeader(keys, vals, comms)
                write(f, Float32.(collect(1:16)); header = hdr)
            end
            ds = MANTA.load_dataset(path)
            @test ds isa MANTA.VectorDataset
            @test ds.wcs !== nothing
            @test ds.unit_label == "Jy"
            fig = MANTA.manta(ds;
                activate_gl = false, display_fig = false,
                figsize = (640, 420))
            @test fig isa Makie.Figure
            MANTA.forget!(fig)
        end
    end

    @testset "CSV export round-trip" begin
        ys = Float64[1.0, 2.0, 3.0, 4.0, NaN]
        vds = MANTA.VectorDataset(ys;
            axis_label = "index", unit_label = "K",
            source_id = "csv_export_test")
        xs, _ = MANTA._vector_x_axis(vds)
        mktempdir() do tmp
            out = joinpath(tmp, "vec.csv")
            MANTA._write_vector_csv(out, xs, ys, 1:length(ys),
                (source_id = vds.source_id,
                 axis_label = vds.axis_label,
                 unit_label = vds.unit_label))
            txt = read(out, String)
            # Comment header with provenance.
            @test occursin("# source=csv_export_test", txt)
            @test occursin("# axis_label=index", txt)
            @test occursin("# unit_label=K", txt)
            # CSV body header is the literal "x,y".
            @test occursin("x,y", txt)
            # NaN values are written explicitly so external tools can detect
            # missing samples.
            @test occursin("NaN", txt)
            # The four finite rows are all present.
            for v in ("1.0", "2.0", "3.0", "4.0")
                @test occursin(v, txt)
            end
        end
    end
end

@testset "cube compare slices use independent indices" begin
    data = reshape(Float32.(1:16), 2, 2, 4)
    cmp  = reshape(Float32.(101:116), 2, 2, 4)

    axis = Observable(3)
    idx = Observable(1)
    compare_idx = Observable(3)

    bundle = MANTA._cube_slice_pipeline_bundle(;
        data,
        siz = size(data),
        wcs = MANTA.read_simple_wcs(Dict{String,Any}(), 3),
        axis,
        idx,
        compare_idx,
        gauss_on = Observable(false),
        sigma = Observable(1.5f0),
        compare_data = Observable{Any}(cmp),
        compare_mode = Observable(:B),
        view_product = Observable(:slice),
        mask_bits_obs = Observable{Union{Nothing,BitArray{3}}}(nothing),
        moment_order = Observable(0),
        img_scale_mode = Observable(:lin),
        moment_threshold = 0.0,
        moment_nsigma = nothing,
        moment_channels = nothing,
    )

    @test bundle.slice_raw[] == MANTA.get_slice_copy(data, 3, 1)
    @test bundle.compare_slice_raw[] == MANTA.get_slice_copy(cmp, 3, 3)

    idx[] = 2
    @test bundle.slice_raw[] == MANTA.get_slice_copy(data, 3, 2)
    @test bundle.compare_slice_raw[] == MANTA.get_slice_copy(cmp, 3, 3)

    compare_idx[] = 4
    @test bundle.slice_raw[] == MANTA.get_slice_copy(data, 3, 2)
    @test bundle.compare_slice_raw[] == MANTA.get_slice_copy(cmp, 3, 4)
end

@testset "SlicePipelineBundle display downsampling preserves native coordinates" begin
    data = reshape(Float32.(1:(4 * 6 * 2)), 4, 6, 2)

    bundle = MANTA._cube_slice_pipeline_bundle(;
        data,
        siz = size(data),
        wcs = MANTA.read_simple_wcs(Dict{String,Any}(), 3),
        axis = Observable(3),
        idx = Observable(1),
        compare_idx = Observable(1),
        gauss_on = Observable(false),
        sigma = Observable(1.5f0),
        compare_data = Observable{Any}(nothing),
        compare_mode = Observable(:B),
        view_product = Observable(:slice),
        mask_bits_obs = Observable{Union{Nothing,BitArray{3}}}(nothing),
        moment_order = Observable(0),
        img_scale_mode = Observable(:lin),
        moment_threshold = 0.0,
        moment_nsigma = nothing,
        moment_channels = nothing,
        display_max_pixels = 2,
    )

    @test bundle.plot_stride[] == 3
    @test bundle.plot_x[] == Float32[2.0, 5.0]
    @test bundle.plot_y[] == Float32[2.0, 4.0]
    @test size(bundle.slice_plot[]) == (2, 2)
    @test bundle.slice_plot[] == permutedims(MANTA.downsample_block_mean(bundle.slice_disp[], 3))
end

@testset "SlicePipelineBundle moment cache tracks mask changes" begin
    data = fill(1f0, 2, 2, 2)
    mask_obs = Observable{Union{Nothing,BitArray{3}}}(nothing)

    bundle = MANTA._cube_slice_pipeline_bundle(;
        data,
        siz = size(data),
        wcs = MANTA.read_simple_wcs(Dict{String,Any}(), 3),
        axis = Observable(3),
        idx = Observable(1),
        compare_idx = Observable(1),
        gauss_on = Observable(false),
        sigma = Observable(1.5f0),
        compare_data = Observable{Any}(nothing),
        compare_mode = Observable(:B),
        view_product = Observable(:moment),
        mask_bits_obs = mask_obs,
        moment_order = Observable(0),
        img_scale_mode = Observable(:lin),
        moment_threshold = 0.0,
        moment_nsigma = nothing,
        moment_channels = nothing,
    )

    @test bundle.moment_raw[] == fill(2f0, 2, 2)

    mask = trues(2, 2, 2)
    mask[:, :, 2] .= false
    mask_obs[] = mask

    @test bundle.moment_raw[] == fill(1f0, 2, 2)
    @test length(bundle._moment_cache) == 2
end

@testset "SlicePipelineBundle OOB idx clamps and warns" begin
    # 2×2×4 cube: valid axis-3 indices are 1..4.
    data = reshape(Float32.(1:16), 2, 2, 4)
    cmp  = reshape(Float32.(101:116), 2, 2, 4)

    axis        = Observable(3)
    idx         = Observable(1)
    compare_idx = Observable(1)

    make_bundle() = MANTA._cube_slice_pipeline_bundle(;
        data,
        siz         = size(data),
        wcs         = MANTA.read_simple_wcs(Dict{String,Any}(), 3),
        axis,
        idx,
        compare_idx,
        gauss_on      = Observable(false),
        sigma         = Observable(1.5f0),
        compare_data  = Observable{Any}(cmp),
        compare_mode  = Observable(:B),
        view_product  = Observable(:slice),
        mask_bits_obs = Observable{Union{Nothing,BitArray{3}}}(nothing),
        moment_order  = Observable(0),
        img_scale_mode = Observable(:lin),
        moment_threshold = 0.0,
        moment_nsigma    = nothing,
        moment_channels  = nothing,
    )

    bundle = make_bundle()

    # --- idx too high: clamp to last slice and emit a warning ---
    @test_logs (:warn, r"idx=5.*out of bounds.*axis=3") begin
        idx[] = 5          # OOB: size along axis 3 is 4
    end
    @test bundle.slice_raw[] == MANTA.get_slice_copy(data, 3, 4)  # clamped to 4

    # --- idx too low: clamp to first slice and emit a warning ---
    @test_logs (:warn, r"idx=0.*out of bounds.*axis=3") begin
        idx[] = 0
    end
    @test bundle.slice_raw[] == MANTA.get_slice_copy(data, 3, 1)  # clamped to 1

    # --- compare_idx OOB warns separately ---
    idx[] = 1   # reset primary (in-bounds, no warning expected)
    @test_logs (:warn, r"compare_idx=9.*out of bounds.*axis=3") begin
        compare_idx[] = 9
    end
    @test bundle.compare_slice_raw[] == MANTA.get_slice_copy(cmp, 3, 4)

    # --- in-bounds indices produce no warning ---
    @test_logs min_level=Base.CoreLogging.Warn begin
        idx[]         = 3
        compare_idx[] = 2
    end
    @test bundle.slice_raw[]         == MANTA.get_slice_copy(data, 3, 3)
    @test bundle.compare_slice_raw[] == MANTA.get_slice_copy(cmp,  3, 2)

    # --- mask_slice OOB also clamps and warns ---
    mask = trues(2, 2, 4)
    mask_obs = Observable{Union{Nothing,BitArray{3}}}(mask)

    bundle2 = MANTA._cube_slice_pipeline_bundle(;
        data,
        siz         = size(data),
        wcs         = MANTA.read_simple_wcs(Dict{String,Any}(), 3),
        axis,
        idx         = Observable(7),   # OOB from construction
        compare_idx = Observable(1),
        gauss_on      = Observable(false),
        sigma         = Observable(1.5f0),
        compare_data  = Observable{Any}(nothing),
        compare_mode  = Observable(:B),
        view_product  = Observable(:slice),
        mask_bits_obs = mask_obs,
        moment_order  = Observable(0),
        img_scale_mode = Observable(:lin),
        moment_threshold = 0.0,
        moment_nsigma    = nothing,
        moment_channels  = nothing,
    )
    # mask_slice should have clamped to slice 4 without throwing
    @test bundle2.mask_slice[] == collect(@view mask[:, :, 4])
end

@testset "SlicePipelineBundle: slice_raw is a zero-copy view" begin
    # Hot-path optimisation: slice_raw must NOT copy the cube — it exposes a
    # view aliasing the parent. The copy only materialises downstream where a
    # transformation needs it (e.g. slice_disp via apply_scale_display).
    data = reshape(Float32.(1:24), 2, 3, 4)

    axis = Observable(3)
    idx = Observable(2)
    compare_idx = Observable(1)
    cmp = reshape(Float32.(101:124), 2, 3, 4)

    bundle = MANTA._cube_slice_pipeline_bundle(;
        data,
        siz = size(data),
        wcs = MANTA.read_simple_wcs(Dict{String,Any}(), 3),
        axis,
        idx,
        compare_idx,
        gauss_on = Observable(false),
        sigma = Observable(1.5f0),
        compare_data = Observable{Any}(cmp),
        compare_mode = Observable(:A),
        view_product = Observable(:slice),
        mask_bits_obs = Observable{Union{Nothing,BitArray{3}}}(nothing),
        moment_order = Observable(0),
        img_scale_mode = Observable(:lin),
        moment_threshold = 0.0,
        moment_nsigma = nothing,
        moment_channels = nothing,
    )

    s = bundle.slice_raw[]
    @test s isa SubArray              # not a freshly allocated Array
    @test parent(s) === data          # aliases the cube, no copy
    @test s == MANTA.get_slice_view(data, 3, 2)

    # compare_slice_raw is likewise a view onto the comparison cube.
    cs = bundle.compare_slice_raw[]
    @test cs isa SubArray
    @test parent(cs) === cmp

    # The display array is still an independent, scale-applied Float32 buffer
    # (the copy happens here, not in slice_raw).
    d = bundle.slice_disp[]
    @test d isa Matrix{Float32}
    @test d !== s
    @test !(d isa SubArray)
end

@testset "SlicePipelineBundle: smoothing can toggle after zero-copy view init" begin
    data = reshape(Float32.(1:24), 2, 3, 4)
    gauss_on = Observable(false)

    bundle = MANTA._cube_slice_pipeline_bundle(;
        data,
        siz = size(data),
        wcs = MANTA.read_simple_wcs(Dict{String,Any}(), 3),
        axis = Observable(3),
        idx = Observable(2),
        compare_idx = Observable(1),
        gauss_on,
        sigma = Observable(1.5f0),
        compare_data = Observable{Any}(nothing),
        compare_mode = Observable(:A),
        view_product = Observable(:slice),
        mask_bits_obs = Observable{Union{Nothing,BitArray{3}}}(nothing),
        moment_order = Observable(0),
        img_scale_mode = Observable(:lin),
        moment_threshold = 0.0,
        moment_nsigma = nothing,
        moment_channels = nothing,
    )

    @test bundle.slice_proc[] isa SubArray
    @test eltype(bundle.slice_proc) === AbstractMatrix{Float32}

    gauss_on[] = true

    @test bundle.slice_proc[] isa Matrix{Float32}
    @test bundle.base_slice_proc[] isa Matrix{Float32}
    @test bundle.compare_slice_proc[] isa Matrix{Float32}
end

@testset "view_cube: lazy cube smoke + prefetch wiring" begin
    # A memory-mapped (LazyFITSCube) cube must drive the full viewer headlessly.
    # This exercises the `if data isa LazyFITSCube` prefetch-wiring branch added
    # to _view_cube (on(idx)/on(axis) registration) without needing a GL context.
    mktempdir() do tmp
        path = joinpath(tmp, "lazy_cube.fits")
        FITS(path, "w") do f
            write(f, Float32[i + 10j + 100k for i in 1:5, j in 1:6, k in 1:7])
        end
        ds = MANTA.load_fits(path; lazy = true)
        @test ds isa MANTA.CubeDataset
        @test ds.data isa MANTA.LazyFITSCube

        fig = MANTA.view_cube(ds; activate_gl = false, display_fig = false,
                              figsize = (700, 450))
        @test fig isa Makie.Figure
        MANTA.forget!(fig)
    end
end

@testset "manta_batch" begin
    mktempdir() do tmp
        # Build two minimal 3D FITS cubes.
        paths = map(1:2) do i
            p = joinpath(tmp, "cube$(i).fits")
            FITS(p, "w") do f
                write(f, reshape(Float32.(1:(3*4*5)), 3, 4, 5) .* i)
            end
            p
        end

        # ── happy path: PNG (default) ────────────────────────────────────────
        out_paths = MANTA.manta_batch(
            paths;
            activate_gl = false,
            display_fig = false,
            figsize     = (400, 300),
        )
        @test length(out_paths) == 2
        for (p, out) in zip(paths, out_paths)
            @test isfile(out)
            @test endswith(out, ".png")
            # Default save_dir === dirname of each source file.
            @test dirname(out) == dirname(p)
        end

        # ── explicit save_dir ────────────────────────────────────────────────
        out_dir = joinpath(tmp, "exported")
        out2 = MANTA.manta_batch(
            paths;
            save_dir    = out_dir,
            activate_gl = false,
            display_fig = false,
            figsize     = (400, 300),
        )
        @test isdir(out_dir)
        @test length(out2) == 2
        for out in out2
            @test startswith(out, out_dir)
        end

        # ── prefix kwarg ─────────────────────────────────────────────────────
        out3 = MANTA.manta_batch(
            paths;
            save_dir    = out_dir,
            prefix      = "batch_",
            activate_gl = false,
            display_fig = false,
            figsize     = (400, 300),
        )
        for out in out3
            @test startswith(basename(out), "batch_")
        end

        # ── PDF format ───────────────────────────────────────────────────────
        out4 = MANTA.manta_batch(
            paths;
            format      = :pdf,
            save_dir    = out_dir,
            activate_gl = false,
            display_fig = false,
            figsize     = (400, 300),
        )
        @test length(out4) == 2
        for out in out4
            @test endswith(out, ".pdf")
        end

        # ── bad format → ArgumentError immediately ───────────────────────────
        @test_throws ArgumentError MANTA.manta_batch(
            paths; format = :bmp, activate_gl = false, display_fig = false)

        # ── missing file → warning, continues; other files succeed ───────────
        paths_with_bad = [paths[1], joinpath(tmp, "ghost.fits"), paths[2]]
        out5 = MANTA.manta_batch(
            paths_with_bad;
            save_dir    = out_dir,
            activate_gl = false,
            display_fig = false,
            figsize     = (400, 300),
        )
        # The two valid files still produce output; the bad one is silently
        # skipped (logged as @warn).
        @test length(out5) == 2

        # ── empty input → empty output, no error ─────────────────────────────
        out_empty = MANTA.manta_batch(String[])
        @test out_empty == String[]
    end
end

@testset "view_cube: respects dataset fields" begin
    # Build a CubeDataset with explicit unit_label/source_id and confirm
    # the viewer constructs a figure that uses them (we can't easily probe
    # the labels post-construction, but the call should succeed and return
    # a Makie.Figure).
    data = rand(Float32, 5, 4, 3)
    ds = MANTA.CubeDataset(data;
        axis_labels = ["RA", "Dec", "v"],
        unit_label  = "Jy/beam",
        source_id   = "synthetic_cube",
    )
    fig = MANTA.view_cube(ds; activate_gl = false, display_fig = false)
    @test fig isa Makie.Figure

    # Float64 input gets coerced to Float32 by `as_float32`, without a crash.
    ds64 = MANTA.CubeDataset(rand(Float64, 4, 4, 3))
    fig64 = MANTA.view_cube(ds64; activate_gl = false, display_fig = false)
    @test fig64 isa Makie.Figure
end

@testset "cube undo/redo snapshot coverage" begin
    # Guards the audited gap: the cube viewer's undo snapshot must capture more
    # than navigation/contrast — the mask source, the region selection and the
    # moment product have to round-trip too, or a mask/region/moment applied by
    # mistake would not be undoable.
    base = (;
        axis = 3, idx = 1, compare_idx = 1,
        cmap_name = :viridis, invert_cmap = false, img_scale_mode = :lin,
        use_manual = false, clims_manual = (0f0, 1f0),
        mask_source = MANTA.NoMaskSource(),
        region_uvs = Tuple{Int,Int}[],
        region_p0 = (0f0, 0f0), region_p1 = (0f0, 0f0),
        selection_mode = :point, view_product = :slice, moment_order = 0,
    )
    mk(; kw...) = MANTA._cube_undo_snapshot(; merge(base, (; kw...))...)

    # 1. The audited fields are present in the snapshot.
    s0 = mk()
    @test s0 isa MANTA._CubeUndoSnapshot
    for f in (:mask_source, :region_uvs, :selection_mode, :view_product, :moment_order)
        @test hasproperty(s0, f)
    end

    # 2. Type stability across concrete MaskSource subtypes — the abstract-field
    #    fix. A single UndoRedoStack{T} must accept every snapshot.
    s_thr = mk(; mask_source = MANTA.ThresholdSource(:ge, 1.0, 0.0))
    @test typeof(s_thr) === typeof(s0) === MANTA._CubeUndoSnapshot

    # 3. NaN-safe dedup: identical empty-selection snapshots compare equal even
    #    though the live drag points are Point2f(NaN, NaN).
    @test mk() == mk()
    @test MANTA._undo_clean_pt((NaN32, NaN32)) == (0f0, 0f0)
    @test MANTA._undo_clean_pt((2.5f0, -1.0f0)) == (2.5f0, -1.0f0)

    # 4. Round-trip through the real stack restores mask + region + moment.
    stack = MANTA.UndoRedoStack(s0)
    s1 = mk(;
        mask_source = MANTA.ThresholdSource(:ge, 2.0, 0.0),
        region_uvs = [(1, 1), (1, 2)], selection_mode = :box,
        region_p0 = (1f0, 1f0), region_p1 = (2f0, 2f0),
        view_product = :moment, moment_order = 1,
    )
    MANTA.register_state!(stack, s1)
    @test length(stack) == 2

    back = MANTA.undo!(stack)
    @test back.mask_source isa MANTA.NoMaskSource
    @test isempty(back.region_uvs)
    @test back.view_product === :slice
    @test back.moment_order == 0

    fwd = MANTA.redo!(stack)
    @test fwd.mask_source isa MANTA.ThresholdSource
    @test fwd.region_uvs == [(1, 1), (1, 2)]
    @test fwd.selection_mode === :box
    @test fwd.view_product === :moment
    @test fwd.moment_order == 1

    # 5. Dedup still collapses a genuinely-unchanged re-register.
    MANTA.register_state!(stack, fwd)
    @test length(stack) == 2
end
