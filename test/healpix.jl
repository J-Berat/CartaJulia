# path: test/healpix.jl
# Testsets covering HEALPix geometry helpers, projected regions, and the
# headless viewer regression suite (scale modes, invert, vmin/vmax, dispatch,
# PPV cube, panels, file-based paths).

@testset "healpix: mollweide graticule geometry" begin
    for lon in (-120, -30, 0, 45, 150), lat in (-60, -15, 0, 35, 70)
        p = MANTA.mollweide_lonlat_to_xy(lon, lat)
        @test p !== nothing
        ll = MANTA.mollweide_xy_to_lonlat(p[1], p[2])
        @test ll !== nothing
        lon2, lat2 = ll
        @test isapprox(lon2, lon; atol=1e-4)
        @test isapprox(lat2, lat; atol=1e-4)
    end

    @testset "pixel index cache and ordering" begin
        res = Healpix.Resolution(2)
        idx1 = MANTA.mollweide_pixel_index(res, 32, 16)
        idx2 = MANTA.mollweide_pixel_index(res, 32, 16)
        @test idx1 == idx2
        @test idx1 !== idx2

        cached1 = MANTA._cached_mollweide_pixel_index(res, 32, 16)
        cached2 = MANTA._cached_mollweide_pixel_index(res, 32, 16)
        @test cached1 === cached2
        @test_throws ArgumentError MANTA.mollweide_pixel_index(res, 0, 16)

        ring_map = HealpixMap{Float64,RingOrder,Vector{Float64}}(collect(1.0:48.0))
        nested_map = Healpix.ring2nest(ring_map)
        @test nested_map isa HealpixMap{Float64,NestedOrder,Vector{Float64}}
        @test isequal(
            MANTA.mollweide_grid(ring_map; nx = 32, ny = 16),
            MANTA.mollweide_grid(nested_map; nx = 32, ny = 16),
        )
    end
end

@testset "healpix: projected regions" begin
    grid = Int32[
        0 1 1 2
        3 3 4 0
        5 6 6 7
    ]
    box_ips = MANTA.projected_region_ipix(grid, -2, -1, 2, 1, :box)
    @test box_ips == [1, 2, 3, 4, 5, 6, 7]

    circle_ips = MANTA.projected_region_ipix(grid, 0, 0, 1, 0, :circle)
    @test all(>(0), circle_ips)
    @test issorted(circle_ips)

    vals = Float32[10, 20, NaN32, 40]
    @test MANTA.healpix_region_mean(vals, [1, 2, 3]) == 15f0

    cube = Float32[
        1 10 100
        3 30 300
        NaN 40 400
    ]
    spec = MANTA.healpix_region_mean_spectrum(cube, [1, 2, 3], 3)
    @test spec == Float32[2, 80 / 3, 800 / 3]
end

# ----------------------------------------------------------------------------
# HEALPix viewer regression tests (headless, activate_gl=false)
# ----------------------------------------------------------------------------
@testset "healpix: viewer headless regression" begin
    # ---- Unit: valid_healpix_npix ----
    @testset "valid_healpix_npix" begin
        # Powers of 2 → correct nside
        @test MANTA.valid_healpix_npix(12)    == 1   # 12·1²
        @test MANTA.valid_healpix_npix(48)    == 2   # 12·2²
        @test MANTA.valid_healpix_npix(192)   == 4   # 12·4²
        @test MANTA.valid_healpix_npix(768)   == 8
        @test MANTA.valid_healpix_npix(3072)  == 16
        # Non-HEALPix counts → 0
        @test MANTA.valid_healpix_npix(0)     == 0
        @test MANTA.valid_healpix_npix(-12)   == 0
        @test MANTA.valid_healpix_npix(13)    == 0
        @test MANTA.valid_healpix_npix(36)    == 0   # 12·3² (nside=3 is not a power of 2)
        @test MANTA.valid_healpix_npix(1)     == 0
    end

    # ---- Unit: detect_velocity_axis ----
    @testset "detect_velocity_axis" begin
        mktempdir() do tmp
            # Write a 2D FITS with a VRAD spectral axis on axis 2.
            path = joinpath(tmp, "vrad_cube.fits")
            arr = reshape(Float32.(1:48), 12, 4)
            FITS(path, "w") do f
                keys = ["CTYPE1", "CTYPE2",
                        "CRVAL2", "CDELT2", "CRPIX2", "CUNIT2"]
                vals = Any["GLON-MOL", "VRAD",
                           2.0e5, 1.0e3, 1.0, "m/s"]
                comms = fill("", length(keys))
                hdr = FITSIO.FITSHeader(keys, vals, comms)
                write(f, arr; header = hdr)
            end
            result = MANTA.detect_velocity_axis(path, 2)
            @test result !== nothing
            ax, v0, dv, unit = result
            @test ax == 2
            @test isapprox(v0, (2.0e5 - (1.0 - 1) * 1.0e3) * 1e-3; atol = 1e-6)   # m/s → km/s
            @test isapprox(dv, 1.0; atol = 1e-6)                                     # 1e3 m/s = 1 km/s
            @test unit == "km/s"

            # A FITS without any recognized spectral CTYPE → nothing
            path2 = joinpath(tmp, "no_spectral.fits")
            FITS(path2, "w") do f
                write(f, arr)
            end
            @test MANTA.detect_velocity_axis(path2, 2) === nothing

            # Non-existent file → nothing (graceful)
            @test MANTA.detect_velocity_axis(joinpath(tmp, "ghost.fits"), 2) === nothing
        end
    end

    # ---- HEALPix map viewer: scale modes ----
    @testset "manta_healpix scale modes" begin
        hpix_map = HealpixMap{Float64,RingOrder,Vector{Float64}}(collect(1.0:48.0))
        for sc in (:lin, :log10, :ln)
            fig = MANTA.manta_healpix(hpix_map;
                activate_gl = false, display_fig = false,
                scale = sc, nx = 60, ny = 30, figsize = (500, 280))
            @test fig isa Makie.Figure
            MANTA.forget!(fig)
        end
    end

    # ---- HEALPix map viewer: invert + vmin/vmax ----
    @testset "manta_healpix invert and vmin/vmax" begin
        hpix_map = HealpixMap{Float64,RingOrder,Vector{Float64}}(collect(1.0:48.0))
        fig_inv = MANTA.manta_healpix(hpix_map;
            activate_gl = false, display_fig = false,
            invert = true, nx = 60, ny = 30, figsize = (500, 280))
        @test fig_inv isa Makie.Figure
        MANTA.forget!(fig_inv)

        fig_clim = MANTA.manta_healpix(hpix_map;
            activate_gl = false, display_fig = false,
            vmin = 5.0, vmax = 40.0, nx = 60, ny = 30, figsize = (500, 280))
        @test fig_clim isa Makie.Figure
        MANTA.forget!(fig_clim)
    end

    # ---- HEALPix map viewer: dataset dispatch ----
    @testset "manta_healpix dataset dispatch" begin
        hpix_map = HealpixMap{Float64,RingOrder,Vector{Float64}}(collect(1.0:12.0))
        ds = MANTA.load_dataset(hpix_map)
        @test ds isa MANTA.HealpixMapDataset
        fig = MANTA.manta(ds;
            activate_gl = false, display_fig = false,
            nx = 40, ny = 20, figsize = (440, 240))
        @test fig isa Makie.Figure
        MANTA.forget!(fig)
    end

    # ---- HEALPix PPV cube viewer: scale modes and invert ----
    @testset "manta_healpix_cube scale modes and invert" begin
        cube_data = reshape(Float32.(1:192), 48, 4)   # nside=2, 4 velocity channels
        ds = MANTA.HealpixCubeDataset(cube_data;
            nside = 2, source_id = "hpix_cube_reg")
        for sc in (:lin, :log10, :ln)
            fig = MANTA.manta(ds;
                activate_gl = false, display_fig = false,
                scale = sc, nx = 60, ny = 30, figsize = (640, 380))
            @test fig isa Makie.Figure
            MANTA.forget!(fig)
        end
        fig_inv = MANTA.manta(ds;
            activate_gl = false, display_fig = false,
            invert = true, nx = 60, ny = 30, figsize = (640, 380))
        @test fig_inv isa Makie.Figure
        MANTA.forget!(fig_inv)
    end

    # ---- manta_healpix_cube: vmin/vmax manual contrast ----
    @testset "manta_healpix_cube vmin/vmax" begin
        cube_data = reshape(Float32.(1:192), 48, 4)
        ds = MANTA.HealpixCubeDataset(cube_data; nside = 2, source_id = "hpix_cube_clim")
        fig = MANTA.manta(ds;
            activate_gl = false, display_fig = false,
            vmin = 10.0, vmax = 150.0, nx = 60, ny = 30, figsize = (640, 380))
        @test fig isa Makie.Figure
        MANTA.forget!(fig)
    end

    # ---- manta_healpix_panels: two scalar panels ----
    @testset "manta_healpix_panels two panels" begin
        p1 = Float32.(collect(1.0:48.0))   # nside=2
        p2 = Float32.(collect(48.0:-1.0:1.0))
        fig = MANTA.manta_healpix_panels(p1, p2;
            titles    = ["Map A", "Map B"],
            cmaps     = [:inferno, :viridis],
            nx = 60, ny = 30, figsize = (700, 280),
            activate_gl = false, display_fig = false)
        @test fig isa Makie.Figure
        MANTA.forget!(fig)

        # Single-panel path also works.
        fig1 = MANTA.manta_healpix_panels(p1;
            nx = 60, ny = 30, figsize = (480, 280),
            activate_gl = false, display_fig = false)
        @test fig1 isa Makie.Figure
        MANTA.forget!(fig1)
    end

    # ---- manta_healpix_cube: file-based path (via manta dispatch) ----
    @testset "manta_healpix_cube file-based" begin
        mktempdir() do tmp
            path = joinpath(tmp, "hpix_ppv.fits")
            arr = reshape(Float32.(1:192), 48, 4)
            FITS(path, "w") do f
                write(f, arr)
            end
            fig = MANTA.manta(path;
                activate_gl = false, display_fig = false,
                nx = 60, ny = 30, figsize = (640, 380))
            @test fig isa Makie.Figure
            MANTA.forget!(fig)

            # manta_healpix_cube entry point directly
            fig2 = MANTA.manta_healpix_cube(path;
                activate_gl = false, display_fig = false,
                nx = 60, ny = 30, figsize = (640, 380))
            @test fig2 isa Makie.Figure
            MANTA.forget!(fig2)
        end
    end
end
