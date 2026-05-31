# path: test/loaders.jl
# Testsets covering dataset loading (in-memory and file paths), abstract type
# aliases, and the HDU kwarg on the eager FITS loader.

@testset "datasets: load_dataset (in-memory)" begin
    # 2D array → ImageDataset
    img32 = rand(Float32, 10, 20)
    ds_img = MANTA.load_dataset(img32)
    @test ds_img isa MANTA.ImageDataset
    @test eltype(ds_img.data) === Float32
    @test size(ds_img.data) == (10, 20)

    # 3D array → CubeDataset
    cube32 = rand(Float32, 10, 20, 30)
    ds_cube = MANTA.load_dataset(cube32)
    @test ds_cube isa MANTA.CubeDataset
    @test eltype(ds_cube.data) === Float32

    # Float64 input is preserved by the type alias rule of in-memory loader
    # (no implicit Float32 cast in this path — display-time conversion is the
    # viewer's responsibility via `as_float32`).
    cube64 = rand(Float64, 4, 5, 6)
    ds_cube64 = MANTA.load_dataset(cube64)
    @test ds_cube64 isa MANTA.CubeDataset
    @test eltype(ds_cube64.data) === Float64

    # NamedTuple → MultiChannelDataset
    Q = rand(Float32, 8, 8, 5)
    U = rand(Float32, 8, 8, 5)
    ds_mc = MANTA.load_dataset((Q = Q, U = U))
    @test ds_mc isa MANTA.MultiChannelDataset
    @test haskey(ds_mc.channels, :Q)
    @test haskey(ds_mc.channels, :U)
    @test ds_mc.kind === :cube

    # Idempotence: load_dataset(ds) === ds
    @test MANTA.load_dataset(ds_img) === ds_img

    # stable_source_id is deterministic per (typeof, size)
    sid1 = MANTA.stable_source_id(rand(Float32, 4, 5))
    sid2 = MANTA.stable_source_id(rand(Float32, 4, 5))
    @test sid1 == sid2
    @test sid1 != MANTA.stable_source_id(rand(Float32, 4, 6))

    # source_id override
    ds_named = MANTA.load_dataset(rand(Float32, 3, 3); source_id = "my_custom_id")
    @test ds_named.source_id == "my_custom_id"

    # Unsupported ndims → ArgumentError with a clear message.
    @test_throws ArgumentError MANTA.load_dataset(rand(Float32, 2, 2, 2, 2))
end

@testset "datasets: load_dataset (paths)" begin
    @test_throws MANTA.FileNotFoundError MANTA.load_dataset("does_not_exist.fits")
    @test_throws MANTA.FileNotFoundError MANTA.load_dataset("does_not_exist.h5")
    @test_throws MANTA.UnsupportedFormatError MANTA.load_dataset("unrecognised.txt")
end

@testset "abstract type rename" begin
    # AbstractMANTADataset is the new name; the carta alias must still resolve
    # to the same type to avoid breaking external code.
    @test MANTA.AbstractCartaDataset === MANTA.AbstractMANTADataset
    @test MANTA.ImageDataset <: MANTA.AbstractMANTADataset
    @test MANTA.CubeDataset <: MANTA.AbstractMANTADataset
    @test MANTA.HealpixMapDataset <: MANTA.AbstractMANTADataset
    @test MANTA.HealpixCubeDataset <: MANTA.AbstractMANTADataset
    @test MANTA.MultiChannelDataset <: MANTA.AbstractMANTADataset
end

# ----------------------------------------------------------------------------
# HDU selection (eager loader)
# ----------------------------------------------------------------------------
@testset "loader: hdu kwarg" begin
    mktempdir() do dir
        path = joinpath(dir, "img.fits")
        FITS(path, "w") do f
            write(f, Float32[i + j for i in 1:6, j in 1:5])
        end
        # Default hdu=1 still works.
        ds = MANTA.load_fits(path)
        @test ds isa MANTA.ImageDataset
        @test size(ds.data) == (6, 5)
        # Explicit hdu=1.
        ds1 = MANTA.load_fits(path; hdu = 1)
        @test size(ds1.data) == (6, 5)
        # Out-of-range HDU → structured error.
        @test_throws MANTA.MANTAError MANTA.load_fits(path; hdu = 9)
        # Negative HDU → InvalidArgumentError.
        @test_throws MANTA.InvalidArgumentError MANTA.load_fits(path; hdu = -1)
        # Auto-pick (hdu=0) still finds the first non-empty HDU.
        ds0 = MANTA.load_fits(path; hdu = 0)
        @test ds0 isa MANTA.ImageDataset
    end
end
