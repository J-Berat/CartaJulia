# path: test/lazy_hdf5.jl
# Testsets covering lazy HDF5 images/cubes, chunk-aligned slice caching, and
# lazy cube analytical helpers.

@testset "lazy HDF5: 2D + 3D" begin
    mktempdir() do dir
        path = joinpath(dir, "lazy.h5")
        img = Float32[i + 10j for i in 1:6, j in 1:5]
        cube = Float32[i + 10j + 100k for i in 1:4, j in 1:5, k in 1:6]
        h5open(path, "w") do f
            f["img"] = img
            f["cube", chunk = (2, 5, 2)] = cube
        end

        ds_img = MANTA.load_hdf5(path, "/img"; lazy = true)
        @test ds_img isa MANTA.ImageDataset
        @test ds_img.data isa MANTA.LazyHDF5Image
        @test size(ds_img.data) == size(img)
        @test ds_img.data[3, 2] == img[3, 2]
        @test collect(ds_img.data) == img
        @test get(ds_img.metadata, :lazy, false) === true

        ds_cube = MANTA.load_dataset("$(path):/cube"; lazy = true)
        @test ds_cube isa MANTA.CubeDataset
        @test ds_cube.data isa MANTA.LazyHDF5Cube
        @test size(ds_cube.data) == size(cube)
        @test get(ds_cube.metadata, :lazy, false) === true
        @test get(ds_cube.metadata, :hdf5_chunk, nothing) == (2, 5, 2)

        lc = ds_cube.data
        @test lc[1, 1, 1] == cube[1, 1, 1]

        s1 = MANTA.read_slice!(lc, 3, 1)
        @test s1 == cube[:, :, 1]
        @test lc.cache_axis == 3
        @test lc.cache_range == 1:2
        cached_block = lc.cache_data
        s2 = MANTA.read_slice!(lc, 3, 2)
        @test s2 == cube[:, :, 2]
        @test lc.cache_data === cached_block

        lc_pre = MANTA.LazyHDF5Cube{Float32}(path, "/cube", size(cube), (2, 5, 2))
        MANTA.prefetch_slice!(lc_pre, 3, 1)
        @test lc_pre.prefetch_axis == 3
        @test lc_pre.prefetch_range == 1:2
        @test lc_pre.prefetch_ch !== nothing
        sleep(0.1)
        s_pre = MANTA.read_slice!(lc_pre, 3, 1)
        @test s_pre == cube[:, :, 1]
        @test lc_pre.cache_range == 1:2
        @test lc_pre.prefetch_ch === nothing

        @test MANTA.prefetch_slice!(lc_pre, 3, 999) === nothing
        @test MANTA.prefetch_slice!(lc_pre, 9, 1) === nothing

        MANTA.prefetch_slice!(lc_pre, 3, 3)
        ch_first = lc_pre.prefetch_ch
        MANTA.prefetch_slice!(lc_pre, 3, 3)
        @test lc_pre.prefetch_ch === ch_first

        MANTA.prefetch_slice!(lc_pre, 1, 3)
        @test lc_pre.prefetch_axis == 1
        @test lc_pre.prefetch_range == 3:4
        @test lc_pre.prefetch_ch !== ch_first
        sleep(0.1)
        @test MANTA.read_slice!(lc_pre, 1, 3) == cube[3, :, :]

        s_axis1 = MANTA.read_slice!(lc, 1, 2)
        @test s_axis1 == cube[2, :, :]
        s_axis2 = MANTA.read_slice!(lc, 2, 3)
        @test s_axis2 == cube[:, 3, :]

        sp = MANTA.read_spectrum(lc, 2, 3)
        @test sp == vec(cube[2:2, 3:3, :])

        m0_lazy = MANTA.moment_map(lc, 3, 0)
        m0_eager = MANTA.moment_map(cube, 3, 0)
        @test m0_lazy == m0_eager

        reg_lazy = MANTA.mean_region_spectrum(lc, 3, [(1, 1), (2, 1)])
        reg_eager = MANTA.mean_region_spectrum(cube, 3, [(1, 1), (2, 1)])
        @test reg_lazy == reg_eager
    end
end

@testset "lazy HDF5: prefetch_adjacent! direction" begin
    mktempdir() do dir
        path = joinpath(dir, "lazy.h5")
        cube = Float32[i + 10j + 100k for i in 1:4, j in 1:5, k in 1:6]
        h5open(path, "w") do f
            f["cube", chunk = (2, 5, 2)] = cube
        end

        lc = MANTA.LazyHDF5Cube{Float32}(path, "/cube", size(cube), (2, 5, 2))
        @test MANTA.prefetch_adjacent!(lc, 3, 3, 2) == 3
        @test lc.prefetch_axis == 3
        @test lc.prefetch_range == 3:4
        @test lc.prefetch_ch !== nothing
        sleep(0.1)
        @test MANTA.read_slice!(lc, 3, 4) == cube[:, :, 4]

        lc2 = MANTA.LazyHDF5Cube{Float32}(path, "/cube", size(cube), (2, 5, 2))
        MANTA.prefetch_adjacent!(lc2, 3, 3, 4)
        @test lc2.prefetch_range == 1:2

        lc3 = MANTA.LazyHDF5Cube{Float32}(path, "/cube", size(cube), (2, 5, 2))
        MANTA.prefetch_adjacent!(lc3, 3, 3, 3)
        @test lc3.prefetch_range == 3:4

        lc4 = MANTA.LazyHDF5Cube{Float32}(path, "/cube", size(cube), (2, 5, 2))
        MANTA.prefetch_adjacent!(lc4, 3, 6, 5)
        @test lc4.prefetch_ch === nothing
        MANTA.prefetch_adjacent!(lc4, 3, 1, 2)
        @test lc4.prefetch_ch === nothing
        @test MANTA.prefetch_adjacent!(lc4, 9, 3, 2) == 3
        @test lc4.prefetch_ch === nothing

        lc5 = MANTA.LazyHDF5Cube{Float32}(path, "/cube", size(cube), (2, 5, 2))
        MANTA.prefetch_adjacent!(lc5, 1, 2, 1)
        @test lc5.prefetch_axis == 1
        @test lc5.prefetch_range == 3:4
    end
end
