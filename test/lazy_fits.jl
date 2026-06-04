# path: test/lazy_fits.jl
# Testsets covering lazy FITS arrays: 2D image and 3D cube (element access,
# slice cache, axis variants, spectrum extraction, prefetch, HDU errors).

# ----------------------------------------------------------------------------
# Lazy FITS arrays
# ----------------------------------------------------------------------------
@testset "lazy FITS: 2D + 3D" begin
    mktempdir() do dir
        # ---- 2D image ----
        path2 = joinpath(dir, "img.fits")
        FITS(path2, "w") do f
            data2 = Float32[i + 10*j for i in 1:6, j in 1:5]
            write(f, data2)
        end
        header, arr, n = MANTA.open_lazy_fits(path2; hdu = 1)
        @test arr isa MANTA.LazyFITSImage{Float32}
        @test size(arr) == (6, 5)
        @test arr[1, 1] == Float32(1 + 10*1)
        @test arr[6, 5] == Float32(6 + 10*5)
        full = collect(arr)
        @test size(full) == (6, 5)
        @test full[3, 2] == Float32(3 + 10*2)

        # load_fits with lazy=true → ImageDataset wrapping the lazy array
        ds = MANTA.load_fits(path2; lazy = true)
        @test ds isa MANTA.ImageDataset
        @test ds.data isa MANTA.LazyFITSImage
        @test get(ds.metadata, :lazy, false) === true

        # ---- 3D cube ----
        path3 = joinpath(dir, "cube.fits")
        FITS(path3, "w") do f
            data3 = Float32[i + 10*j + 100*k for i in 1:4, j in 1:5, k in 1:3]
            write(f, data3)
        end
        h, lc, _ = MANTA.open_lazy_fits(path3; hdu = 1)
        @test lc isa MANTA.LazyFITSCube{Float32}
        @test size(lc) == (4, 5, 3)
        @test lc.cache_capacity == MANTA.DEFAULT_FITS_SLICE_CACHE_CAPACITY
        # Single-element + axis-3 slice.
        @test lc[1, 1, 1] == Float32(1 + 10 + 100)
        slice = MANTA.read_slice!(lc, 3, 2)
        @test size(slice) == (4, 5)
        @test slice[2, 3] == Float32(2 + 30 + 200)
        # Cache hit: same axis/idx returns the same matrix object.
        slice2 = MANTA.read_slice!(lc, 3, 2)
        @test slice2 === slice
        # Axis-1 and axis-2 slices.
        s1 = MANTA.read_slice!(lc, 1, 2)
        @test size(s1) == (5, 3)
        s2 = MANTA.read_slice!(lc, 2, 3)
        @test size(s2) == (4, 3)
        # Spectrum extraction.
        sp = MANTA.read_spectrum(lc, 1, 1)
        @test length(sp) == 3
        @test sp[1] == Float32(1 + 10 + 100)
        @test sp[2] == Float32(1 + 10 + 200)
        # CubeDataset wrapping the lazy cube.
        ds3 = MANTA.load_fits(path3; lazy = true)
        @test ds3 isa MANTA.CubeDataset
        @test ds3.data isa MANTA.LazyFITSCube

        # HDU out-of-range → HDUSelectionError.
        @test_throws MANTA.HDUSelectionError MANTA.open_lazy_fits(path2; hdu = 99)

        # ---- prefetch_slice! ----
        # prefetch a valid slice, then read it — result must match a cold read.
        h2, lc2, _ = MANTA.open_lazy_fits(path3; hdu = 1)
        MANTA.prefetch_slice!(lc2, 3, 1)
        # Give the background task a chance to complete.
        sleep(0.1)
        s_pre = MANTA.read_slice!(lc2, 3, 1)
        @test haskey(lc2.slice_cache, (3, 1))
        # Build a reference via a fresh lazy cube (no cache / no prefetch).
        _, lc_ref, _ = MANTA.open_lazy_fits(path3; hdu = 1)
        s_ref = MANTA.read_slice!(lc_ref, 3, 1)
        @test s_pre == s_ref

        # prefetch out-of-bounds → no-op, no error.
        @test MANTA.prefetch_slice!(lc2, 3, 999) === nothing
        @test MANTA.prefetch_slice!(lc2, 9, 1)   === nothing

        # Double prefetch same key → no-op (channel not replaced).
        MANTA.prefetch_slice!(lc2, 3, 2)
        ch_first = lc2.prefetch_ch
        MANTA.prefetch_slice!(lc2, 3, 2)   # same key → no new channel
        @test lc2.prefetch_ch === ch_first

        # Prefetch superseded by a different key — stale channel closed, new one
        # allocated, and the cold read still returns correct data.
        MANTA.prefetch_slice!(lc2, 3, 3)
        s3 = MANTA.read_slice!(lc2, 3, 3)
        _, lc_ref3, _ = MANTA.open_lazy_fits(path3; hdu = 1)
        s3_ref = MANTA.read_slice!(lc_ref3, 3, 3)
        @test s3 == s3_ref
    end
end

@testset "lazy FITS: slice LRU cache" begin
    mktempdir() do dir
        path = joinpath(dir, "cube.fits")
        FITS(path, "w") do f
            data = Float32[i + 10j + 100k for i in 1:4, j in 1:5, k in 1:10]
            write(f, data)
        end

        _, lc, _ = MANTA.open_lazy_fits(path; hdu = 1)
        @test lc.cache_capacity == 7

        seen = Matrix{Float32}[]
        for k in 1:7
            push!(seen, MANTA.read_slice!(lc, 3, k))
        end
        @test length(lc.slice_cache) == 7
        @test all(haskey(lc.slice_cache, (3, k)) for k in 1:7)

        # Touch slice 2, then add slice 8: slice 1 is the least recent entry.
        @test MANTA.read_slice!(lc, 3, 2) === seen[2]
        s8 = MANTA.read_slice!(lc, 3, 8)
        @test s8[1, 1] == Float32(1 + 10 + 800)
        @test length(lc.slice_cache) == 7
        @test !haskey(lc.slice_cache, (3, 1))
        @test all(haskey(lc.slice_cache, (3, k)) for k in 2:8)

        # Going back to an evicted slice performs a fresh read and re-enters LRU.
        s1_again = MANTA.read_slice!(lc, 3, 1)
        @test s1_again == seen[1]
        @test s1_again !== seen[1]
        @test haskey(lc.slice_cache, (3, 1))
        @test length(lc.slice_cache) == 7
    end
end

# ----------------------------------------------------------------------------
# Direction-aware prefetch (prefetch_adjacent!) — the wiring used by CubeView's
# on(idx) callback to keep scrolling smooth on memory-mapped cubes.
# ----------------------------------------------------------------------------
@testset "lazy FITS: prefetch_adjacent! direction" begin
    mktempdir() do dir
        path = joinpath(dir, "cube.fits")
        FITS(path, "w") do f
            data = Float32[i + 10j + 100k for i in 1:4, j in 1:5, k in 1:6]
            write(f, data)
        end

        # Ascending (last=2, new=3 ⇒ dir +1): prefetch neighbour 4 on axis 3.
        _, lc, _ = MANTA.open_lazy_fits(path; hdu = 1)
        @test MANTA.prefetch_adjacent!(lc, 3, 3, 2) == 3   # returns idx_new
        @test lc.prefetch_axis == 3
        @test lc.prefetch_idx  == 4
        @test lc.prefetch_ch  !== nothing
        # The prefetched slice must match a cold reference read.
        sleep(0.1)
        s4 = MANTA.read_slice!(lc, 3, 4)
        _, lc_ref, _ = MANTA.open_lazy_fits(path; hdu = 1)
        @test s4 == MANTA.read_slice!(lc_ref, 3, 4)

        # Descending (last=4, new=3 ⇒ dir -1): prefetch neighbour 2.
        _, lc2, _ = MANTA.open_lazy_fits(path; hdu = 1)
        MANTA.prefetch_adjacent!(lc2, 3, 3, 4)
        @test lc2.prefetch_idx == 2

        # Unchanged index (last=3, new=3) defaults to +1 ⇒ neighbour 4.
        _, lc3, _ = MANTA.open_lazy_fits(path; hdu = 1)
        MANTA.prefetch_adjacent!(lc3, 3, 3, 3)
        @test lc3.prefetch_idx == 4

        # Neighbour out of bounds at the high edge ⇒ no prefetch launched.
        _, lc4, _ = MANTA.open_lazy_fits(path; hdu = 1)
        MANTA.prefetch_adjacent!(lc4, 3, 6, 5)   # neighbour 7 > nz=6
        @test lc4.prefetch_ch === nothing
        # ...and at the low edge (descending past slice 1).
        MANTA.prefetch_adjacent!(lc4, 3, 1, 2)   # neighbour 0 < 1
        @test lc4.prefetch_ch === nothing

        # Invalid axis ⇒ no-op, still returns idx_new untouched.
        @test MANTA.prefetch_adjacent!(lc4, 9, 3, 2) == 3
        @test lc4.prefetch_ch === nothing

        # Works along axis 1 too (nx=4): last=1,new=2 ⇒ neighbour 3.
        _, lc5, _ = MANTA.open_lazy_fits(path; hdu = 1)
        MANTA.prefetch_adjacent!(lc5, 1, 2, 1)
        @test lc5.prefetch_axis == 1
        @test lc5.prefetch_idx  == 3
    end
end
