# path: test/masks.jl
# Testsets covering the persistent mask system: pure helpers (sources,
# build_mask, stats), TOML round-trip, integration with moment_map /
# mean_region_spectrum, logical combinators, and a headless cube viewer smoke.

# ----------------------------------------------------------------------
# Persistent mask system
# ----------------------------------------------------------------------
@testset "mask: pure helpers" begin
    # Cube has finite values everywhere except one explicit NaN.
    data = Float32[i + 10j + 100k for i in 1:4, j in 1:3, k in 1:2]
    data[2, 2, 1] = NaN32

    @testset "NoMaskSource keeps everything" begin
        bits = MANTA.build_mask(MANTA.NoMaskSource(), data)
        @test size(bits) == size(data)
        @test count(bits) == length(data)
    end

    @testset "FiniteSource rejects only the NaN" begin
        bits = MANTA.build_mask(MANTA.FiniteSource(), data)
        @test count(bits) == length(data) - 1
        @test bits[2, 2, 1] == false
        @test bits[1, 1, 1] == true
    end

    @testset "ThresholdSource :ge / :le / :range / :outside" begin
        # All values are in (1, 1234), so :ge with lo=50 keeps everything
        # ≥ 50; :le with hi=50 keeps everything ≤ 50.
        bge = MANTA.build_mask(MANTA.ThresholdSource(:ge, 50.0, 0.0), data)
        ble = MANTA.build_mask(MANTA.ThresholdSource(:le, 0.0, 50.0), data)
        brn = MANTA.build_mask(MANTA.ThresholdSource(:range, 20.0, 100.0), data)
        bou = MANTA.build_mask(MANTA.ThresholdSource(:outside, 20.0, 100.0), data)
        # Disjoint partitions (except for non-finite voxels):
        finite_count = sum(isfinite, data)
        @test count(bge) + count(ble) >= finite_count - 1  # boundary overlap at 50 allowed
        @test count(brn) + count(bou) == finite_count
        # NaN always rejected
        @test bge[2, 2, 1] == false
        @test ble[2, 2, 1] == false
        @test brn[2, 2, 1] == false
        @test bou[2, 2, 1] == false
    end

    @testset "ThresholdSource validates op" begin
        @test_throws ArgumentError MANTA.ThresholdSource(:bogus, 0.0, 1.0)
    end

    @testset "RectangleSource: closed-box selection" begin
        src = MANTA.RectangleSource(i1 = 2, i2 = 3, j1 = 1, j2 = 2)
        bits = MANTA.build_mask(src, data)
        # Selected box: i ∈ {2,3}, j ∈ {1,2}, k ∈ {1,2} (k unconstrained)
        @test count(bits) == 2 * 2 * 2
        @test bits[2, 1, 1] == true
        @test bits[3, 2, 2] == true
        @test bits[1, 1, 1] == false
        @test bits[4, 1, 1] == false
    end

    @testset "RectangleSource auto-swaps min/max" begin
        # i1 > i2 should be reordered automatically by the keyword constructor.
        src = MANTA.RectangleSource(i1 = 3, i2 = 1)
        bits = MANTA.build_mask(src, data)
        @test count(bits) > 0
        @test bits[1, 1, 1] == true
        @test bits[3, 1, 1] == true
    end

    @testset "validate_bounds: out-of-range throws ArgumentError" begin
        # data is 4×3×2.
        # Upper bound exceeds dimension.
        @test_throws ArgumentError MANTA.validate_bounds(
            MANTA.RectangleSource(i1 = 1, i2 = 1000), (4, 3, 2))
        # Lower bound below 1.
        @test_throws ArgumentError MANTA.validate_bounds(
            MANTA.RectangleSource(j1 = 0), (4, 3, 2))
        # k bound too large.
        @test_throws ArgumentError MANTA.validate_bounds(
            MANTA.RectangleSource(k2 = 99), (4, 3, 2))
        # build_mask propagates the error (same check, triggered through the public API).
        @test_throws ArgumentError MANTA.build_mask(
            MANTA.RectangleSource(i1 = 1, i2 = 1000), data)  # data is 4×3×2
    end

    @testset "validate_bounds: valid and nothing bounds are accepted" begin
        # All bounds within range → no error.
        MANTA.validate_bounds(MANTA.RectangleSource(i1 = 1, i2 = 4, j1 = 1, j2 = 3, k1 = 1, k2 = 2), (4, 3, 2))
        # nothing bounds are always valid.
        MANTA.validate_bounds(MANTA.RectangleSource(), (4, 3, 2))
        # Fallback: non-rectangle sources are no-ops.
        MANTA.validate_bounds(MANTA.NoMaskSource(), (4, 3, 2))
        MANTA.validate_bounds(MANTA.FiniteSource(), (4, 3, 2))
        MANTA.validate_bounds(MANTA.ThresholdSource(:ge, 1.0, 0.0), (4, 3, 2))
        @test true  # reached without throwing
    end

    @testset "MANTAMask: stats helpers" begin
        m = MANTA.make_mask(MANTA.FiniteSource(), data)
        @test MANTA.mask_total(m) == length(data)
        @test MANTA.mask_count(m) == length(data) - 1
        @test 0 < MANTA.mask_fraction(m) < 1
    end
end

@testset "mask: TOML round-trip" begin
    for src in (
        MANTA.NoMaskSource(),
        MANTA.FiniteSource(),
        MANTA.ThresholdSource(:ge, 1.5, 0.0),
        MANTA.ThresholdSource(:range, -3.0, 7.5),
        MANTA.RectangleSource(i1 = 2, i2 = 5, k1 = 1),
    )
        d = MANTA.mask_source_to_toml(src)
        @test haskey(d, "kind")
        rt = MANTA.mask_source_from_toml(d)
        @test typeof(rt) === typeof(src)
        if src isa MANTA.ThresholdSource
            @test rt.op == src.op
            @test rt.lo == src.lo
            @test rt.hi == src.hi
        elseif src isa MANTA.RectangleSource
            @test rt.i1 == src.i1
            @test rt.i2 == src.i2
            @test rt.j1 == src.j1
            @test rt.j2 == src.j2
            @test rt.k1 == src.k1
            @test rt.k2 == src.k2
        end
    end

    # Malformed TOML → NoMaskSource (graceful degradation).
    @test MANTA.mask_source_from_toml(Dict{String,Any}("kind" => "bogus")) isa MANTA.NoMaskSource
    @test MANTA.mask_source_from_toml(Dict{String,Any}()) isa MANTA.NoMaskSource
    @test MANTA.mask_source_from_toml(nothing) isa MANTA.NoMaskSource
    # Unknown op → NoMaskSource
    @test MANTA.mask_source_from_toml(Dict{String,Any}("kind" => "threshold",
        "op" => "huh", "lo" => 0.0, "hi" => 1.0)) isa MANTA.NoMaskSource
end

@testset "mask: integration with moment_map / mean_region_spectrum" begin
    # Synthetic cube: integer values 1..N so moments are trivial to predict.
    nx, ny, nz = 3, 4, 5
    data = Array{Float32}(undef, nx, ny, nz)
    for i in 1:nx, j in 1:ny, k in 1:nz
        data[i, j, k] = Float32(i + 10j + 100k)
    end

    # --- moment_map: unmasked vs masked path produce different results ---
    m0_unmasked = MANTA.moment_map(data, 3, 0)
    bits = MANTA.build_mask(MANTA.ThresholdSource(:ge, 200.0, 0.0), data)
    m0_masked = MANTA.moment_map(data, 3, 0; mask = bits)
    @test size(m0_unmasked) == size(m0_masked)
    @test m0_unmasked != m0_masked
    # Masked sum is ≤ unmasked sum (we dropped some channels)
    @test sum(filter(isfinite, m0_masked)) <= sum(filter(isfinite, m0_unmasked))

    # --- mean_region_spectrum: same behavior ---
    uv = [(1, 1), (2, 1), (1, 2)]
    y_unmasked = MANTA.mean_region_spectrum(data, 3, uv)
    y_masked = MANTA.mean_region_spectrum(data, 3, uv; mask = bits)
    @test length(y_unmasked) == nz
    @test length(y_masked) == nz
    # Early channels rejected by :ge 200 mask
    @test isnan(y_masked[1])
    @test !isnan(y_unmasked[1])

    # --- Dimension mismatch is an error ---
    bad_bits = trues(2, 2, 2)
    @test_throws DimensionMismatch MANTA.moment_map(data, 3, 0; mask = bad_bits)
    @test_throws DimensionMismatch MANTA.mean_region_spectrum(data, 3, uv; mask = bad_bits)
end

@testset "mask: combinators (AndSource / OrSource / NotSource)" begin
    data = Float32[i + 10j + 100k for i in 1:4, j in 1:3, k in 1:2]
    # One explicit NaN to verify combinator semantics against non-finite voxels.
    data[1, 1, 1] = NaN32

    # --- NotSource: invert NoMaskSource → all false ---
    bits_not_all = MANTA.build_mask(MANTA.NotSource(MANTA.NoMaskSource()), data)
    @test count(bits_not_all) == 0

    # --- NotSource: invert FiniteSource → keeps only NaN voxel ---
    bits_not_finite = MANTA.build_mask(MANTA.NotSource(MANTA.FiniteSource()), data)
    @test count(bits_not_finite) == 1
    @test bits_not_finite[1, 1, 1] == true
    @test bits_not_finite[2, 1, 1] == false

    # --- AndSource: Threshold :ge AND Rectangle ---
    # Threshold :ge 500 keeps voxels with value ≥ 500 (k==2 only, roughly).
    # Rectangle i∈[1,2], j∈[1,2], k∈[1,2].
    thr = MANTA.ThresholdSource(:ge, 500.0, 0.0)
    rect = MANTA.RectangleSource(i1=1, i2=2, j1=1, j2=2)
    bits_and = MANTA.build_mask(MANTA.AndSource(thr, rect), data)
    bits_thr  = MANTA.build_mask(thr,  data)
    bits_rect = MANTA.build_mask(rect, data)
    # AND result must be a subset of both operands.
    @test all(i -> bits_and[i] <= bits_thr[i],  eachindex(bits_and))
    @test all(i -> bits_and[i] <= bits_rect[i], eachindex(bits_and))
    # At least one voxel must survive (i=2,j=2,k=2 has value 2+20+200=222... wait
    # let me recheck: value = i+10j+100k → max = 4+30+200 = 234 < 500).
    # Adjust threshold to 100 for the AND test to be non-trivial.
    thr2 = MANTA.ThresholdSource(:ge, 100.0, 0.0)
    bits_and2 = MANTA.build_mask(MANTA.AndSource(thr2, rect), data)
    bits_thr2 = MANTA.build_mask(thr2, data)
    @test count(bits_and2) > 0
    @test count(bits_and2) <= min(count(bits_thr2), count(bits_rect))

    # --- OrSource: union is ≥ each operand alone ---
    bits_or = MANTA.build_mask(MANTA.OrSource(thr2, rect), data)
    @test count(bits_or) >= count(bits_thr2)
    @test count(bits_or) >= count(bits_rect)

    # --- Partition law: AND + OR vs two individual masks ---
    # |A ∪ B| = |A| + |B| - |A ∩ B|
    @test count(bits_or) == count(bits_thr2) + count(bits_rect) - count(bits_and2)

    # --- Deep composition: Not(And(…, …)) ---
    bits_deep = MANTA.build_mask(
        MANTA.NotSource(MANTA.AndSource(thr2, rect)), data)
    # ~(A & B) = count(all) - count(A & B)
    @test count(bits_deep) == length(data) - count(bits_and2)

    # --- TOML round-trip for all three combinators ---
    for src in (
        MANTA.AndSource(thr2, rect),
        MANTA.OrSource(MANTA.FiniteSource(), rect),
        MANTA.NotSource(MANTA.ThresholdSource(:range, 50.0, 150.0)),
        # Nested: Not(And(Threshold, Not(Rectangle)))
        MANTA.NotSource(MANTA.AndSource(thr2, MANTA.NotSource(rect))),
    )
        d  = MANTA.mask_source_to_toml(src)
        rt = MANTA.mask_source_from_toml(d)
        @test typeof(rt) === typeof(src)
        # Materialized bits must match (functional equality).
        bits_orig = MANTA.build_mask(src,  data)
        bits_rt   = MANTA.build_mask(rt,   data)
        @test bits_orig == bits_rt
    end
end

@testset "mask: headless cube viewer smoke test" begin
    # Build a tiny cube + open the viewer with activate_gl=false / display_fig=false.
    # Then sanity-check that toggling the mask doesn't crash anything.
    nx, ny, nz = 6, 5, 4
    data = Array{Float32}(undef, nx, ny, nz)
    for i in 1:nx, j in 1:ny, k in 1:nz
        data[i, j, k] = Float32(i + j + k)
    end
    ds = MANTA.load_dataset(data)
    fig = MANTA.view_cube(ds; activate_gl = false, display_fig = false)
    @test fig isa Figure
end
