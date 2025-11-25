# path: test/runtests.jl
import Pkg
using Test

# charge le module local
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using CartaViewer

# deps utilisées par les helpers
using Observables
using Makie
using LaTeXStrings
using ColorTypes

@testset "helpers: scaling" begin
    A = Float32.([1, 10, 100, 0, -1])
    lin = CartaViewer.apply_scale(A, :lin)
    log10v = CartaViewer.apply_scale(A, :log10)
    lnv = CartaViewer.apply_scale(A, :ln)

    @test eltype(lin) == Float32
    @test eltype(log10v) == Float32
    @test eltype(lnv) == Float32

    @test lin[1:3] == A[1:3]
    @test isapprox(log10v[1], 0f0; atol=1e-6)
    @test isapprox(log10v[2], 1f0; atol=1e-6)
    @test isfinite(lnv[1])
    @test !isfinite(lnv[4]) && !isfinite(lnv[5])

    mn, mx = CartaViewer.clamped_extrema(Float32.([1, 2, 3]))
    @test mn == 1f0 && mx == 3f0

    mn2, mx2 = CartaViewer.clamped_extrema(Float32.([5, 5, 5]))
    @test mn2 < 5.0f0 && mx2 > 5.0f0

    mn3, mx3 = CartaViewer.clamped_extrema(Float32.([NaN32, NaN32]))
    @test mn3 == 0f0 && mx3 == 1f0

    mn4, mx4 = CartaViewer.clamped_extrema(Float32.([]))
    @test mn4 == 0f0 && mx4 == 1f0
end

@testset "helpers: mapping" begin
    # bijection uv <-> ijk selon l'axe
    for axis in 1:3
        i, j, k = 3, 2, 1
        u, v = CartaViewer.ijk_to_uv(i, j, k, axis)
        ii, jj, kk = CartaViewer.uv_to_ijk(u, v, axis, axis == 1 ? i : axis == 2 ? j : k)
        @test (ii, jj, kk) == (i, j, k)
    end

    # get_slice dims et type
    data = Array{Float32}(undef, 7, 5, 4)
    fill!(data, 1f0)
    s1 = CartaViewer.get_slice(data, 1, 2)
    s2 = CartaViewer.get_slice(data, 2, 3)
    s3 = CartaViewer.get_slice(data, 3, 1)
    @test size(s1) == (size(data, 2), size(data, 3))
    @test size(s2) == (size(data, 1), size(data, 3))
    @test size(s3) == (size(data, 1), size(data, 2))
    @test eltype(s1) == Float32 && eltype(s2) == Float32 && eltype(s3) == Float32
end

@testset "helpers: latex" begin
    s = CartaViewer.make_info_tex(1, 2, 3, 4, 5, 6f0)
    t1 = CartaViewer.make_slice_title("fname", 3, 10)
    t2 = CartaViewer.make_spec_title(1, 2, 3)

    @test s isa LaTeXString
    @test t1 isa LaTeXString
    @test t2 isa LaTeXString

    # Pas de sauts de ligne LaTeX : interdit "\\ " et "\\\n"
    raw_s  = String(s)
    raw_t1 = String(t1)
    raw_t2 = String(t2)
    for raw in (raw_s, raw_t1, raw_t2)
        @test !occursin("\\\\ ", raw)
        @test !occursin("\\\\\\n", raw)
    end

    # Présence de LaTeX inline attendue (ex: \\, pour espace fine)
    @test occursin("\\\\,", raw_s) || occursin("\\\\,", raw_t1) || occursin("\\\\,", raw_t2)
end

@testset "helpers: io" begin
    # to_cmap
    cm = CartaViewer.to_cmap(:viridis)
    @test length(cm) > 0
    @test cm[1] isa ColorTypes.Colorant

    # get_box_str via mock (pas de Textbox Makie)
    struct MockTB
        stored_string::Observable{String}
    end
    tb = MockTB(Observable("   hello world   "))
    @test CartaViewer.get_box_str(tb) == "hello world"
    
    struct MockDisplayTB
        displayed_string::Observable{String}
    end
    tb2 = MockDisplayTB(Observable("   fallback value   "))
    @test CartaViewer.get_box_str(tb2) == "fallback value"
end

@testset "helpers: ui" begin
    # override explicite
    @test CartaViewer._pick_fig_size((111, 222)) == (111, 222)
    # défaut sans taille explicite
    @test CartaViewer._pick_fig_size(nothing) == (1800, 900)
end
