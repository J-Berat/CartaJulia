# path: test/drag_drop.jl
# Drag-and-drop file loading. All tests run headless (activate_gl=false,
# display_fig=false): with no GLMakie screen attached the reload is built but
# never displayed, so the drop logic is exercised without an OpenGL context.

@testset "drag-and-drop: supported_drop_path" begin
    mktempdir() do tmp
        fits_path = joinpath(tmp, "img.fits")
        FITS(fits_path, "w") do f
            write(f, Float32[1 2; 3 4])
        end
        h5_path = joinpath(tmp, "data.h5")
        HDF5.h5open(h5_path, "w") do f
            f["x"] = Float32[1.0, 2.0, 3.0]
        end
        txt_path = joinpath(tmp, "notes.txt")
        write(txt_path, "not a dataset")

        # First supported entry wins; unsupported / missing entries are skipped.
        @test MANTA.supported_drop_path([txt_path, fits_path]) == fits_path
        @test MANTA.supported_drop_path([txt_path, h5_path]) == h5_path
        @test MANTA.supported_drop_path([fits_path, h5_path]) == fits_path
        # Nonexistent paths never qualify, even with a supported extension.
        @test MANTA.supported_drop_path([joinpath(tmp, "ghost.fits")]) === nothing
        # No supported file / empty drop → nothing.
        @test MANTA.supported_drop_path([txt_path]) === nothing
        @test MANTA.supported_drop_path(String[]) === nothing
    end
end

@testset "drag-and-drop: _handle_drop dispatch" begin
    mktempdir() do tmp
        fits_path = joinpath(tmp, "img.fits")
        FITS(fits_path, "w") do f
            write(f, Float32[1 2; 3 4])
        end

        # Unsupported drop: reopen is never called and the current view stands.
        called = Ref(false)
        res = MANTA._handle_drop(Figure(), [joinpath(tmp, "x.txt")],
                                 p -> (called[] = true; Figure()); display_fig = false)
        @test res === nothing
        @test called[] == false

        # Supported drop: reopen is called with the resolved path; with no screen
        # and display_fig=false the freshly built figure is just returned.
        seen = Ref("")
        new_fig = Figure()
        res2 = MANTA._handle_drop(Figure(), [fits_path],
                                  p -> (seen[] = p; new_fig); display_fig = false)
        @test res2 === new_fig
        @test seen[] == fits_path

        # A failing reopen is swallowed so a bad drop never tears down the window.
        res3 = MANTA._handle_drop(Figure(), [fits_path],
                                  p -> error("boom"); display_fig = false)
        @test res3 === nothing
    end
end

@testset "drag-and-drop: enable_file_drop! arming" begin
    mktempdir() do tmp
        fits_path = joinpath(tmp, "img.fits")
        FITS(fits_path, "w") do f
            write(f, Float32[1 2; 3 4])
        end

        # Idempotent: arming twice registers a single listener, so a drop fires
        # the reopen exactly once.
        fig = Figure()
        n = Ref(0)
        rfn = p -> (n[] += 1; Figure())
        MANTA.enable_file_drop!(fig; activate_gl = false, display_fig = false, reopen = rfn)
        MANTA.enable_file_drop!(fig; activate_gl = false, display_fig = false, reopen = rfn)
        events(fig).dropped_files[] = [fits_path]
        @test n[] == 1

        # Unsupported drop on an armed figure is a no-op.
        events(fig).dropped_files[] = [joinpath(tmp, "x.txt")]
        @test n[] == 1
    end
end

@testset "drag-and-drop: wired into viewers (headless)" begin
    mktempdir() do tmp
        img_path = joinpath(tmp, "image2d.fits")
        FITS(img_path, "w") do f
            write(f, Float32[i + 10j for i in 1:6, j in 1:5])
        end

        # The 2-D viewer arms the drop target on construction.
        fig = MANTA.manta(Float32[1 2 3; 4 5 6]; activate_gl = false, display_fig = false)
        @test fig isa Makie.Figure
        @test !isempty(events(fig).dropped_files.listeners)

        # Dropping a real FITS file rebuilds a viewer headlessly without error.
        events(fig).dropped_files[] = [img_path]
        MANTA.forget!(fig)
    end
end
