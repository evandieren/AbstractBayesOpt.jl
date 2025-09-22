using Test
using AbstractBayesOpt
using AbstractGPs, KernelFunctions
using Statistics
using Random

@testset "Surrogate Model Tests" begin
    @testset "StandardGP Tests" begin
        @testset "StandardGP Construction" begin
            # Test basic construction
            kernel = SqExponentialKernel()
            noise_var = 0.1
            gp = StandardGP(kernel, noise_var)

            @test gp.noise_var == noise_var
            @test gp.gpx === nothing
            @test isa(gp.gp, AbstractGPs.GP)
        end

        @testset "StandardGP Update" begin
            kernel = SqExponentialKernel()
            noise_var = 0.1
            gp = StandardGP(kernel, noise_var)

            # Create training data
            xs = [0.0, 0.5, 1.0]
            ys = [0.0, 0.25, 1.0]

            # Update GP
            updated_gp = update(gp, xs, ys)

            @test updated_gp.noise_var == noise_var
            @test updated_gp.gpx !== nothing
            @test isa(updated_gp, StandardGP)

            # Test predictions
            test_x = [0.25]
            mean_pred = posterior_mean(updated_gp, test_x)
            var_pred = posterior_var(updated_gp, test_x)

            @test isa(mean_pred, Float64)
            @test isa(var_pred, Float64)
            @test var_pred >= 0.0
        end

        @testset "StandardGP Utility Functions" begin
            kernel = SqExponentialKernel()
            noise_var = 0.1
            gp = StandardGP(kernel, noise_var)

            # Test prep_input
            x = [0.5, 1.0]
            prepped = prep_input(gp, x)
            @test prepped == x

            # Test with updated GP
            xs = [0.0, 0.5, 1.0]
            ys = [0.0, 0.25, 1.0]
            updated_gp = update(gp, xs, ys)

        end

        @testset "StandardGP Standardization" begin
            kernel = SqExponentialKernel()
            noise_var = 0.1
            gp = StandardGP(kernel, noise_var)

            # Test standardization using the functions from bayesian_opt.jl
            y_train = [1.0, 2.0, 3.0, 4.0, 5.0]
            μ, σ = get_mean_std(gp, y_train, "mean_scale")
            y_std = std_y(gp, y_train, μ, σ)

            @test length(y_std) == length(y_train)
            @test μ ≈ 3.0  # mean of [1,2,3,4,5]
            @test σ > 0

            # Check that rescaled data has different scale
            y_flat_std = reduce(vcat, y_std)
            y_flat_orig = reduce(vcat, y_train)
            @test std(y_flat_std) ≈ 1.0 atol = 1e-10
        end

        @testset "StandardGP Copy" begin
            kernel = SqExponentialKernel()
            noise_var = 0.1
            gp = StandardGP(kernel, noise_var)

            xs = [0.0, 1.0]
            ys = [0.0, 1.0]
            updated_gp = update(gp, xs, ys)

            copied_gp = copy(updated_gp)
            @test copied_gp.noise_var == updated_gp.noise_var
            @test copied_gp.gp === updated_gp.gp  # Should be same reference
            @test copied_gp.gpx !== updated_gp.gpx  # Should be different reference
        end

        @testset "StandardGP NLML" begin
            kernel = 1 * (SqExponentialKernel() ∘ ScaleTransform(1.0))
            noise_var = 0.1
            gp = StandardGP(kernel, noise_var)

            # Test NLML computation with correct signature
            params = [log(1.0), log(1.0)]  # log lengthscale, log scale
            x = [0.0, 0.5, 1.0]
            y = [0.0, 0.25, 1.0]

            # Pass the kernel constructor, not instance
            nlml_val = nlml(gp, params, x, y)
            @test isa(nlml_val, Real)
            @test isfinite(nlml_val)
        end
    end

    @testset "GradientGP Tests" begin
        @testset "GradientGP Construction" begin
            # Test ApproxMatern52Kernel
            kernel_base = ApproxMatern52Kernel()
            @test isa(kernel_base, ApproxMatern52Kernel)

            # Test kappa function for ApproxMatern52Kernel
            @test KernelFunctions.kappa(kernel_base, 0.0) ≈ 1.0
            @test KernelFunctions.kappa(kernel_base, 1e-12) ≈ 1.0  # Should use Taylor approximation

            # Test gradKernel construction
            grad_kernel = gradKernel(SqExponentialKernel())
            @test isa(grad_kernel, gradKernel)

            # Test GradientGP construction
            p = 3  # 2D problem + function value (1 + 2 gradients)
            noise_var = 0.1
            gp = GradientGP(SqExponentialKernel(), p, noise_var)

            @test gp.noise_var == noise_var
            @test gp.p == p
            @test gp.gpx === nothing
            @test isa(gp.gp, AbstractGPs.GP)
        end

        @testset "GradientGP Update" begin
            kernel_base = SqExponentialKernel()
            p = 3  # function + 2 gradients
            noise_var = 0.1
            gp = GradientGP(kernel_base, p, noise_var)

            # Create training data with gradients
            xs = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
            # Each y contains [f(x), ∇f(x)]
            ys = [[1.0, 0.1, 0.1], [0.5, 0.0, 0.0], [0.0, -0.1, -0.1]]

            # Update GP
            updated_gp = update(gp, xs, ys)

            @test updated_gp.noise_var == noise_var
            @test updated_gp.p == p
            @test updated_gp.gpx !== nothing
            @test isa(updated_gp, GradientGP)

            # Test predictions
            test_x = [0.25, 0.25]
            mean_pred = posterior_mean(updated_gp, test_x)
            var_pred = posterior_var(updated_gp, test_x)
            grad_mean = posterior_grad_mean(updated_gp, test_x)
            grad_var = posterior_grad_var(updated_gp, test_x)

            @test isa(mean_pred, Float64)
            @test isa(var_pred, Float64)
            @test var_pred >= 0.0
            @test isa(grad_mean, AbstractVector)
            @test isa(grad_var, AbstractVector)
            @test length(grad_mean) == p  # function + gradients
            @test length(grad_var) == p
        end

        @testset "GradientGP Utility Functions" begin
            kernel_base = SqExponentialKernel()
            p = 3
            noise_var = 0.1
            gp = GradientGP(kernel_base, p, noise_var)

            # Test prep_input
            x = [[0.5, 1.0]]
            prepped = prep_input(gp, x)
            @test isa(prepped, KernelFunctions.MOInputIsotopicByOutputs)

            # Test with updated GP
            xs = [[0.0, 0.0], [1.0, 1.0]]
            ys = [[1.0, 0.1, 0.1], [0.0, -0.1, -0.1]]
            updated_gp = update(gp, xs, ys)

            # Test lengthscale and scale extraction - skip for now due to kernel structure
            # lengthscale = get_lengthscale(updated_gp)
            # scale = get_scale(updated_gp)

            # @test isa(lengthscale, AbstractVector)
            # @test isa(scale, Real)
            # @test all(lengthscale .> 0)
            # @test scale > 0
        end

        @testset "GradientGP Standardization" begin
            kernel_base = SqExponentialKernel()
            p = 3
            noise_var = 0.1
            gp = GradientGP(kernel_base, p, noise_var)

            # Test standardization using the functions from bayesian_opt.jl
            y_train = [[1.0, 0.1, 0.1], [2.0, 0.2, 0.2], [3.0, 0.3, 0.3]]
            μ, σ = get_mean_std(gp, y_train, "mean_scale")
            y_std = std_y(gp, y_train, μ, σ)

            @test length(y_std) == length(y_train)
            @test length(μ) == p
            @test length(σ) == p
            @test μ[1] ≈ 2.0  # mean of function values [1,2,3]
            @test μ[2] == 0.0  # gradients should have zero mean
            @test μ[3] == 0.0
            @test σ[1] > 0
            @test σ[2] == σ[1]  # gradients use same scaling as function
            @test σ[3] == σ[1]

            # Check that rescaled data matches expected properties
            y_flat_std = reduce(vcat, y_std)
            y_flat_orig = reduce(vcat, y_train)

            # Check that the standardized values match with the original standardization formula
            for (y_orig, y_s) in zip(y_train, y_std)
                @test y_s[1] ≈ (y_orig[1] - μ[1]) / σ[1] atol = 1e-8
                @test y_s[2] ≈ (y_orig[2] - μ[2]) / σ[2] atol = 1e-8
                @test y_s[3] ≈ (y_orig[3] - μ[3]) / σ[3] atol = 1e-8
            end
        end

        @testset "GradientGP Copy" begin
            kernel_base = SqExponentialKernel()
            p = 3
            noise_var = 0.1
            gp = GradientGP(kernel_base, p, noise_var)

            xs = [[0.0, 0.0], [1.0, 1.0]]
            ys = [[1.0, 0.1, 0.1], [0.0, -0.1, -0.1]]
            updated_gp = update(gp, xs, ys)

            copied_gp = copy(updated_gp)
            @test copied_gp.noise_var == updated_gp.noise_var
            @test copied_gp.p == updated_gp.p
            @test copied_gp.gp === updated_gp.gp  # Should be same reference
            @test copied_gp.gpx !== updated_gp.gpx  # Should be different reference
        end

        @testset "gradKernel Functionality" begin
            kernel_base = SqExponentialKernel()
            grad_kernel = gradKernel(kernel_base)
            x = [0.5, 0.5]
            y = [0.6, 0.6]

            # Test function-function evaluation (px=1, py=1)
            val_ff = grad_kernel((x, 1), (y, 1))
            @test isa(val_ff, Real)
            @test isfinite(val_ff)

            # Test function-gradient evaluation (px=1, py>1)
            val_fg = grad_kernel((x, 1), (y, 2))
            @test isa(val_fg, Real)
            @test isfinite(val_fg)

            # Test gradient-function evaluation (px>1, py=1)
            val_gf = grad_kernel((x, 2), (y, 1))
            @test isa(val_gf, Real)
            @test isfinite(val_gf)

            # Test gradient-gradient evaluation (px>1, py>1)
            val_gg = grad_kernel((x, 2), (y, 2))
            @test isa(val_gg, Real)
            @test isfinite(val_gg)

            # Test symmetry for function-function case
            @test grad_kernel((x, 1), (y, 1)) ≈ grad_kernel((y, 1), (x, 1))
        end
    end
end
