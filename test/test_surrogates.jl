using Test
using AbstractBayesOpt
using AbstractGPs, KernelFunctions
using Statistics
using Random
using LinearAlgebra

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

            # Test the kernel with lengthscale and scale
            ℓ = get_lengthscale(gp)[1]
            scale = get_scale(gp)[1]
            @test ℓ == 1.0
            @test scale == 1.0

            # Test with custom lengthscale and scale
            custom_ℓ = 0.5
            custom_scale = 2.0
            custom_kernel = custom_scale * (with_lengthscale(kernel, custom_ℓ))
            gp_custom = StandardGP(custom_kernel, noise_var)
            @test get_lengthscale(gp_custom) == [custom_ℓ]
            @test get_scale(gp_custom) == [custom_scale]
            @test gp_custom.noise_var == noise_var
            @test gp_custom.gpx === nothing
            @test isa(gp_custom.gp, AbstractGPs.GP)

            # Test with only lengthscale
            custom_ℓ2 = 0.3
            kernel_ls = with_lengthscale(kernel, custom_ℓ2)
            gp_ls = StandardGP(kernel_ls, noise_var)
            @test get_lengthscale(gp_ls) == [custom_ℓ2]
            @test get_scale(gp_ls) == [1.0]  # default scale
            @test gp_ls.noise_var == noise_var
            @test gp_ls.gpx === nothing
            @test isa(gp_ls.gp, AbstractGPs.GP)

            # Test with only scale
            custom_scale2 = 3.0
            kernel_sc = custom_scale2 * kernel
            gp_sc = StandardGP(kernel_sc, noise_var)
            @test get_lengthscale(gp_sc) == [1.0]  # default length
            @test get_scale(gp_sc) == [custom_scale2]
            @test gp_sc.noise_var == noise_var
            @test gp_sc.gpx === nothing
            @test isa(gp_sc.gp, AbstractGPs.GP)
        end

        @testset "StandardGP Update" begin
            # Tests updating the surrogate with training data and compare predictions with theory

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

            @test isa(mean_pred[1], Float64)
            @test isa(var_pred[1], Float64)
            @test var_pred[1] >= 0.0

            # Check the value of posterior mean and var
            k_xX = kernel.(Ref(test_x), xs)
            K̃ = kernelmatrix(kernel, xs) + noise_var * I

            true_mean_post = k_xX' * (K̃ \ ys)
            true_var_post = kernel(test_x, test_x) - k_xX' * (K̃ \ k_xX)

            # Test prep_input
            x = [0.5, 1.0]
            prepped = prep_input(gp, x)
            @test prepped == x

            # Test with updated GP
            xs = [0.0, 0.5, 1.0]
            ys = [0.0, 0.25, 1.0]

            updated_gp = update(gp, xs, ys)
            @test isapprox(mean_pred[1], true_mean_post, atol=1e-10)
            @test isapprox(var_pred[1], true_var_post, atol=1e-10)
        end

        @testset "StandardGP Standardization output" begin
            # Testing functions std_y and get_mean_std

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
            @test std(y_flat_std)≈1.0 atol=1e-10
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
            kernel = SqExponentialKernel()
            noise_var = 0.1
            gp = StandardGP(kernel, noise_var)

            # Test NLML computation with correct signature
            params = [log(1.0), log(1.0)]  # log lengthscale, log scale
            x = [0.0, 0.5, 1.0]
            y = [0.0, 0.25, 1.0]

            # Pass the kernel constructor, not instance
            nlml_val = nlml(gp, params, x, y)

            # Compute analyic NLML for comparison

            # Get the current kernel matrix for gradient GP
            K̃ = kernelmatrix(kernel, x) + noise_var * I

            # Compute the three components of the logpdf
            K_inv_y = K̃ \ y
            quadratic_form = y' * K_inv_y
            constant_term = length(y) * log(2π)
            analytical_logpdf = -0.5 * (quadratic_form + logdet(K̃) + constant_term)
            true_nlml = -analytical_logpdf
            @test isapprox(nlml_val, true_nlml, atol=1e-10)
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
            gp = GradientGP(kernel_base, p, noise_var)

            @test gp.noise_var == noise_var
            @test gp.p == p
            @test gp.gpx === nothing
            @test isa(gp.gp, AbstractGPs.GP)

            # Test the kernel with lengthscale and scale
            ℓ = get_lengthscale(gp)[1]
            scale = get_scale(gp)[1]
            @test ℓ == 1.0
            @test scale == 1.0

            # Test with custom lengthscale and scale
            custom_ℓ = 0.5
            custom_scale = 2.0
            custom_kernel = custom_scale * (with_lengthscale(kernel_base, custom_ℓ))
            gp_custom = GradientGP(custom_kernel, p, noise_var)
            @test get_lengthscale(gp_custom) == [custom_ℓ]
            @test get_scale(gp_custom) == [custom_scale]
            @test gp_custom.noise_var == noise_var
            @test gp_custom.gpx === nothing
            @test isa(gp_custom.gp, AbstractGPs.GP)

            # Test with only lengthscale
            custom_ℓ2 = 0.3
            kernel_ls = with_lengthscale(kernel_base, custom_ℓ2)
            gp_ls = GradientGP(kernel_ls, p, noise_var)
            @test get_lengthscale(gp_ls) == [custom_ℓ2]
            @test get_scale(gp_ls) == [1.0]  # default scale
            @test gp_ls.noise_var == noise_var
            @test gp_ls.gpx === nothing
            @test isa(gp_ls.gp, AbstractGPs.GP)

            # Test with only scale
            custom_scale2 = 3.0
            kernel_sc = custom_scale2 * kernel_base
            gp_sc = GradientGP(kernel_sc, p, noise_var)
            @test get_lengthscale(gp_sc) == [1.0]  # default length
            @test get_scale(gp_sc) == [custom_scale2]
            @test gp_sc.noise_var == noise_var
            @test gp_sc.gpx === nothing
            @test isa(gp_sc.gp, AbstractGPs.GP)
        end

        @testset "gradKernel Functionality" begin
            kernel_base = SqExponentialKernel()

            ∇₁kernel(x, y) = ForwardDiff.gradient(t -> kernel_base(t, y), x)
            ∇₂kernel(x, y) = ForwardDiff.gradient(t -> kernel_base(x, t), y)

            ∇₁₂kernel((x, px), (y, py)) = ForwardDiff.derivative(
                h1 -> ForwardDiff.derivative(
                    h2 -> kernel_base(
                        x .+ h1 .* (1:length(x) .== (px - 1)),
                        y .+ h2 .* (1:length(y) .== (py - 1)),
                    ),
                    0.0,
                ),
                0.0,
            )

            grad_kernel = gradKernel(kernel_base)
            x = [0.5, 0.5]
            y = [0.6, 0.6]

            # Test function-function evaluation (px=1, py=1)
            val_ff = grad_kernel((x, 1), (y, 1))

            true_val_ff = kernel_base(x, y)
            @test isapprox(val_ff, true_val_ff, atol=1e-10)

            # Test function-gradient evaluation (px=1, py>1)
            val_fg = [grad_kernel((x, 1), (y, 2)); grad_kernel((x, 1), (y, 3))]
            true_val_fg = ∇₂kernel(x, y)
            @test isapprox(val_fg, true_val_fg, atol=1e-10)

            # Test gradient-function evaluation (px>1, py=1)
            val_gf = [grad_kernel((x, 2), (y, 1)); grad_kernel((x, 3), (y, 1))]
            true_val_gf = ∇₁kernel(x, y)
            @test isapprox(val_gf, true_val_gf, atol=1e-10)

            # Test gradient-gradient evaluation (px>1, py>1)
            val_gg = grad_kernel((x, 2), (y, 2))
            true_val_gg = ∇₁₂kernel((x, 2), (y, 2))
            @test isapprox(val_gg, true_val_gg, atol=1e-10)

            val_gg_23 = grad_kernel((x, 2), (y, 3))
            true_val_gg_23 = ∇₁₂kernel((x, 2), (y, 3))
            @test isapprox(val_gg_23, true_val_gg_23, atol=1e-10)

            val_gg_32 = grad_kernel((x, 3), (y, 2))
            true_val_gg_32 = ∇₁₂kernel((x, 3), (y, 2))
            @test isapprox(val_gg_32, true_val_gg_32, atol=1e-10)

            val_gg_33 = grad_kernel((x, 3), (y, 3))
            true_val_gg_33 = ∇₁₂kernel((x, 3), (y, 3))
            @test isapprox(val_gg_33, true_val_gg_33, atol=1e-10)

            # Test symmetry for function-function case
            @test grad_kernel((x, 1), (y, 1)) ≈ grad_kernel((y, 1), (x, 1))
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
            test_x = [[0.25, 0.25]]
            mean_pred = posterior_mean(updated_gp, test_x)[1]
            var_pred = posterior_var(updated_gp, test_x)[1]
            grad_mean = posterior_grad_mean(updated_gp, test_x)
            grad_var = posterior_grad_var(updated_gp, test_x)

            grad_mean_pred = posterior_grad_mean(updated_gp, test_x)
            grad_var_pred = posterior_grad_var(updated_gp, test_x)
            grad_cov_pred = posterior_grad_cov(updated_gp, test_x)

            @test isa(mean_pred, Float64)
            @test isa(var_pred, Float64)
            @test var_pred >= 0.0
            @test isa(grad_mean_pred, AbstractVector)
            @test isa(grad_var_pred, AbstractVector)
            @test length(grad_mean_pred) == p  # function + gradients
            @test length(grad_var_pred) == p

            # Now checking compared to true values
            prepped_input = prep_input(gp, test_x)
            prepped_input_train = prep_input(gp, xs)
            grad_kernel = gradKernel(kernel_base)

            # Mean check
            # Creates Vector{Vector{Float64}} here
            k_xX = [grad_kernel.(Ref(prepped_input[i]), prepped_input_train) for i in 1:p]
            K̃ = kernelmatrix(grad_kernel, prepped_input_train) + noise_var * I
            ỹ = vec(permutedims(reduce(hcat, ys)))  # Convert to single vector with right ordering

            true_mean_post = reduce(vcat, permutedims.(k_xX)) * (K̃ \ ỹ)
            @test isapprox(grad_mean_pred, true_mean_post, atol=1e-10)

            # Covariance check
            # p x p matrix
            k_xx = grad_kernel.(permutedims(prepped_input), prepped_input)
            true_cov_post =
                k_xx - reduce(vcat, permutedims.(k_xX)) * (K̃ \ reduce(hcat, k_xX))

            @test isapprox(grad_cov_pred, true_cov_post, atol=1e-10)
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
                @test y_s[1]≈(y_orig[1] - μ[1]) / σ[1] atol=1e-8
                @test y_s[2]≈(y_orig[2] - μ[2]) / σ[2] atol=1e-8
                @test y_s[3]≈(y_orig[3] - μ[3]) / σ[3] atol=1e-8
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
    end

    @testset "LinearOperatorGP Tests" begin
        @testset "AbstractLinearOperator Construction" begin
            # Test IdentityOperator
            Id = IdentityOperator()
            @test isa(Id, IdentityOperator)

            # Test PartialDerivative
            ∂_1 = PartialDerivative(1)
            ∂_2 = PartialDerivative(2)
            @test isa(∂_1, PartialDerivative)
            @test ∂_1.dim == 1
            @test isa(∂_2, PartialDerivative)
            @test ∂_2.dim == 2

            # Test LaplacianOperator
            ∆ = LaplacianOperator()
            @test isa(∆, LaplacianOperator)
        end

        @testset "AbstractLinearOperator evaluation" begin
            # Functions used as reference 
            f(x) = x[1]^2 + x[2]^2 
            g(x) = x[1]^2
            h(x) = x[2]^2 + 0.25
            k(x) = x[1]*x[2]^2
            
            # Test IdentityOperator evaluation for a selected set of points S 
            S = [[-5, -5], [-5, 5], [5, -5], [5, 5], [-2.5, -2.5], [-2.5, 2.5], [2.5, -2.5], [2.5, 2.5], [0.0, 0.0]]
            f_S = [50, 50, 50, 50, 12.5, 12.5, 12.5, 12.5, 0.0]
            g_S = [25, 25, 25, 25, 6.25, 6.25, 6.25, 6.25, 0.0]
            h_S = [25.25, 25.25, 25.25, 25.25, 6.5, 6.5, 6.5, 6.5, 0.25]

            @test all(IdentityOperator()(f, x) == y for (x, y) in zip(S, f_S))
            @test all(IdentityOperator()(g, x) == y for (x, y) in zip(S, g_S))
            @test all(IdentityOperator()(h, x) == y for (x, y) in zip(S, h_S))

            # Test PartialDerivative evaluation over the same set S
            ∂_1f(x) = 2*x[1]
            ∂_2f(x) = 2*x[2]
            
            ∂_1g(x) = 2*x[1]
            ∂_2g(x) = 0.0

            ∂_1h(x) = 0.0
            ∂_2h(x) = 2*x[2]

            @test all(isapprox(PartialDerivative(1)(f, x), ∂_1f(x), atol=1e-10) && isapprox(PartialDerivative(2)(f, x), ∂_2f(x), atol=1e-10) for x in S)
            @test all(isapprox(PartialDerivative(1)(g, x), ∂_1g(x), atol=1e-10) && isapprox(PartialDerivative(2)(g, x), ∂_2g(x), atol=1e-10) for x in S)
            @test all(isapprox(PartialDerivative(1)(h, x), ∂_1h(x), atol=1e-10) && isapprox(PartialDerivative(2)(h, x), ∂_2h(x), atol=1e-10) for x in S)

            # Test LaplacianOperator evaluation over S
            ∆f(x) = 4.0
            ∆g(x) = 2.0
            ∆h(x) = 2.0
            ∆k(x) = 2*x[1]

            @test all(isapprox(LaplacianOperator()(f, x), ∆f(x), atol=1e-10) for x in S)
            @test all(isapprox(LaplacianOperator()(g, x), ∆g(x), atol=1e-10) for x in S)
            @test all(isapprox(LaplacianOperator()(h, x), ∆h(x), atol=1e-10) for x in S)
            @test all(isapprox(LaplacianOperator()(k, x), ∆k(x), atol=1e-10) for x in S)
        end

        @testset "Helper functions" begin
            @testset "_unitvec" begin
                n = 6
                vec = fill(0.5, n)
                
                @test all(length(_unitvec(vec, i)) == n && _unitvec(vec, i) == [j == i ? one(eltype(vec)) : zero(eltype(vec)) for j in 1:n]
                          for i in 1:n)
            end

            @testset "_scalar_mean" begin
                # Test _scalar_mean evaluation over the set m_set for the base mean m
                m_set = [[-5, 6.5], [68.43, -50], [13, 4], [2.4, 5.6]]
                m = CustomMean(x -> x[1] + x[2])

                @test all(isa(_scalar_mean(m, x), Float64) && isapprox(_scalar_mean(m, x), x[1] + x[2], atol=1e-10) for x in m_set)
            end

            @testset "_build_noise" begin
                # Test for _build_noise(noise_var::Real, xs::AbstractVector)
                noise_var_1 = 0.1
                xs_1 = [1.0, 2.3, 5.6]
                n_1 = length(xs_1)
                true_∑_1 = 0.1*Diagonal(ones(n_1))
                ∑_1 = _build_noise(noise_var_1, xs_1)
                @test isa(∑_1, Diagonal) && size(∑_1) == (n_1, n_1) && ∑_1 == true_∑_1 

                # Test for _build_noise(noise_var::Tuple, xs::AbstractVector{<:Tuple})
                noise_var_2 = (0.4, 0.1, 1.2)
                xs_2 = [(4.2, 2), (5.3, 1), (6.0, 2), (3.1, 3), (0.7, 1)]
                n_2 = length(xs_2)
                true_∑_2 = Diagonal([0.1, 0.4, 0.1, 1.2, 0.4])
                ∑_2 = _build_noise(noise_var_2, xs_2)

                @test isa(∑_2, Diagonal) && size(∑_2) == (n_2, n_2) && ∑_2 == true_∑_2

                # Test for _build_noise(noise_var::Tuple, xs::AbstractVector)
                noise_var_3 = (0.4, 0.1, 1.2)
                xs_3 = [4.2, 2.0, 15.1, 12.7, 2.7, 7.9]
                n_3 = length(xs_3)
                true_∑_3 = Diagonal([0.4, 0.4, 0.1, 0.1, 1.2, 1.2])
                ∑_3 = _build_noise(noise_var_3, xs_3)

                @test isa(∑_3, Diagonal) && size(∑_3) == (n_3, n_3) && ∑_3 == true_∑_3

                # Test for _build_noise(noise_var::AbstractVector, xs::AbstractVector)
                noise_var_4 = [0.3, 0.1, 1.3, 0.7, 1.5]
                xs_4 = [4.0, 1.9, 4.5, 21.8, 17.2]
                n_4 = length(xs_4)
                true_∑_4 = Diagonal(noise_var_4)
                ∑_4 = _build_noise(noise_var_4, xs_4)

                @test isa(∑_4, Diagonal) && size(∑_4) == (n_4, n_4) && ∑_4 == true_∑_4

                noise_var_4_bad = [0.3, 0.1, 1.3, 0.7]
                @test_throws ArgumentError _build_noise(noise_var_4_bad, xs_4)
            end

            @testset "_prep_input" begin
                xs = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
                p = 3
            
                # Test _prep_input(xs::AbstractVector{X}, p::Int)
                prepped_1 = _prep_input(xs, p)
                @test isa(prepped_1, KernelFunctions.MOInputIsotopicByOutputs)
                @test prepped_1.x == xs
                @test prepped_1.out_dim == p
            
                # Test _prep_input(xs::AbstractVector{<:Tuple}, p::Int)
                xs_tagged = [(x, i) for x in xs for i in 1:p]
                prepped_2 = _prep_input(xs_tagged, p)
                @test prepped_2 === xs_tagged
            
                # Test _prep_input(x::Tuple, p::Int)
                x_single = ([1.0, 2.0], 2)
                prepped_3 = _prep_input(x_single, p)
                @test isa(prepped_3, Vector)
                @test length(prepped_3) == 1
                @test prepped_3[1] === x_single
            
                # Test _prep_input(x::Real, p::Int)
                x_real = 1.5
                prepped_4 = _prep_input(x_real, p)
                @test isa(prepped_4, KernelFunctions.MOInputIsotopicByOutputs)
                @test prepped_4.x == [x_real]
                @test prepped_4.out_dim == p
            end
        end

        @testset "LinearOperatorGP Construction" begin
            # Base mean and kernel functions 
            base_mean = CustomMean(x -> (x[1]^2)*(x[2]^2) + x[1]^2 + x[2]^2)
            base_kernel = SqExponentialKernel()

            # Operators considered
            ops = [IdentityOperator(), PartialDerivative(1), PartialDerivative(2), LaplacianOperator()]

            # Test LinearOperatorMean construction
            linop_mean = LinearOperatorMean(base_mean, ops)
            @test isa(linop_mean, LinearOperatorMean)

            # Test LinearOperatorKernel construction
            linop_kernel = LinearOperatorKernel(base_kernel, ops)
            @test isa(linop_kernel, LinearOperatorKernel)

            # Test LinearOperatorGP construction
            noise_var = 0.1
            p = 4 # =length(ops)
            gp = LinearOperatorGP(base_kernel, ops, noise_var; mean = base_mean)

            @test gp.noise_var == noise_var
            @test gp.p == p
            @test gp.gpx === nothing
            @test isa(gp.gp, AbstractGPs.GP)
        end

        @testset "LinearOperatorGP Utility Functions" begin
            @testset "prep_input and prep_output" begin
                ops = [IdentityOperator(), PartialDerivative(1), PartialDerivative(2), LaplacianOperator()]
                kernel = SqExponentialKernel()
                noise_var = 0.1
                model = LinearOperatorGP(kernel, ops, noise_var)
            
                xs = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
                p = model.p
            
                # Test prep_input
                prepped = prep_input(model, xs)
                @test isa(prepped, KernelFunctions.MOInputIsotopicByOutputs)
                @test prepped.x == xs
                @test prepped.out_dim == p
            
                # Test prep_output
                ys = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
                prepped_out = prep_output(model, ys)
                true_out = vec(permutedims(reduce(hcat, ys)))
                @test isa(prepped_out, AbstractVector)
                @test length(prepped_out) == length(xs) * p
                @test prepped_out == true_out
            end

            @testset "get_scale, get_lengthscale and get_kernel_constructor" begin
                # Base mean and kernel functions 
                base_mean = CustomMean(x -> (x[1]^2)*(x[2]^2) + x[1]^2 + x[2]^2)
                base_kernel = SqExponentialKernel()

                # Operators considered
                ops = [IdentityOperator(), PartialDerivative(1), PartialDerivative(2), LaplacianOperator()]
                
                noise_var = 0.1
                p = 4 # =length(ops)
                gp = LinearOperatorGP(base_kernel, ops, noise_var; mean = base_mean)
                
                # Test the kernel with lengthscale and scale
                ℓ = get_lengthscale(gp)[1]
                scale = get_scale(gp)[1]
                @test ℓ == 1.0
                @test scale == 1.0

                # Test with custom lengthscale and scale
                custom_ℓ = 0.5
                custom_scale = 2.0
                custom_kernel = custom_scale * (with_lengthscale(base_kernel, custom_ℓ))
                gp_custom = LinearOperatorGP(custom_kernel, ops, noise_var)
                @test get_lengthscale(gp_custom) == [custom_ℓ]
                @test get_scale(gp_custom) == [custom_scale]
                @test gp_custom.noise_var == noise_var
                @test gp_custom.gpx === nothing
                @test isa(gp_custom.gp, AbstractGPs.GP)

                # Test with only lengthscale
                custom_ℓ_2 = 0.6
                ls_kernel = with_lengthscale(base_kernel, custom_ℓ_2)
                gp_ls = LinearOperatorGP(ls_kernel, ops, noise_var)
                @test get_lengthscale(gp_ls) == [custom_ℓ_2]
                @test get_scale(gp_ls) == [1.0] # default scale
                @test gp_ls.noise_var == noise_var
                @test gp_ls.gpx === nothing
                @test isa(gp_ls.gp, AbstractGPs.AbstractGP)

                # Test with only scale
                custom_scale_2 = 2.5
                sc_kernel = custom_scale_2 * base_kernel
                gp_sc = LinearOperatorGP(sc_kernel, ops, noise_var)
                @test get_lengthscale(gp_sc) == [1.0] # default length
                @test get_scale(gp_sc) == [custom_scale_2]
                @test gp_sc.noise_var == noise_var
                @test gp_sc.gpx === nothing
                @test isa(gp_sc.gp, AbstractGPs.AbstractGP)

                # Test get_kernel_constructor 
                ad_kernel_52 = ADMatern52Kernel()
                ad_kernel_72 = ADMatern72Kernel()
                gp_52 = LinearOperatorGP(ad_kernel_52, ops, noise_var)
                gp_72 = LinearOperatorGP(ad_kernel_72, ops, noise_var)

                @test isa(get_kernel_constructor(gp), SqExponentialKernel)
                @test isa(get_kernel_constructor(gp_52), ADMatern52Kernel)
                @test isa(get_kernel_constructor(gp_72), ADMatern72Kernel)
            end

            @testset "get_mean_std and std_y" begin
                base_kernel = SqExponentialKernel()
                ops = [IdentityOperator(), PartialDerivative(1), PartialDerivative(2), LaplacianOperator()]
                noise_var = 0.1
                gp = LinearOperatorGP(base_kernel, ops, noise_var)
            
                y_train = [[0.1, 1.0, 2.0, 0.5], [0.2, 2.0, 3.0, 1.5], [0.3, 3.0, 4.0, 2.5]]
            
                # Default: mean and scale
                μ, σ = get_mean_std(gp, y_train, "mean_scale")
                y_std = std_y(gp, y_train, μ, σ)
            
                @test length(μ) == gp.p
                @test length(σ) == gp.p
                @test length(y_std) == length(y_train)
                @test all(length(y) == gp.p for y in y_std)
            
                @test μ ≈ [0.2, 0.0, 0.0, 0.0]
                @test σ ≈ [0.1, 0.1, 0.1, 0.1]
            
                true_y_std = [[-1.0, 10.0, 20.0, 5.0], [ 0.0, 20.0, 30.0, 15.0], [ 1.0, 30.0, 40.0, 25.0]]
            
                @test y_std ≈ true_y_std
            
                # scale_only: mean forced to zero
                μ_scale, σ_scale = get_mean_std(gp, y_train, "scale_only")
                @test μ_scale ≈ zeros(gp.p)
                @test σ_scale ≈ [0.1, 0.1, 0.1, 0.1]
            
                # mean_only: std forced to one
                μ_mean, σ_mean = get_mean_std(gp, y_train, "mean_only")
                @test μ_mean ≈ [0.2, 0.0, 0.0, 0.0]
                @test σ_mean ≈ ones(gp.p)
            
                # Constant first output: zero std replaced by one
                y_constant = [[1.0, 2.0, 3.0, 4.0], [1.0, 5.0, 6.0, 7.0], [1.0, 8.0, 9.0, 10.0]]
            
                μ_const, σ_const = get_mean_std(gp, y_constant, "mean_scale")
            
                @test μ_const ≈ [1.0, 0.0, 0.0, 0.0]
                @test σ_const ≈ ones(gp.p)
            
                y_const_std = std_y(gp, y_constant, μ_const, σ_const)
            
                @test y_const_std ≈ [[0.0, 2.0, 3.0, 4.0], [0.0, 5.0, 6.0, 7.0], [0.0, 8.0, 9.0, 10.0]]
            end
        end

        @testset "LinearOperatorMean Functionality" begin
            base_mean = CustomMean(x -> x[1]^2 + x[2]^2)
            ops = [IdentityOperator(), PartialDerivative(1), PartialDerivative(2), LaplacianOperator()]
            linop_mean = LinearOperatorMean(base_mean, ops)
        
            x = [2.0, 3.0]
            xs_tagged = [(x, 1), (x, 2), (x, 3), (x, 4)]

            m_vec = AbstractGPs.mean_vector(linop_mean, xs_tagged)
            @test m_vec ≈ [13.0, 4.0, 6.0, 4.0]
        
            custom_mean = make_linear_operator_mean(base_mean, ops)
            m_custom_vec = AbstractGPs.mean_vector(custom_mean, xs_tagged)
            @test m_custom_vec ≈ m_vec
        end

        @testset "LinearOperatorKernel Functionality" begin
            base_kernel = SqExponentialKernel()
            ops = [IdentityOperator(), PartialDerivative(1), PartialDerivative(2), LaplacianOperator()]
            linop_kernel = LinearOperatorKernel(base_kernel, ops)
        
            x = [0.5, 0.5]
            y = [0.6, 0.7]
        
            ∇₁kernel(x, y) = ForwardDiff.gradient(a -> base_kernel(a, y), x)
            ∇₂kernel(x, y) = ForwardDiff.gradient(b -> base_kernel(x, b), y)
        
            ∂₁∂₂kernel(x, y, i, j) = ForwardDiff.derivative(
                h1 -> ForwardDiff.derivative(
                    h2 -> base_kernel(
                        x .+ h1 .* _unitvec(x, i),
                        y .+ h2 .* _unitvec(y, j),
                    ),
                    0.0,
                ),
                0.0,
            )
        
            Δ₁kernel(x, y) = tr(ForwardDiff.hessian(a -> base_kernel(a, y), x))
            Δ₂kernel(x, y) = tr(ForwardDiff.hessian(b -> base_kernel(x, b), y))

            ∂ᵢΔ₂kernel(x, y, i) = ForwardDiff.derivative(h -> Δ₂kernel(x .+ h .* _unitvec(x, i), y), 0.0)
            Δ₁∂ᵢkernel(x, y, j) = tr(ForwardDiff.hessian(a -> ∇₂kernel(a, y)[j], x))

            Δ₁Δ₂kernel(x, y) = tr(ForwardDiff.hessian(a -> Δ₂kernel(a, y), x))
        
            # Identity-Identity
            @test linop_kernel((x, 1), (y, 1)) ≈ base_kernel(x, y)
        
            # Identity-PartialDerivative
            @test linop_kernel((x, 1), (y, 2)) ≈ ∇₂kernel(x, y)[1]
            @test linop_kernel((x, 1), (y, 3)) ≈ ∇₂kernel(x, y)[2]
        
            # PartialDerivative-Identity
            @test linop_kernel((x, 2), (y, 1)) ≈ ∇₁kernel(x, y)[1]
            @test linop_kernel((x, 3), (y, 1)) ≈ ∇₁kernel(x, y)[2]
        
            # PartialDerivative-PartialDerivative
            @test linop_kernel((x, 2), (y, 2)) ≈ ∂₁∂₂kernel(x, y, 1, 1)
            @test linop_kernel((x, 2), (y, 3)) ≈ ∂₁∂₂kernel(x, y, 1, 2)
            @test linop_kernel((x, 3), (y, 2)) ≈ ∂₁∂₂kernel(x, y, 2, 1)
            @test linop_kernel((x, 3), (y, 3)) ≈ ∂₁∂₂kernel(x, y, 2, 2)
        
            # Identity-Laplacian and Laplacian-Identity
            @test linop_kernel((x, 1), (y, 4)) ≈ Δ₂kernel(x, y)
            @test linop_kernel((x, 4), (y, 1)) ≈ Δ₁kernel(x, y)
            
            # PartialDerivative-Laplacian
            @test linop_kernel((x, 2), (y, 4)) ≈ ∂ᵢΔ₂kernel(x, y, 1)
            @test linop_kernel((x, 3), (y, 4)) ≈ ∂ᵢΔ₂kernel(x, y, 2)

            # Laplacian-PartialDerivative
            @test linop_kernel((x, 4), (y, 2)) ≈ Δ₁∂ᵢkernel(x, y, 1)
            @test linop_kernel((x, 4), (y, 3)) ≈ Δ₁∂ᵢkernel(x, y, 2)

            # Laplacian-Laplacian
            @test linop_kernel((x, 4), (y, 4)) ≈ Δ₁Δ₂kernel(x, y)

            # Symmetry for function-function case
            @test linop_kernel((x, 1), (y, 1)) ≈ linop_kernel((y, 1), (x, 1))
        end

        @testset "LinearOperatorGP Updates" begin
            @testset "Full observation case" begin
                base_kernel = SqExponentialKernel()
                ops = [IdentityOperator(), PartialDerivative(1), PartialDerivative(2), LaplacianOperator()]
                noise_var = 0.1
                gp = LinearOperatorGP(base_kernel, ops, noise_var)
            
                xs = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
                ys = [[1.0, 0.1, 0.1, 2.0], [0.5, 0.0, 0.0, 2.0], [0.0, -0.1, -0.1, 2.0]]
                updated_gp = update(gp, xs, ys)
            
                # Return type and immutability of update
                @test isa(updated_gp, LinearOperatorGP)
                @test updated_gp !== gp
            
                # Fields preserved
                @test updated_gp.gp === gp.gp
                @test updated_gp.noise_var == gp.noise_var
                @test updated_gp.ops === gp.ops
                @test updated_gp.p == gp.p
            
                # Update actually conditions the model
                @test gp.gpx === nothing
                @test updated_gp.gpx !== nothing
                @test isa(updated_gp.gpx, AbstractGPs.PosteriorGP)
            
                # Full-output data is converted to n*p scalar observations
                x_tilde, y_tilde = prepare_isotopic_multi_output_data(xs, ColVecs(reduce(hcat, ys)))
                @test length(x_tilde) == length(xs) * gp.p
                @test length(y_tilde) == length(xs) * gp.p
            end

            @testset "Heterogeneous observation case" begin
                base_kernel = SqExponentialKernel()
                ops = [IdentityOperator(), PartialDerivative(1), PartialDerivative(2), LaplacianOperator()]
                noise_var = 0.1
                gp = LinearOperatorGP(base_kernel, ops, noise_var)
            
                xs = [
                    ([0.0, 0.0], 1),
                    ([0.5, 0.5], 2),
                    ([1.0, 1.0], 3),
                    ([1.5, 1.5], 4),
                    ([2.0, 2.0], 1),
                ]
            
                ys = [1.0, 0.1, -0.1, 2.0, 0.0]
            
                updated_gp = update(gp, xs, ys)
            
                # Return type and immutability of update
                @test isa(updated_gp, LinearOperatorGP)
                @test updated_gp !== gp
            
                # Fields preserved
                @test updated_gp.gp === gp.gp
                @test updated_gp.noise_var == gp.noise_var
                @test updated_gp.ops === gp.ops
                @test updated_gp.p == gp.p
            
                # Update actually conditions the model
                @test gp.gpx === nothing
                @test updated_gp.gpx !== nothing
                @test isa(updated_gp.gpx, AbstractGPs.PosteriorGP)
            
                # Heterogeneous data is already scalar/tagged
                @test length(xs) == length(ys)
            
                # Tagged inputs should be passed through unchanged by _prep_input
                @test _prep_input(xs, gp.p) === xs
            
                # Length mismatch should throw
                ys_bad = ys[1:end-1]
                @test_throws ArgumentError update(gp, xs, ys_bad)
            end
        end

        @testset "LinearOperatorGP Posterior Functionality" begin
            base_kernel = SqExponentialKernel()
            ops = [IdentityOperator(), PartialDerivative(1), PartialDerivative(2), LaplacianOperator()]
            noise_var = 0.1
            gp = LinearOperatorGP(base_kernel, ops, noise_var)
            
            # Create training data with full observation
            xs = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
            ys = [[1.0, 0.1, 0.1, 2.0], [0.5, 0.0, 0.0, 2.0], [0.0, -0.1, -0.1, 2.0]]
        
            # Updated GP
            updated_gp = update(gp, xs, ys)
        
            # Test predictions
            test_x = [0.25, 1.25]
        
            mean_preds = [posterior_mean(updated_gp, test_x, i) for i in 1:gp.p]
            var_preds = [posterior_var(updated_gp, test_x, i) for i in 1:gp.p]
            
            # Type, length, and variance checks
            for i in 1:gp.p
                @test isa(mean_preds[i], AbstractVector)
                @test isa(var_preds[i], AbstractVector)
                @test length(mean_preds[i]) == 1
                @test length(var_preds[i]) == 1
                @test all(isfinite, mean_preds[i])
                @test all(isfinite, var_preds[i])
                @test all(var_preds[i] .≥ 0)
            end
            
            # Manual posterior checks for all operators i = 1, ..., p
            linop_kernel = updated_gp.gp.kernel
            x_tilde, y_tilde = prepare_isotopic_multi_output_data(xs, ColVecs(reduce(hcat, ys)))
            K̃ = kernelmatrix(linop_kernel, x_tilde) + noise_var * I
        
            x0 = test_x
        
            for i in 1:gp.p
                test_i = (x0, i)
        
                k_xX_i = [linop_kernel(test_i, x_train) for x_train in x_tilde]
        
                true_mean_i = k_xX_i' * (K̃ \ y_tilde)
                true_var_i = linop_kernel(test_i, test_i) - k_xX_i' * (K̃ \ k_xX_i)
        
                @test isapprox(mean_preds[i][1], true_mean_i, atol=1e-10)
                @test isapprox(var_preds[i][1], true_var_i, atol=1e-10)
            end
            
            # Default i omitted should match i = 1
            mean_pred_default = posterior_mean(updated_gp, test_x)
            var_pred_default = posterior_var(updated_gp, test_x)
        
            @test mean_pred_default ≈ mean_preds[1]
            @test var_pred_default ≈ var_preds[1]
        end

        @testset "LinearOperatorGP Copy" begin
            base_kernel = SqExponentialKernel()
            ops = [IdentityOperator(), PartialDerivative(1), PartialDerivative(2), LaplacianOperator()]
            noise_var = 0.1
            gp = LinearOperatorGP(base_kernel, ops, noise_var)

            xs = [[0.5, 1.3], [23.1, 5.8], [3.8, 4.0]]
            ys = [[31.0, 0.0, 1.9, 3.0], [6.7, 5.0, 12.3, 4.2], [6.8, 13.4, 20.0, 9.2]]
            
            # Unconditioned model
            copied_gp = copy(gp)
            @test copied_gp.noise_var == gp.noise_var
            @test copied_gp.gp === gp.gp
            @test copied_gp.ops === gp.ops
            @test copied_gp.gpx === nothing

            # Conditioned model
            updated_gp = update(gp, xs, ys)
            copied_updated_gp = copy(updated_gp)

            @test copied_updated_gp.gp === updated_gp.gp
            @test copied_updated_gp.ops === updated_gp.ops
            @test copied_updated_gp.noise_var == updated_gp.noise_var
            @test copied_updated_gp.p == updated_gp.p
            @test copied_updated_gp.gpx !== updated_gp.gpx
        end

        @testset "LinearOperatorGP nlml" begin
            base_kernel = SqExponentialKernel()
            ops = [IdentityOperator(), PartialDerivative(1), PartialDerivative(2), LaplacianOperator()]
            noise_var = 0.1
        
            kernel = 1.5 * with_lengthscale(base_kernel, 0.8)
            gp = LinearOperatorGP(kernel, ops, noise_var)
        
            params = log.([0.8, 1.5])
        
            @testset "Full observation case" begin
                xs = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
                ys = [
                    [1.0, 0.1, 0.1, 2.0],
                    [0.5, 0.0, 0.0, 2.0],
                    [0.0, -0.1, -0.1, 2.0],
                ]
        
                val = nlml(gp, params, xs, ys)
        
                x_tilde, y_tilde = prepare_isotopic_multi_output_data(xs, ColVecs(reduce(hcat, ys)))
                noise = _build_noise(noise_var, x_tilde)
                gpx = gp.gp(x_tilde, noise)
        
                true_val = -AbstractGPs.logpdf(gpx, y_tilde)
        
                @test isa(val, Real)
                @test isfinite(val)
                @test isapprox(val, true_val, atol=1e-10)

                params_wrong = log.([0.3, 5.0])
                val_wrong = nlml(gp, params_wrong, xs, ys)
                @test isfinite(val_wrong)
                @test val_wrong != val
            end
        
            @testset "Heterogeneous observation case" begin
                xs = [
                    ([0.0, 0.0], 1),
                    ([0.5, 0.5], 2),
                    ([1.0, 1.0], 3),
                    ([1.5, 1.5], 4),
                    ([2.0, 2.0], 1),
                ]
        
                ys = [1.0, 0.1, -0.1, 2.0, 0.0]
        
                val = nlml(gp, params, xs, ys)
        
                noise = _build_noise(noise_var, xs)
                gpx = gp.gp(xs, noise)
        
                true_val = -AbstractGPs.logpdf(gpx, ys)
        
                @test isa(val, Real)
                @test isfinite(val)
                @test isapprox(val, true_val, atol=1e-10)
                params_wrong = log.([0.3, 5.0])
                val_wrong = nlml(gp, params_wrong, xs, ys)
                @test isfinite(val_wrong)
                @test val_wrong != val
            end
        end
    end
end
