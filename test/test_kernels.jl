using Test
using AbstractBayesOpt
using AbstractGPs, KernelFunctions
using ForwardDiff
using LinearAlgebra
using Random

Random.seed!(1234)

@testset "Matern 5/2 Tests" begin

    ## definition of reference kernel and GP
    d = 2
    ℓ = 2.0
    σ² = 4.0
    reference_kernel = σ²*with_lengthscale(Matern52Kernel(), ℓ)
    GP_ref = GradientGP(reference_kernel, d+1, 0)

    # Hyperparameter testing for construction
    approx_kernel = σ²*with_lengthscale(ApproxMatern52Kernel(), ℓ)
    GP_approx = GradientGP(approx_kernel, d+1, 0)

    ad_kernel = σ²*with_lengthscale(ADMatern52Kernel(), ℓ)
    GP_ad = GradientGP(ad_kernel, d+1, 0)

    @test AbstractBayesOpt.get_lengthscale(GP_ref)[1] ==
        AbstractBayesOpt.get_lengthscale(GP_approx)[1] ==
        2.0
    @test AbstractBayesOpt.get_scale(GP_ref)[1] ==
        AbstractBayesOpt.get_scale(GP_approx)[1] ==
        4.0
    @test AbstractBayesOpt.get_lengthscale(GP_ad)[1] == 2.0
    @test AbstractBayesOpt.get_scale(GP_ad)[1] == 4.0

    x1 = rand(d)
    x2 = rand(d)
    X = [rand(2) for i in 1:5]
    X_prepped = AbstractBayesOpt.prep_input(GP_approx, X)

    @testset "Kernel value" begin
        ## at two different points
        @test isapprox(approx_kernel(x1, x2), reference_kernel(x1, x2); atol=1e-12)
        @test isapprox(ad_kernel(x1, x2), reference_kernel(x1, x2); atol=1e-12)

        ### check hyperparameter scaling
        ref_val = σ²*KernelFunctions.kappa(Matern52Kernel(), norm(x1 - x2)/ℓ)
        @test isapprox(approx_kernel(x1, x2), ref_val; atol=1e-12)
        @test isapprox(ad_kernel(x1, x2), ref_val; atol=1e-12)

        ## at the same point
        @test isapprox(approx_kernel(x1, x1), reference_kernel(x1, x1); atol=1e-12)
        @test isapprox(ad_kernel(x1, x1), reference_kernel(x1, x1); atol=1e-12)

        ref_val = σ²*KernelFunctions.kappa(Matern52Kernel(), 0.0)
        @test isapprox(approx_kernel(x1, x1), ref_val; atol=1e-12)
        @test isapprox(ad_kernel(x1, x1), ref_val; atol=1e-12)

        # Check kernelmatrix
        K_ref = kernelmatrix(GP_ref.gp.kernel, X_prepped) # this produces NaNs for x==y, as expected
        K_approx = kernelmatrix(GP_approx.gp.kernel, X_prepped)
        K_ad = kernelmatrix(GP_ad.gp.kernel, X_prepped)
        @test isapprox(K_approx, K_ad; atol=1e-12)
    end # of kernel value tests

    @testset "Kernel gradients" begin
        ∇k_ref(x, y) = ForwardDiff.gradient(z -> reference_kernel(x, z), y)
        ∇k_approx(x, y) = ForwardDiff.gradient(z -> approx_kernel(x, z), y)
        ∇k_ad(x, y) = ForwardDiff.gradient(z -> ad_kernel(x, z), y)

        ## hyperparameter tests (over second variable to match above)
        ∇k_ref_val(x, y) =
            σ²*ForwardDiff.derivative(
                z -> KernelFunctions.kappa(Matern52Kernel(), z/ℓ), norm(x-y)
            )*(y-x)/norm(x-y)
        ref_val = ∇k_ref_val(x1, x2)

        @test isapprox(∇k_ref(x1, x2), ref_val; atol=1e-12)
        @test isapprox(∇k_approx(x1, x2), ref_val; atol=1e-12)
        @test isapprox(∇k_ad(x1, x2), ref_val; atol=1e-12)

        ## at two different points
        @test isapprox(∇k_approx(x1, x2), ∇k_ref(x1, x2); atol=1e-12)
        @test isapprox(∇k_ad(x1, x2), ∇k_ref(x1, x2); atol=1e-12)
        ## at the same point (should be zero)
        @test isapprox(∇k_approx(x1, x1), [0.0; 0.0]; atol=1e-12)
        @test isapprox(∇k_ad(x1, x1), [0.0; 0.0]; atol=1e-12)
        @test all(isnan.(∇k_ref(x1, x1))) # this produces NaNs, as expected
    end # of kernel gradient tests

    @testset "Posteriors" begin

        # Check posterior mean value and derivatives
        f(x) = sin(π*x[1])*cos(π*x[2])
        ∇f(x) = ForwardDiff.gradient(f, x)
        f_val_grad(x) = [f(x); ∇f(x)]
        y_∂y = [f_val_grad(x) for x in X]

        post_approx = AbstractBayesOpt.update(GP_approx, X, y_∂y)
        post_ad = AbstractBayesOpt.update(GP_ad, X, y_∂y)

        post_mean_approx(x) = posterior_mean(post_approx, [x])[1]
        post_mean_ad(x) = posterior_mean(post_ad, [x])[1]
        ∇post_mean_approx(x) = ForwardDiff.gradient(z -> post_mean_approx(z), x)
        ∇post_mean_ad(x) = ForwardDiff.gradient(z -> post_mean_ad(z), x)
        hessian_post_mean_approx(x) = ForwardDiff.hessian(z -> post_mean_approx(z), x)
        hessian_post_mean_ad(x) = ForwardDiff.hessian(z -> post_mean_ad(z), x)

        ## at a test point
        # see that the more we derive, the more tolerance we need to give away.
        @test isapprox(post_mean_approx(x1), post_mean_ad(x1); atol=1e-12)
        @test isapprox(∇post_mean_approx(x1), ∇post_mean_ad(x1); atol=1e-10)
        @test isapprox(hessian_post_mean_approx(x1), hessian_post_mean_ad(x1); atol=1e-8)

        ## at a training point
        x_train = X[1]
        @test isapprox(post_mean_approx(x_train), post_mean_ad(x_train); atol=1e-12)
        @test isapprox(∇post_mean_approx(x_train), ∇post_mean_ad(x_train); atol=1e-10)
        @test !isapprox(
            hessian_post_mean_approx(x_train), hessian_post_mean_ad(x_train); atol=1e-8
        )
        @test all(isnan.(hessian_post_mean_ad(x_train))) # this produces NaNs, as expected
        # One should not believe in the Hessian at a training point, for the Matern 5/2 kernel!

        # posterior gradient mean
        post_grad_mean_approx(x) = posterior_grad_mean(post_approx, [x])
        post_grad_mean_ad(x) = posterior_grad_mean(post_ad, [x])

        ## at a test point
        @test isapprox(post_grad_mean_approx(x1), post_grad_mean_ad(x1); atol=1e-10)

        ## at a training point
        @test isapprox(
            post_grad_mean_approx(x_train), post_grad_mean_ad(x_train); atol=1e-10
        )

        # Check posterior variance value
        post_var_approx(x) = posterior_var(post_approx, [x])[1]
        post_var_ad(x) = posterior_var(post_ad, [x])[1]

        ## at a test point
        @test isapprox(post_var_approx(x1), post_var_ad(x1); atol=1e-10)
        @test post_var_approx(x1) > 0.0
        @test post_var_ad(x1) > 0.0

        ## at a training point
        @test isapprox(post_var_approx(x_train), post_var_ad(x_train); atol=1e-10)
        @test isapprox(post_var_approx(x_train), 0.0; atol=1e-10)
        @test isapprox(post_var_ad(x_train), 0.0; atol=1e-10)

        # Check posterior gradient variance value
        post_grad_var_approx(x) = posterior_grad_cov(post_approx, [x])
        post_grad_var_ad(x) = posterior_grad_cov(post_ad, [x])
        ## at a test point
        @test isapprox(post_grad_var_approx(x1), post_grad_var_ad(x1); atol=1e-10)

        ## at a training point
        @test isapprox(post_grad_var_approx(x_train), post_grad_var_ad(x_train); atol=1e-10)
    end # of posteriors tests

    @testset "Printing tests" begin
        io = IOBuffer()

        show(io, approx_kernel)
        output = String(take!(io))
        @test output ==
            "Matern 5/2 Kernel, quadratic approximation around d=0 (metric = Distances.SqEuclidean(0.0))\n\t- Scale Transform (s = $(1/ℓ))\n\t- σ² = $σ²"

        show(io, ad_kernel)
        output = String(take!(io))
        @test output ==
            "Matern 5/2 Kernel with AD support (metric = Distances.SqEuclidean(0.0))\n\t- Scale Transform (s = $(1/ℓ))\n\t- σ² = $σ²"
    end # of printing tests
end # of Matern 5/2 tests

@testset "Matern 7/2 Tests" begin

    ## definition of reference kernel and GP
    d = 2
    ℓ = 2.0
    σ² = 4.0
    reference_kernel = σ²*with_lengthscale(Matern72Kernel(), ℓ)
    GP_ref = GradientGP(reference_kernel, d+1, 0)

    # Hyperparameter testing for construction
    approx_kernel = σ²*with_lengthscale(ApproxMatern72Kernel(), ℓ)
    GP_approx = GradientGP(approx_kernel, d+1, 0)

    ad_kernel = σ²*with_lengthscale(ADMatern72Kernel(), ℓ)
    GP_ad = GradientGP(ad_kernel, d+1, 0)

    @test AbstractBayesOpt.get_lengthscale(GP_ref)[1] ==
        AbstractBayesOpt.get_lengthscale(GP_approx)[1] ==
        2.0
    @test AbstractBayesOpt.get_scale(GP_ref)[1] ==
        AbstractBayesOpt.get_scale(GP_approx)[1] ==
        4.0
    @test AbstractBayesOpt.get_lengthscale(GP_ad)[1] == 2.0
    @test AbstractBayesOpt.get_scale(GP_ad)[1] == 4.0

    x1 = rand(d)
    x2 = rand(d)
    X = [rand(2) for i in 1:5]
    X_prepped = AbstractBayesOpt.prep_input(GP_approx, X)

    @testset "Kernel value tests" begin
        # Kernel value tests
        ## at two different points
        @test isapprox(approx_kernel(x1, x2), reference_kernel(x1, x2); atol=1e-12)
        @test isapprox(ad_kernel(x1, x2), reference_kernel(x1, x2); atol=1e-12)

        ### check hyperparameter scaling
        ref_val = σ²*KernelFunctions.kappa(Matern72Kernel(), norm(x1 - x2)/ℓ)
        @test isapprox(approx_kernel(x1, x2), ref_val; atol=1e-12)
        @test isapprox(ad_kernel(x1, x2), ref_val; atol=1e-12)

        ## at the same point
        @test isapprox(approx_kernel(x1, x1), reference_kernel(x1, x1); atol=1e-12)
        @test isapprox(ad_kernel(x1, x1), reference_kernel(x1, x1); atol=1e-12)

        ref_val = σ²*KernelFunctions.kappa(Matern72Kernel(), 0.0)
        @test isapprox(approx_kernel(x1, x1), ref_val; atol=1e-12)
        @test isapprox(ad_kernel(x1, x1), ref_val; atol=1e-12)

        # Check kernelmatrix
        K_ref = kernelmatrix(GP_ref.gp.kernel, X_prepped) # this produces NaNs for x==y, as expected
        K_approx = kernelmatrix(GP_approx.gp.kernel, X_prepped)
        K_ad = kernelmatrix(GP_ad.gp.kernel, X_prepped)
        @test isapprox(K_approx, K_ad; atol=1e-12)
    end # of kernel value tests

    @testset "Kernel gradient tests" begin
        ∇k_ref(x, y) = ForwardDiff.gradient(z -> reference_kernel(x, z), y)
        ∇k_approx(x, y) = ForwardDiff.gradient(z -> approx_kernel(x, z), y)
        ∇k_ad(x, y) = ForwardDiff.gradient(z -> ad_kernel(x, z), y)

        ## hyperparameter tests (over second variable to match above)
        ∇k_ref_val(x, y) =
            σ²*ForwardDiff.derivative(
                z -> KernelFunctions.kappa(Matern72Kernel(), z/ℓ), norm(x-y)
            )*(y-x)/norm(x-y)
        ref_val = ∇k_ref_val(x1, x2)

        @test isapprox(∇k_ref(x1, x2), ref_val; atol=1e-12)
        @test isapprox(∇k_approx(x1, x2), ref_val; atol=1e-12)
        @test isapprox(∇k_ad(x1, x2), ref_val; atol=1e-12)

        ## at two different points
        @test isapprox(∇k_approx(x1, x2), ∇k_ref(x1, x2); atol=1e-12)
        @test isapprox(∇k_ad(x1, x2), ∇k_ref(x1, x2); atol=1e-12)
        ## at the same point (should be zero)
        @test isapprox(∇k_approx(x1, x1), [0.0; 0.0]; atol=1e-12)
        @test isapprox(∇k_ad(x1, x1), [0.0; 0.0]; atol=1e-12)
        @test all(isnan.(∇k_ref(x1, x1))) # this produces NaNs, as expected
    end # of kernel gradient tests

    @testset "Posteriors tests" begin

        # Check posterior mean derivatives
        f(x) = sin(π*x[1])*cos(π*x[2])
        ∇f(x) = ForwardDiff.gradient(f, x)
        f_val_grad(x) = [f(x); ∇f(x)]
        y_∂y = [f_val_grad(x) for x in X]

        post_approx = AbstractBayesOpt.update(GP_approx, X, y_∂y)
        post_ad = AbstractBayesOpt.update(GP_ad, X, y_∂y)

        post_mean_approx(x) = posterior_mean(post_approx, [x])[1]
        post_mean_ad(x) = posterior_mean(post_ad, [x])[1]
        ∇post_mean_approx(x) = ForwardDiff.gradient(z -> post_mean_approx(z), x)
        ∇post_mean_ad(x) = ForwardDiff.gradient(z -> post_mean_ad(z), x)
        hessian_post_mean_approx(x) = ForwardDiff.hessian(z -> post_mean_approx(z), x)
        hessian_post_mean_ad(x) = ForwardDiff.hessian(z -> post_mean_ad(z), x)

        ## at a test point
        # see that the more we derive, the more tolerance we need to give away.
        @test isapprox(post_mean_approx(x1), post_mean_ad(x1); atol=1e-9)
        @test isapprox(∇post_mean_approx(x1), ∇post_mean_ad(x1); atol=1e-8)
        @test isapprox(hessian_post_mean_approx(x1), hessian_post_mean_ad(x1); atol=1e-7)

        ## at a training point
        x_train = X[1]
        @test isapprox(post_mean_approx(x_train), post_mean_ad(x_train); atol=1e-9)
        @test isapprox(∇post_mean_approx(x_train), ∇post_mean_ad(x_train); atol=1e-8)
        @test isapprox(
            hessian_post_mean_approx(x_train), hessian_post_mean_ad(x_train); atol=1e-7
        )
        # This time, the Hessian makes sense, as the Matern 7/2 kernel is thrice differentiable.

        # posterior gradient mean
        post_grad_mean_approx(x) = posterior_grad_mean(post_approx, [x])
        post_grad_mean_ad(x) = posterior_grad_mean(post_ad, [x])

        ## at a test point
        @test isapprox(post_grad_mean_approx(x1), post_grad_mean_ad(x1); atol=1e-8)

        ## at a training point
        @test isapprox(
            post_grad_mean_approx(x_train), post_grad_mean_ad(x_train); atol=1e-8
        )

        # Check posterior variance value
        post_var_approx(x) = posterior_var(post_approx, [x])[1]
        post_var_ad(x) = posterior_var(post_ad, [x])[1]

        ## at a test point
        @test isapprox(post_var_approx(x1), post_var_ad(x1); atol=1e-8)
        @test post_var_approx(x1) > 0.0
        @test post_var_ad(x1) > 0.0

        ## at a training point
        @test isapprox(post_var_approx(x_train), post_var_ad(x_train); atol=1e-8)
        @test isapprox(post_var_approx(x_train), 0.0; atol=1e-8)
        @test isapprox(post_var_ad(x_train), 0.0; atol=1e-8)

        # Check posterior gradient variance value
        post_grad_var_approx(x) = posterior_grad_cov(post_approx, [x])
        post_grad_var_ad(x) = posterior_grad_cov(post_ad, [x])
        ## at a test point
        @test isapprox(post_grad_var_approx(x1), post_grad_var_ad(x1); atol=1e-10)

        ## at a training point
        @test isapprox(post_grad_var_approx(x_train), post_grad_var_ad(x_train); atol=1e-10)
    end # of posteriors tests

    @testset "Printing tests" begin
        io = IOBuffer()

        show(io, approx_kernel)
        output = String(take!(io))
        @test output ==
            "Matern 7/2 Kernel, Taylor approximation around d=0 (metric = Distances.SqEuclidean(0.0))\n\t- Scale Transform (s = $(1/ℓ))\n\t- σ² = $σ²"

        show(io, ad_kernel)
        output = String(take!(io))
        @test output ==
            "Matern 7/2 Kernel with AD support (metric = Distances.SqEuclidean(0.0))\n\t- Scale Transform (s = $(1/ℓ))\n\t- σ² = $σ²"
    end # of printing tests
end # of Matern 7/2 tests
