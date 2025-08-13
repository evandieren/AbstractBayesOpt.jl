"""
Test file 2: Computing nlml gradients analytically for GradientGP

This file explores computing the gradient of the negative log marginal likelihood
analytically rather than using automatic differentiation. This should avoid the
dual number conflicts entirely.

The gradient of the log marginal likelihood for a GP is:
∂/∂θ log p(y|X,θ) = 0.5 * tr((α*α^T - K^(-1)) * ∂K/∂θ)
where α = K^(-1) * y and K is the covariance matrix.

For the kernel hyperparameters [log(ℓ), log(σ²)]:
- ∂K/∂(log ℓ) involves derivatives w.r.t. lengthscale
- ∂K/∂(log σ²) = K (since σ² multiplies the entire kernel)
"""

using AbstractGPs
using KernelFunctions
using ForwardDiff
using BayesOpt
using Optim
using LinearAlgebra
using Statistics

# Test function
f(x) = sin(sum(x.+1)) + sin((10.0 / 3.0) * sum(x .+1))
∇f(x) = ForwardDiff.gradient(f, x)
f_val_grad(x) = [f(x); ∇f(x)]

# Simple 1D test case
d = 1
lower = [-2.0]
upper = [2.0]
n_train = 5
x_train = [lower .+ (upper .- lower) .* rand(d) for _ in 1:n_train]
y_train = [f_val_grad(x) for x in x_train]

println("Testing analytical gradient computation for GradientGP...")
println("Training data:")
println("x_train = ", x_train)
println("y_train = ", y_train)

# Setup GradientGP
kernel_constructor = ApproxMatern52Kernel()
gp_model = GradientGP(gradKernel(kernel_constructor), d+1, 1e-12)

# Prepare data
x_train_prepped = KernelFunctions.MOInputIsotopicByOutputs(x_train, length(y_train[1]))
y_train_prepped = vec(permutedims(reduce(hcat, y_train)))

println("\nPrepared data shapes:")
println("x_train_prepped: ", typeof(x_train_prepped))
println("y_train_prepped length: ", length(y_train_prepped))

"""
Analytical gradient computation for nlml w.r.t. log-hyperparameters
"""
function nlml_and_gradient(gp_model, params, kernel_constructor, x, y; mean=ZeroMean())
    log_ℓ, log_scale = params
    ℓ = exp(log_ℓ)
    scale = exp(log_scale)
    
    # Build kernel with current parameters
    k = scale * (kernel_constructor ∘ ScaleTransform(1/ℓ))
    gp = GradientGP(gradKernel(k), gp_model.p, gp_model.noise_var, mean=mean)
    
    # Create the finite GP
    gpx = gp.gp(x, gp_model.noise_var)
    
    # Compute the covariance matrix
    K = KernelFunctions.kernelmatrix(gp.gp.kernel, x) + gp_model.noise_var * I
    
    # Compute α = K^(-1) * y
    α = K \ y
    
    # Compute log marginal likelihood
    nlml_val = 0.5 * (y' * α + logdet(K) + length(y) * log(2π))
    
    # Compute gradients
    # For computational efficiency, compute K^(-1) once
    K_inv = inv(K)
    
    # Common term: α*α^T - K^(-1)
    αα_minus_Kinv = α * α' - K_inv
    
    # Gradient w.r.t. log(scale): ∂K/∂(log σ²) = K
    grad_log_scale = 0.5 * tr(αα_minus_Kinv * (K - gp_model.noise_var * I))
    
    # Gradient w.r.t. log(lengthscale): ∂K/∂(log ℓ) 
    # This is more complex for the gradient kernel, we'll approximate it
    h = 1e-8
    ℓ_plus = ℓ * exp(h)
    k_plus = scale * (kernel_constructor ∘ ScaleTransform(1/ℓ_plus))
    gp_plus = GradientGP(gradKernel(k_plus), gp_model.p, gp_model.noise_var, mean=mean)
    K_plus = KernelFunctions.kernelmatrix(gp_plus.gp.kernel, x) + gp_model.noise_var * I
    
    dK_dlog_ℓ = (K_plus - K) / h
    grad_log_ℓ = 0.5 * tr(αα_minus_Kinv * dK_dlog_ℓ)
    
    return nlml_val, [grad_log_ℓ, grad_log_scale]
end

"""
More robust analytical gradient using finite differences for the lengthscale derivative
"""
function nlml_with_finite_diff_gradient(gp_model, params, kernel_constructor, x, y; mean=ZeroMean())
    function nlml_func(p)
        log_ℓ, log_scale = p
        ℓ = exp(log_ℓ)
        scale = exp(log_scale)
        
        k = scale * (kernel_constructor ∘ ScaleTransform(1/ℓ))
        gp = GradientGP(gradKernel(k), gp_model.p, gp_model.noise_var, mean=mean)
        gpx = gp.gp(x, gp_model.noise_var)
        
        return -AbstractGPs.logpdf(gpx, y)
    end
    
    # Compute function value
    nlml_val = nlml_func(params)
    
    # Compute gradient using finite differences
    h = 1e-8
    gradient = zeros(length(params))
    
    for i in 1:length(params)
        params_plus = copy(params)
        params_minus = copy(params)
        params_plus[i] += h
        params_minus[i] -= h
        
        gradient[i] = (nlml_func(params_plus) - nlml_func(params_minus)) / (2*h)
    end
    
    return nlml_val, gradient
end

# Test parameters
old_params = [log(1.0), log(1.0)]

println("\n=== Testing analytical gradient computation ===")

try
    nlml_val, grad = nlml_and_gradient(gp_model, old_params, kernel_constructor, 
                                      x_train_prepped, y_train_prepped)
    
    println("SUCCESS: Analytical gradient computed!")
    println("NLML value: ", nlml_val)
    println("Gradient: ", grad)
    
    # Test optimization with analytical gradient
    function objective_with_gradient!(F, G, p)
        if G !== nothing
            val, grad = nlml_and_gradient(gp_model, p, kernel_constructor, 
                                        x_train_prepped, y_train_prepped)
            G .= grad
            return val
        else
            val, _ = nlml_and_gradient(gp_model, p, kernel_constructor, 
                                    x_train_prepped, y_train_prepped)
            return val
        end
    end
    
    result = Optim.optimize(objective_with_gradient!,
                           log.([1e-3, 1e-6]), 
                           log.([1e1, 1e2]), 
                           old_params, 
                           Fminbox(LBFGS()),
                           Optim.Options(g_tol=1e-5))
    
    println("SUCCESS: Optimization with analytical gradient worked!")
    println("Optimized parameters: ", result.minimizer)
    
catch e
    println("FAILED: ", e)
    println("Stack trace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("\n=== Testing finite difference gradient computation ===")

try
    nlml_val, grad = nlml_with_finite_diff_gradient(gp_model, old_params, kernel_constructor, 
                                                   x_train_prepped, y_train_prepped)
    
    println("SUCCESS: Finite difference gradient computed!")
    println("NLML value: ", nlml_val)
    println("Gradient: ", grad)
    
    # Test optimization with finite difference gradient
    function objective_with_fd_gradient!(F, G, p)
        if G !== nothing
            val, grad = nlml_with_finite_diff_gradient(gp_model, p, kernel_constructor, 
                                                      x_train_prepped, y_train_prepped)
            G .= grad
            return val
        else
            val, _ = nlml_with_finite_diff_gradient(gp_model, p, kernel_constructor, 
                                                  x_train_prepped, y_train_prepped)
            return val
        end
    end
    
    result = Optim.optimize(objective_with_fd_gradient!,
                           log.([1e-3, 1e-6]), 
                           log.([1e1, 1e2]), 
                           old_params, 
                           Fminbox(LBFGS()),
                           Optim.Options(g_tol=1e-5))
    
    println("SUCCESS: Optimization with finite difference gradient worked!")
    println("Optimized parameters: ", result.minimizer)
    
catch e
    println("FAILED: ", e)
    println("Stack trace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("\n=== Comparing gradient computation methods ===")

try
    # Compare ForwardDiff vs our methods
    obj = p -> nlml(gp_model, p, kernel_constructor, x_train_prepped, y_train_prepped)
    
    println("Original NLML value: ", obj(old_params))
    
    # Try ForwardDiff (might fail)
    try
        fd_grad = ForwardDiff.gradient(obj, old_params)
        println("ForwardDiff gradient: ", fd_grad)
    catch e
        println("ForwardDiff failed (expected): ", e)
    end
    
    # Our analytical gradient
    try
        _, analytical_grad = nlml_and_gradient(gp_model, old_params, kernel_constructor, 
                                             x_train_prepped, y_train_prepped)
        println("Analytical gradient: ", analytical_grad)
    catch e
        println("Analytical gradient failed: ", e)
    end
    
    # Finite difference gradient
    try
        _, fd_grad = nlml_with_finite_diff_gradient(gp_model, old_params, kernel_constructor, 
                                                   x_train_prepped, y_train_prepped)
        println("Finite difference gradient: ", fd_grad)
    catch e
        println("Finite difference gradient failed: ", e)
    end
    
catch e
    println("Comparison failed: ", e)
end

println("\nTest completed.")
