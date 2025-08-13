"""
Test file 1: Attempting to make autodiff = :forward work with GradientGP

The issue is that GradientGP already uses ForwardDiff internally in gradKernel,
and when we use autodiff = :forward in the optimizer, we get nested dual numbers.

This file explores different approaches to make it work:
1. Using a flag/config to disable internal ForwardDiff when optimizing
2. Using ReverseDiff instead of ForwardDiff for the optimizer
3. Using finite differences for the optimizer
"""

using AbstractGPs
using KernelFunctions
using ForwardDiff
using ReverseDiff
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

println("Testing autodiff workarounds for GradientGP...")
println("Training data:")
println("x_train = ", x_train)
println("y_train = ", y_train)

# Setup GradientGP
kernel_constructor = ApproxMatern52Kernel()
kernel = 1.0 * (kernel_constructor ∘ ScaleTransform(1.0))
grad_kernel = gradKernel(kernel)
gp_model = GradientGP(grad_kernel, d+1, 1e-12)

# Prepare data for nlml
x_train_prepped = KernelFunctions.MOInputIsotopicByOutputs(x_train, length(y_train[1]))
y_train_prepped = vec(permutedims(reduce(hcat, y_train)))

println("\nPrepared data shapes:")
println("x_train_prepped: ", typeof(x_train_prepped))
println("y_train_prepped length: ", length(y_train_prepped))

# Test parameters
old_params = [log(1.0), log(1.0)]  # [log(lengthscale), log(scale)]

println("\n=== Approach 1: Try autodiff = :forward directly ===")
try
    obj = p -> nlml(gp_model, p, kernel_constructor, x_train_prepped, y_train_prepped)
    
    println("NLML value at initial params: ", obj(old_params))
    
    # This should fail with dual number conflicts
    result = Optim.optimize(obj, 
                           log.([1e-3, 1e-6]), 
                           log.([1e1, 1e2]), 
                           old_params, 
                           Fminbox(LBFGS()),
                           Optim.Options(g_tol=1e-5),
                           autodiff = :forward)
    
    println("SUCCESS: autodiff = :forward worked!")
    println("Optimized parameters: ", result.minimizer)
    
catch e
    println("FAILED as expected: ", e)
end

println("\n=== Approach 2: Use ReverseDiff instead ===")
try
    obj = p -> nlml(gp_model, p, kernel_constructor, x_train_prepped, y_train_prepped)
    
    # Use ReverseDiff for gradient computation
    # grad_obj! = (G, p) -> ReverseDiff.gradient!(G, obj, p)
    
    result = Optim.optimize(obj, #grad_obj!,
                           log.([1e-3, 1e-6]), 
                           log.([1e1, 1e2]), 
                           old_params, 
                           Fminbox(LBFGS()),
                           Optim.Options(g_tol=1e-5),
                           autodiff = :reverse)
    
    println("SUCCESS: ReverseDiff worked!")
    println("Optimized parameters: ", result.minimizer)
    
catch e
    println("FAILED: ", e)
end

println("\n=== Approach 3: Use finite differences ===")
try
    obj = p -> nlml(gp_model, p, kernel_constructor, x_train_prepped, y_train_prepped)
    
    result = Optim.optimize(obj, 
                           log.([1e-3, 1e-6]), 
                           log.([1e1, 1e2]), 
                           old_params, 
                           Fminbox(LBFGS()),
                           Optim.Options(g_tol=1e-5))  # No autodiff specified, uses finite differences
    
    println("SUCCESS: Finite differences worked!")
    println("Optimized parameters: ", result.minimizer)
    
catch e
    println("FAILED: ", e)
end

println("\n=== Approach 4: Manual gradient computation ===")
try
    obj = p -> nlml(gp_model, p, kernel_constructor, x_train_prepped, y_train_prepped)
    
    # Compute gradient manually using finite differences
    function manual_grad!(G, p)
        h = 1e-8
        for i in 1:length(p)
            p_plus = copy(p)
            p_minus = copy(p)
            p_plus[i] += h
            p_minus[i] -= h
            G[i] = (obj(p_plus) - obj(p_minus)) / (2*h)
        end
    end
    
    result = Optim.optimize(obj, manual_grad!,
                           log.([1e-3, 1e-6]), 
                           log.([1e1, 1e2]), 
                           old_params, 
                           Fminbox(LBFGS()),
                           Optim.Options(g_tol=1e-5))
    
    println("SUCCESS: Manual finite differences worked!")
    println("Optimized parameters: ", result.minimizer)
    
catch e
    println("FAILED: ", e)
end

println("\n=== Approach 5: Try to isolate the dual number issue ===")
# Let's see if we can identify exactly where the conflict occurs
try
    # Test if ForwardDiff can differentiate through nlml
    obj = p -> nlml(gp_model, p, kernel_constructor, x_train_prepped, y_train_prepped)
    
    println("Testing ForwardDiff.gradient on nlml directly...")
    grad = ForwardDiff.gradient(obj, old_params)
    println("SUCCESS: ForwardDiff.gradient worked directly!")
    println("Gradient: ", grad)
    
    # Now try with custom gradient in optimizer
    grad_obj! = (G, p) -> G .= ForwardDiff.gradient(obj, p)
    
    result = Optim.optimize(obj, grad_obj!,
                           log.([1e-3, 1e-6]), 
                           log.([1e1, 1e2]), 
                           old_params, 
                           Fminbox(LBFGS()),
                           Optim.Options(g_tol=1e-5))
    
    println("SUCCESS: Custom ForwardDiff gradient worked!")
    println("Optimized parameters: ", result.minimizer)
    
catch e
    println("FAILED: ", e)
    println("This suggests the issue is indeed with nested dual numbers")
end

println("\nTest completed. Check which approaches worked.")
