"""
Test file for scale invariance in 2D Bayesian Optimization with GradientGPs

This test compares the behavior of BO when optimizing f(x) vs optimizing f(x)/σ̄
where σ̄ is the standard deviation of the training outputs, using gradient-enhanced GPs.

Theoretically, BO should be scale invariant - optimizing f or f/σ̄ should yield
the same sequence of candidate points xs and proportionally scaled acquisition values.
"""

using AbstractGPs
using KernelFunctions
using Plots
using Distributions
using LaTeXStrings
using LinearAlgebra
using AbstractBayesOpt
using Test
using ForwardDiff

import Random
Random.seed!(42)  # Fixed seed for reproducibility

# Himmelblau function and its gradient
himmelblau(x::AbstractVector) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
∇himmelblau(x::AbstractVector) = ForwardDiff.gradient(himmelblau, x)
himmelblau_val_grad(x::AbstractVector) = [himmelblau(x); ∇himmelblau(x)]

# Problem setup
problem_dim = 2
lower = [-6.0, -6.0]
upper = [6.0, 6.0]
domain = ContinuousDomain(lower, upper)
σ² = 1e-6
n_train = 10
n_iterations =15

# Generate the same initial training data for both tests
Random.seed!(42)
x_train = [lower .+ (upper .- lower) .* rand(problem_dim) for _ in 1:n_train]
y_train_original = himmelblau_val_grad.(x_train)

# Calculate scaling factor (standard deviation of function values only)
function_values = [y[1] for y in y_train_original]
scaling_factor = std(function_values)
@info "Scaling factor (σ̄): $scaling_factor"


# Kernel and model setup
kernel_constructor = ApproxMatern52Kernel()
kernel = 1 * (kernel_constructor ∘ ScaleTransform(1))
grad_kernel = gradKernel(kernel)

# Test 1: Original function with no standardization
@info "Running BO on original function (no standardization)..."
Random.seed!(42)

model_orig = GradientGP(grad_kernel, problem_dim + 1, σ²)
acqf_orig = ExpectedImprovement(0.0, minimum([y[1] for y in y_train_original]))

bo_struct_orig = BOStruct(
    himmelblau_val_grad,
    acqf_orig,
    model_orig,
    domain,
    copy(x_train),
    copy(y_train_original),
    n_iterations,
    0.0
)

result_orig, acq_list_orig, std_params_orig = AbstractBayesOpt.optimize(
    bo_struct_orig, 
    standardize=nothing, 
    hyper_params=nothing
)

# Test 2: Original function with scale_only standardization
@info "Running BO on original function (scale_only standardization)..."
Random.seed!(42)

model_orig_std = GradientGP(grad_kernel, problem_dim + 1, σ²)
acqf_orig_std = ExpectedImprovement(0.0, minimum([y[1] for y in y_train_original]))

bo_struct_orig_std = BOStruct(
    himmelblau_val_grad,
    acqf_orig_std,
    model_orig_std,
    domain,
    copy(x_train),
    copy(y_train_original),
    n_iterations,
    0.0
)

result_orig_std, acq_list_orig_std, std_params_orig_std = AbstractBayesOpt.optimize(
    bo_struct_orig_std, 
    standardize="scale_only", 
    hyper_params=nothing
)

xs_orig = result_orig.xs
xs_orig_std = result_orig_std.xs


# Plot 1: Function evaluations over iterations
p1 = plot(title="Function Values at Sampled Points (should match)", xlabel="Sample point index",
            ylabel="f(x)", yaxis=:log, legend=:bottomleft)
plot!(p1, (n_train+1):length(xs_orig), himmelblau.(xs_orig)[(n_train+1):end], 
      label="Original (no std)", marker=:circle, linewidth=2)
plot!(p1, (n_train+1):length(xs_orig_std), himmelblau.(xs_orig_std)[(n_train+1):end], 
      label="Original (scale_only std)", marker=:diamond, linewidth=2, linestyle=:dot)

# Plot 2: Acquisition values over iterations
p2 = plot(title="Acquisition Values (should match)", xlabel="Iteration", ylabel="Acquisition Value", yaxis=:log)
plot!(p2, (n_train+1):(length(acq_list_orig)+n_train), acq_list_orig .+ eps(), 
      label="Original (no std)", marker=:circle, linewidth=2)
plot!(p2, (n_train+1):(length(acq_list_orig_std)+n_train), acq_list_orig_std .* scaling_factor .+ eps(), 
      label="Original (scale_only std) (rescaled)", marker=:diamond, linewidth=2, linestyle=:dot)

# Plot 3: Running minimum
running_min_orig = accumulate(min, himmelblau.(xs_orig))
running_min_orig_std = accumulate(min, himmelblau.(xs_orig_std))

p3 = plot(title="Running Minimum", xlabel="Function Evaluations", ylabel="Best f(x) Found", yaxis=:log)
plot!(p3, (n_train+1):length(running_min_orig), running_min_orig[n_train+1:end], 
      label="Original (no std)", linewidth=2)
plot!(p3, (n_train+1):length(running_min_orig_std), running_min_orig_std[n_train+1:end], 
      label="Original (scale_only std)", linewidth=2, linestyle=:dot)

# Combine plots
p_combined = plot(p1, p2, p3, layout=(3,1), size=(800, 900))
display(p_combined)