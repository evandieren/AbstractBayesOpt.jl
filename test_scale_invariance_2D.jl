"""
Test file for scale invariance in 2D Bayesian Optimization

This test compares the behavior of BO when optimizing f(x) vs optimizing f(x)/σ̄
where σ̄ is the standard deviation of the training outputs.

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

import Random
Random.seed!(42)  # Fixed seed for reproducibility

# Himmelblau function
himmelblau(x::AbstractVector) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

# Problem setup
problem_dim = 2
lower = [-6.0, -6.0]
upper = [6.0, 6.0]
domain = ContinuousDomain(lower, upper)
σ² = 1e-6
n_train = 10
n_iterations = 45

# Generate the same initial training data for both tests
Random.seed!(42)
x_train = [lower .+ (upper .- lower) .* rand(problem_dim) for _ in 1:n_train]
y_train_original = himmelblau.(x_train)

# Calculate scaling factor (standard deviation of training outputs)
scaling_factor = std(y_train_original)
@info "Scaling factor (σ̄): $scaling_factor"

# Scaled function
himmelblau_scaled(x) = himmelblau(x) / scaling_factor

# Prepare training data for both cases
y_train_original_vec = map(x -> [x], y_train_original)
y_train_scaled = himmelblau_scaled.(x_train)
y_train_scaled_vec = map(x -> [x], y_train_scaled)

# Kernel and model setup
kernel_constructor = ApproxMatern52Kernel()
kernel = 1 * (kernel_constructor ∘ ScaleTransform(1))

# Test 1: Original function with no standardization
@info "Running BO on original function (no standardization)..."
Random.seed!(42)

model_orig = StandardGP(kernel, σ²)
acqf_orig = ExpectedImprovement(0.0, minimum(y_train_original)[1])

bo_struct_orig = BOStruct(
    himmelblau,
    acqf_orig,
    model_orig,
    domain,
    copy(x_train),
    copy(y_train_original_vec),
    n_iterations,
    0.0
)

result_orig, acq_list_orig, std_params_orig = AbstractBayesOpt.optimize(
    bo_struct_orig, 
    standardize=nothing, 
    # hyper_params=nothing
)

# Test 2: Scaled function with no standardization
@info "Running BO on scaled function (no standardization)..."
Random.seed!(42)

kernel_scaled = 1/(scaling_factor^2) * (kernel_constructor ∘ ScaleTransform(1))

model_scaled = StandardGP(kernel_scaled, σ²/(scaling_factor^2))
acqf_scaled = ExpectedImprovement(0.0, minimum(y_train_scaled)[1])

bo_struct_scaled = BOStruct(
    himmelblau_scaled,
    acqf_scaled,
    model_scaled,
    domain,
    copy(x_train),
    copy(y_train_scaled_vec),
    n_iterations,
    0.0
)

result_scaled, acq_list_scaled, std_params_scaled = AbstractBayesOpt.optimize(
    bo_struct_scaled, 
    standardize=nothing, 
    # hyper_params=nothing
)

# Test 3: Original function with scale_only standardization
@info "Running BO on original function (scale_only standardization)..."
Random.seed!(42)

model_orig_std = StandardGP(kernel, σ²)
acqf_orig_std = ExpectedImprovement(0.0, minimum(y_train_original)[1])

bo_struct_orig_std = BOStruct(
    himmelblau,
    acqf_orig_std,
    model_orig_std,
    domain,
    copy(x_train),
    copy(y_train_original_vec),
    n_iterations,
    0.0
)

result_orig_std, acq_list_orig_std, std_params_orig_std = AbstractBayesOpt.optimize(
    bo_struct_orig_std, 
    standardize="scale_only", 
    # hyper_params=nothing
)

# Analysis and comparison
println("\n" * "="^60)
println("RESULTS COMPARISON")
println("="^60)

xs_orig = result_orig.xs
xs_scaled = result_scaled.xs
xs_orig_std = result_orig_std.xs


# Plot 1: Function evaluations over iterations
p1 = plot(title="Function Values at Sampled Points (should match)", xlabel="Sample point index", ylabel="f(x)")
plot!(p1, (n_train+1):length(xs_orig), himmelblau.(xs_orig)[(n_train+1):end], 
      label="Original (no std)", marker=:circle, linewidth=2)
plot!(p1, (n_train+1):length(xs_scaled), himmelblau.(xs_scaled)[(n_train+1):end], 
      label="Scaled function", marker=:square, linewidth=2, linestyle=:dash)
plot!(p1, (n_train+1):length(xs_orig_std), himmelblau.(xs_orig_std)[(n_train+1):end], 
      label="Original (scale_only std)", marker=:diamond, linewidth=2, linestyle=:dot)

# Plot 2: Acquisition values over iterations
p2 = plot(title="Acquisition Values (should match)", xlabel="Iteration", ylabel="Acquisition Value", yaxis=:log)
plot!(p2, (n_train+1):length(acq_list_orig), acq_list_orig[(n_train+1):end] .+ eps(), 
      label="Original (no std)", marker=:circle, linewidth=2)
plot!(p2, (n_train+1):length(acq_list_scaled), acq_list_scaled[(n_train+1):end] .* scaling_factor .+ eps(), 
      label="Scaled function (rescaled)", marker=:square, linewidth=2, linestyle=:dash)
plot!(p2, (n_train+1):length(acq_list_orig_std), acq_list_orig_std[(n_train+1):end] .* scaling_factor .+ eps(), 
      label="Original (scale_only std) (rescaled)", marker=:diamond, linewidth=2, linestyle=:dot)

# Plot 3: Running minimum
running_min_orig = accumulate(min, himmelblau.(xs_orig))
running_min_scaled = accumulate(min, himmelblau.(xs_scaled))
running_min_orig_std = accumulate(min, himmelblau.(xs_orig_std))

p3 = plot(title="Running Minimum", xlabel="Function Evaluations", ylabel="Best f(x) Found", yaxis=:log)
plot!(p3, 1:length(running_min_orig), running_min_orig, 
      label="Original (no std)", linewidth=2)
plot!(p3, 1:length(running_min_scaled), running_min_scaled, 
      label="Scaled function", linewidth=2, linestyle=:dash)
plot!(p3, 1:length(running_min_orig_std), running_min_orig_std, 
      label="Original (scale_only std)", linewidth=2, linestyle=:dot)
vspan!(p3, [1, n_train], color=:blue, alpha=0.2, label="Initial training")

# Combine plots
p_combined = plot(p1, p2, p3, layout=(3,1), size=(800, 900))
display(p_combined)


# For me, this is fine.