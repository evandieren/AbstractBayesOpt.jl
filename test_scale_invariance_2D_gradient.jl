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

# Scaled function
himmelblau_scaled_val_grad(x) = begin
    val_grad = himmelblau_val_grad(x)
    return [val_grad[1] / scaling_factor; val_grad[2:end] ./ scaling_factor]
end

# Prepare training data for both cases
y_train_scaled = himmelblau_scaled_val_grad.(x_train)

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
    # hyper_params=nothing
)

# Test 2: Scaled function with no standardization
@info "Running BO on scaled function (no standardization)..."
Random.seed!(42)

kernel_scaled = 1/(scaling_factor^2) * (kernel_constructor ∘ ScaleTransform(1))
grad_kernel_scaled = gradKernel(kernel_scaled)

model_scaled = GradientGP(grad_kernel_scaled, problem_dim + 1, σ²/(scaling_factor^2))
acqf_scaled = ExpectedImprovement(0.0, minimum([y[1] for y in y_train_scaled]))

bo_struct_scaled = BOStruct(
    himmelblau_scaled_val_grad,
    acqf_scaled,
    model_scaled,
    domain,
    copy(x_train),
    copy(y_train_scaled),
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
    # hyper_params=nothing
)

xs_orig = result_orig.xs
xs_scaled = result_scaled.xs
xs_orig_std = result_orig_std.xs


# Plot 1: Function evaluations over iterations
p1 = plot(title="Function Values at Sampled Points (should match)", xlabel="Sample point index", ylabel="f(x)")
plot!(p1, (n_train+1):length(xs_orig), himmelblau.(xs_orig)[(n_train+1):end], 
      label="Original (no std)", marker=:circle, linewidth=2,yaxis=:log)
plot!(p1, (n_train+1):length(xs_scaled), himmelblau.(xs_scaled)[(n_train+1):end], 
      label="Scaled function", marker=:square, linewidth=2, linestyle=:dash)
plot!(p1, (n_train+1):length(xs_orig_std), himmelblau.(xs_orig_std)[(n_train+1):end], 
      label="Original (scale_only std)", marker=:diamond, linewidth=2, linestyle=:dot)

# Plot 2: Acquisition values over iterations
p2 = plot(title="Acquisition Values (should match)", xlabel="Iteration", ylabel="Acquisition Value", yaxis=:log)
plot!(p2, (n_train+1):length(acq_list_orig), acq_list_orig[(n_train+1):end] .+ eps(), 
      label="Original (no std)", marker=:circle, linewidth=2,legend=:bottomleft)
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

# Plot 4: Gradient norms comparison (unique to GradientGP)
gradient_norms_orig = [norm(himmelblau_val_grad(x)[2:end]) for x in xs_orig[(n_train+1):end]]
gradient_norms_scaled = [norm(himmelblau_val_grad(x)[2:end]) for x in xs_scaled[(n_train+1):end]]
gradient_norms_orig_std = [norm(himmelblau_val_grad(x)[2:end]) for x in xs_orig_std[(n_train+1):end]]

p4 = plot(title="Gradient Norms at Sampled Points", xlabel="Sample point index", ylabel="||∇f(x)||")
plot!(p4, (n_train+1):length(xs_orig), gradient_norms_orig, 
      label="Original (no std)", marker=:circle, linewidth=2,yaxis=:log)
plot!(p4, (n_train+1):length(xs_scaled), gradient_norms_scaled, 
      label="Scaled function", marker=:square, linewidth=2, linestyle=:dash)
plot!(p4, (n_train+1):length(xs_orig_std), gradient_norms_orig_std, 
      label="Original (scale_only std)", marker=:diamond, linewidth=2, linestyle=:dot)

# Combine plots
p_combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))
display(p_combined)
savefig(p_combined, "scale_invariance_2D_gradient_comparison.png")

cond(kernelmatrix(kernel, xs_orig, xs_orig) + 1e-6*I)
cond(kernelmatrix(kernel_scaled, xs_orig_std, xs_orig_std) + (1e-6/scaling_factor^2)*I)

# Some issues at the 15th, seems like we do not find the same point