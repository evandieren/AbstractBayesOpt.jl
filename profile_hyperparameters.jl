"""
Profiling script to identify bottlenecks in hyperparameter tuning for gradient-enhanced GPs.

This script creates comparable optimization problems for StandardGP and GradientGP 
and profiles the hyperparameter optimization to identify performance differences.
"""

using AbstractGPs
using KernelFunctions
using ForwardDiff
using BayesOpt
using BenchmarkTools
using Random
using ReverseDiff

Random.seed!(42)

# Test function and its gradient
f(x) = sin(sum(x .+ 1)) + sin((10.0 / 3.0) * sum(x .+ 1))
∇f(x) = ForwardDiff.gradient(f, x)
f_val_grad(x) = [f(x); ∇f(x)]

# Problem setup
d = 2  # 2D problem
lower = [-10.0, -10.0]
upper = [10.0, 10.0]
domain = ContinuousDomain(lower, upper)
σ² = 1e-12

# Generate training data
n_train = 15
x_train = [lower .+ (upper .- lower) .* rand(d) for _ in 1:n_train]

# Standard GP data
y_train_standard = f.(x_train)
y_train_standard = map(x -> [x], y_train_standard)

# Gradient GP data
y_train_gradient = f_val_grad.(x_train)

println("=== Profiling Hyperparameter Optimization ===")
println("Problem dimension: $d")
println("Number of training points: $n_train")
println("Standard GP output dimension: 1")
println("Gradient GP output dimension: $(d+1)")
println()

# Kernel constructor
kernel_constructor = ApproxMatern52Kernel()

# ============================================================================
# Standard GP Setup
# ============================================================================
println("Setting up Standard GP...")
kernel_standard = 1 * (kernel_constructor ∘ ScaleTransform(1))
gp_standard = StandardGP(kernel_standard, σ²)
gp_standard = update!(gp_standard, x_train, y_train_standard)

# Initial hyperparameters (log scale)
old_params = [log(1.0), log(1.0)]  # [log(lengthscale), log(scale)]

# ============================================================================
# Gradient GP Setup  
# ============================================================================
println("Setting up Gradient GP...")
kernel_gradient = 1 * (kernel_constructor ∘ ScaleTransform(1))
grad_kernel = gradKernel(kernel_gradient)
gp_gradient = GradientGP(grad_kernel, d+1, σ²)
gp_gradient = update!(gp_gradient, x_train, y_train_gradient)

# ============================================================================
# Benchmark Standard GP Hyperparameter Optimization
# ============================================================================
println("\n=== Benchmarking Standard GP Hyperparameter Optimization ===")

function benchmark_standard_hp()
    
    for i = 1:1e3

    optimize_hyperparameters(
        gp_standard, x_train, y_train_standard, kernel_constructor,
        old_params, true, length_scale_only=true, num_restarts=1
    )

    end
    
end

println("Warming up Standard GP...")
#benchmark_standard_hp()


y_train_nlml = reduce(vcat,y_train_standard)
using BenchmarkTools
@btime nlml($gp_standard, [0.0, 0.0], $kernel_constructor, $x_train, $y_train_nlml)

@benchmark nlml($gp_standard, [0.0, 0.0], $kernel_constructor, $x_train, $y_train_nlml)


@benchmark nlml_ls($gp_standard, 0.0, 0.0, $kernel_constructor, $x_train, $y_train_nlml)



function bench_nlml()
    old_params = [0.0, 0.0]
    for i = 1:1e6
        nlml(gp_standard, old_params, kernel_constructor, x_train, y_train_nlml)
    end
end

function bench_nlml_ls()
    old_params = [0.0, 0.0]
    for i = 1:1e6
        nlml_ls(gp_standard, old_params[1], old_params[2], kernel_constructor, x_train, y_train_nlml)
    end
end

@btime bench_nlml()

@btime bench_nlml_ls()

println("Benchmarking nlml...")
# println("Benchmarking Standard GP (5 runs)...")
# standard_benchmark = @benchmark benchmark_standard_hp() samples=5 seconds=60

# println("Standard GP Results:")
# println("  Median time: $(median(standard_benchmark.times) / 1e6) ms")
# println("  Mean time: $(mean(standard_benchmark.times) / 1e6) ms")
# println("  Min time: $(minimum(standard_benchmark.times) / 1e6) ms")
# println("  Max time: $(maximum(standard_benchmark.times) / 1e6) ms")

# # ============================================================================
# # Benchmark Gradient GP Hyperparameter Optimization
# # ============================================================================
# println("\n=== Benchmarking Gradient GP Hyperparameter Optimization ===")

# function benchmark_gradient_hp()
#     optimize_hyperparameters(
#         gp_gradient, x_train, y_train_gradient, kernel_constructor,
#         old_params, false, length_scale_only=false, num_restarts=1
#     )
# end

# println("Warming up Gradient GP...")
# benchmark_gradient_hp()

# println("Benchmarking Gradient GP (5 runs)...")
# gradient_benchmark = @benchmark benchmark_gradient_hp() samples=5 seconds=120

# println("Gradient GP Results:")
# println("  Median time: $(median(gradient_benchmark.times) / 1e6) ms")
# println("  Mean time: $(mean(gradient_benchmark.times) / 1e6) ms")
# println("  Min time: $(minimum(gradient_benchmark.times) / 1e6) ms")
# println("  Max time: $(maximum(gradient_benchmark.times) / 1e6) ms")

# # ============================================================================
# # Compare Results
# # ============================================================================
# println("\n=== Performance Comparison ===")
# standard_median = median(standard_benchmark.times) / 1e6
# gradient_median = median(gradient_benchmark.times) / 1e6
# speedup_ratio = gradient_median / standard_median

# println("Standard GP median time: $(standard_median) ms")
# println("Gradient GP median time: $(gradient_median) ms")
# println("Gradient GP is $(speedup_ratio)x slower than Standard GP")
# println("Expected scaling factor (dimension-based): $((d+1)^2) ≈ $(9.0)")

# # ============================================================================
# # Detailed Profiling of Components
# # ============================================================================
# println("\n=== Detailed Component Profiling ===")

# # Profile nlml function for both GP types
# println("\nProfiling NLML evaluation...")

# function profile_standard_nlml()
#     x_prepped = prep_input(gp_standard, x_train)
#     y_prepped = reduce(vcat, y_train_standard)
#     for i in 1:1
#         nlml(gp_standard, old_params, kernel_constructor, x_prepped, y_prepped)
#     end
# end

# function profile_gradient_nlml()
#     x_prepped = prep_input(gp_gradient, x_train)
#     y_prepped = vec(permutedims(reduce(hcat, y_train_gradient)))
#     for i in 1:1
#         nlml(gp_gradient, old_params, kernel_constructor, x_prepped, y_prepped)
#         #println("Gradient GP NLML output: $out")
#     end
# end

# function profile_fast_nlml()
#     x_prepped = prep_input(gp_gradient, x_train)
#     y_prepped = vec(permutedims(reduce(hcat, y_train_gradient)))
#     grad_cache = build_grad_cache(gp_gradient, kernel_constructor, x_prepped, y_prepped)
#     for i in 1:1
#         fast_nlml!(grad_cache, old_params)
#         # println("Fast NLML output: $out")
#     end
# end


# profile_standard_nlml()
# profile_gradient_nlml()
# profile_fast_nlml()

# # Benchmark NLML evaluations
# println("Standard GP NLML (50 evaluations):")
# standard_nlml_bench = @benchmark profile_standard_nlml() samples=10

# println("Gradient GP NLML (50 evaluations):")
# gradient_nlml_bench = @benchmark profile_gradient_nlml() samples=10

# println("Fast NLML (50 evaluations):")
# fast_nlml_bench = @benchmark profile_fast_nlml() samples=10

# standard_nlml_time = median(standard_nlml_bench.times) / 1e6 / 50
# gradient_nlml_time = median(gradient_nlml_bench.times) / 1e6 / 50
# fast_nlml_time = median(fast_nlml_bench.times) / 1e6 / 50

# println("Standard GP NLML per evaluation: $(standard_nlml_time) ms")
# println("Gradient GP NLML per evaluation: $(gradient_nlml_time) ms")
# println("Fast NLML per evaluation: $(fast_nlml_time) ms")
# println("Gradient GP NLML is $(gradient_nlml_time / standard_nlml_time)x slower")


# # ============================================================================
# # Memory Allocation Analysis
# # ============================================================================
# println("\n=== Memory Allocation Analysis ===")

# println("Standard GP NLML memory allocation:")
# x_prepped_std = prep_input(gp_standard, x_train)
# y_prepped_std = reduce(vcat, y_train_standard)
# @time nlml(gp_standard, old_params, kernel_constructor, x_prepped_std, y_prepped_std)

# println("\nGradient GP NLML memory allocation:")
# @time nlml(gp_gradient, old_params, kernel_constructor, x_prepped_grad, y_prepped_grad)

# # ============================================================================
# # Kernel Matrix Size Analysis
# # ============================================================================
# println("\n=== Kernel Matrix Analysis ===")

# println("Standard GP:")
# println("  Training data size: $(length(x_train)) points")
# println("  Kernel matrix size: $(length(x_train)) x $(length(x_train)) = $(length(x_train)^2) elements")

# println("Gradient GP:")
# n_outputs = d + 1
# total_grad_points = length(x_train) * n_outputs
# println("  Training data size: $(length(x_train)) points x $n_outputs outputs = $total_grad_points total points")
# println("  Kernel matrix size: $total_grad_points x $total_grad_points = $(total_grad_points^2) elements")
# println("  Matrix size ratio: $(total_grad_points^2 / length(x_train)^2) = $((n_outputs)^2)")
