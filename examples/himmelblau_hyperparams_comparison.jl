"""
Himmelblau Function Optimization with Gradient-Enhanced GP:
Comparison of Hyperparameter Tuning Methods and Standardization

This example demonstrates gradient-enhanced Bayesian Optimization on the Himmelblau function
with different configurations:
1. Different hyperparameter optimization strategies ("all", "length_scale_only", "none")
2. With and without standardization
3. Performance comparison and convergence analysis

"""

using AbstractGPs, KernelFunctions
using Plots
using Distributions
using ForwardDiff
using AbstractBayesOpt
using LinearAlgebra
using LaTeXStrings
using QuasiMonteCarlo
import Random
using Optim
using Statistics

Random.seed!(42)

# Himmelblau function and its gradient
himmelblau(x::AbstractVector) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
∇himmelblau(x) = ForwardDiff.gradient(himmelblau, x)

# Combined function that returns both value and gradient
f_val_grad(x) = [himmelblau(x); ∇himmelblau(x)]

# Known global minimum value
global_min = 0.0

# Problem setup
d = 2
lower = [-6.0, -6.0]
upper = [6.0, 6.0]
domain = ContinuousDomain(lower, upper)

σ² = 1e-12

# Generate initial training data using Sobol sampling for better coverage
n_train = 8
x_train = [collect(col) for col in eachcol(QuasiMonteCarlo.sample(n_train, lower, upper, SobolSample()))]

# Evaluate function and gradients at training points
val_grad = f_val_grad.(x_train)
y_train = [val_grad[i] for i = eachindex(val_grad)]

# Setup kernel and base model
kernel_constructor = ApproxMatern52Kernel()
kernel = 1 * (kernel_constructor ∘ ScaleTransform(1))
grad_kernel = gradKernel(kernel)
base_model = GradientGP(grad_kernel, d+1, σ²)

# Define test configurations
test_configs = [
    # (name, hyper_params, standardize)
    ("HP:all + Std", "all", true),
    ("HP:all + NoStd", "all", false), 
    ("HP:length + Std", "length_scale_only", true),
    ("HP:length + NoStd", "length_scale_only", false),
    ("HP:none + Std", nothing, true),
    ("HP:none + NoStd", nothing, false)
]

# Run optimization comparison
function run_himmelblau_comparison(n_iterations=40)
    results = Dict()
    
    for (config_name, hyper_params, standardize) in test_configs
        
        model = deepcopy(base_model)
        
        best_y = minimum(hcat(y_train...)[1,:])
        acq_func = ExpectedImprovement(0.0, best_y)
        
        problem = BOStruct(
            f_val_grad,
            acq_func,
            model,
            kernel_constructor,
            domain,
            copy(x_train),
            copy(y_train),
            n_iterations,
            0.0
        )
        
        start_time = time()
        
        # Run optimization with specified configuration
        try
            result, _, standard_params = AbstractBayesOpt.optimize(
                problem, 
                hyper_params=hyper_params,
                standardize=standardize
            )
            
            # Record end time
            end_time = time()
            elapsed_time = end_time - start_time
            
            # Extract results
            xs = result.xs
            ys_non_std = result.ys_non_std
            ys_values = hcat(ys_non_std...)[1,:]
            
            # Find optimal solution
            n_actual = length(ys_values)
            if n_actual > 0
                optimal_idx = argmin(ys_values)
                optimal_point = xs[optimal_idx]
                optimal_value = minimum(ys_values)
            else
                optimal_point = x_train[1]
                optimal_value = himmelblau(x_train[1])
            end
            
            # Compute running minimum for convergence analysis
            all_evals = himmelblau.(xs)
            running_min = accumulate(min, all_evals)
            
            # Compute errors from global minimum
            errors = max.(running_min .- global_min, 1e-16)
            
            # Store results
            results[config_name] = (
                xs = xs,
                ys_values = ys_values,
                running_min = running_min,
                errors = errors,
                optimal_point = optimal_point,
                optimal_value = optimal_value,
                error_from_global = abs(optimal_value - global_min),
                elapsed_time = elapsed_time,
                hyper_params = hyper_params,
                standardize = standardize,
                standard_params = standard_params,
                n_evaluations = length(xs)
            )
            
        catch e
            println("ERROR in configuration $config_name: $e")
            # Store minimal error result
            results[config_name] = (
                error_from_global = Inf,
                elapsed_time = Inf,
                hyper_params = hyper_params,
                standardize = standardize,
                n_evaluations = 0
            )
        end
    end
    
    return results
end

# Execute the comparison
println("Starting Himmelblau function optimization comparison...")
results = run_himmelblau_comparison(40)

# Analysis and visualization functions
function plot_convergence_comparison(results)
    p = plot(title="Himmelblau Optimization: Hyperparameter & Standardization Comparison",
            xlabel="Number of iterations",
            ylabel="Error from global minimum",
            yaxis=:log,
            legend=:topright,
            linewidth=2,
            size=(1000, 600))
    
    # Define colors and line styles for different configurations
    colors = [:blue, :lightblue, :red, :pink, :green, :lightgreen]
    styles = [:solid, :solid, :solid, :solid, :solid, :solid]
    
    config_names = [
        "HP:all + Std", "HP:all + NoStd",
        "HP:length + Std", "HP:length + NoStd", 
        "HP:none + Std", "HP:none + NoStd"
    ]
    
    for (i, config_name) in enumerate(config_names)
        if haskey(results, config_name) && haskey(results[config_name], :errors)
            result = results[config_name]
            plot!(p, 1:length(result.errors), result.errors,
                  label=config_name,
                  color=colors[i],
                  linestyle=styles[i])
        end
    end
    
    # Add reference lines
    hline!(p, [1e-8], linestyle=:dot, color=:gray, alpha=0.5, label="1e-8 tolerance")
    
    # Add initial training data region
    vspan!(p, [1, n_train]; color=:gray, alpha=0.2, label="Initial data")
    
    return p
end

# Create visualizations and analysis
conv_plot = plot_convergence_comparison(results)

