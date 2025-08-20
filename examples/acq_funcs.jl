# filepath: /home/vandiere/.julia/dev/BayesOpt/examples/acq_funcs.jl
"""
Comparison of acquisition functions for 1D Bayesian Optimization using Gradient GP.

This example compares:
1. Expected Improvement (EI)
2. Probability of Improvement (PI) 
4. Ensemble of UCB and GradientNormUCB
5. Ensemble of EI and GradientNormUCB

All using GradientGP with gradient information.
"""

using AbstractGPs
using KernelFunctions
using Plots
using Distributions
using ForwardDiff
using BayesOpt
using LinearAlgebra
using LaTeXStrings
import Random
using Optim
Random.seed!(555)

# Objective Function - same as 1D example
f(x) = sin(sum(x.+1)) + sin((10.0 / 3.0) * sum(x .+1))
∇f(x) = ForwardDiff.gradient(f, x)
min_f = −1.988699758534924
f_val_grad(x) = [f(x); ∇f(x)]

# Problem setup
d = 1
lower = [-10.0]
upper = [10.0]
domain = ContinuousDomain(lower, upper)

σ² = 1e-12

# Generate initial training data
n_train = 5
x_train = [lower .+ (upper .- lower) .* rand(d) for _ in 1:n_train]
val_grad = f_val_grad.(x_train)
y_train = [val_grad[i] for i = eachindex(val_grad)]

# Setup kernel and model
kernel_constructor = ApproxMatern52Kernel()
kernel = 1 * (kernel_constructor ∘ ScaleTransform(1))
grad_kernel = gradKernel(kernel)
model = GradientGP(grad_kernel, d+1, σ²)

# Define acquisition functions to compare
function setup_acquisition_functions(y_train)
    best_y = minimum(hcat(y_train...)[1,:])
    
    # 1. Expected Improvement
    ei_acq = ExpectedImprovement(0.0, best_y)
    
    # 2. Probability of Improvement  
    pi_acq = ProbabilityImprovement(0.0, best_y)
   
    # 3. Upper-Confidence-Bound
    ucb_acq = UpperConfidenceBound(2.0)

    # 4. Tracked Ensemble of UCB and GradientNormUCB
    ucb_acq_for_ensemble = UpperConfidenceBound(2.0)
    grad_ucb_acq = GradientNormUCB(2.0)
    ensemble_ucb_grad = EnsembleAcquisition([0.7, 0.3], [ucb_acq_for_ensemble, grad_ucb_acq])
    
    # 5. Tracked Ensemble of EI and GradientNormUCB
    ei_for_ensemble = ExpectedImprovement(0.0, best_y)
    grad_ucb_for_ensemble = GradientNormUCB(1.5)
    ensemble_ei_grad = EnsembleAcquisition([0.7, 0.3], [ei_for_ensemble, grad_ucb_for_ensemble])
    
    return [
        ("EI", ei_acq),
        ("PI", pi_acq),
        ("UCB", ucb_acq), 
        ("UCB+GradUCB", ensemble_ucb_grad),
        ("EI+GradUCB", ensemble_ei_grad)
    ]
end

# Run optimization for each acquisition function
function run_comparison(n_iterations=30)
    results = Dict()
    
    for (name, acq_func) in setup_acquisition_functions(y_train)
        println("\n=== Running optimization with $name ===")
        
        # Create problem for this acquisition function
        problem = BOProblem(
            f_val_grad,
            domain,
            model,
            kernel_constructor,
            copy(x_train),
            copy(y_train),
            acq_func,
            n_iterations,
            0.0
        )
        
        # Run optimization
        result, acqf_list, standard_params = BayesOpt.optimize(problem, hyper_params="all")
        
        # Extract results - handle early termination
        xs = reduce(vcat, result.xs)
        ys = result.ys_non_std
        ys_values = hcat(ys...)[1,:]
        
        # Handle case where optimization terminated early
        n_actual = length(ys_values)
        if n_actual > 0
            optimal_point = xs[argmin(ys_values)]
            optimal_value = minimum(ys_values)
        else
            # Fallback if no evaluations were made
            optimal_point = x_train[1]
            optimal_value = f(x_train[1])
        end
        
        println("Optimal point: $optimal_point")
        println("Optimal value: $optimal_value")
        println("Error from true minimum: $(abs(optimal_value - min_f))")
        
        # Compute running minimum
        running_min = accumulate(min, f.(xs))
        
        # Extract ensemble tracking data if available
        component_data = nothing
        if isa(acqf_list[end], TrackedEnsemble)
            component_data = (
                values = acqf_list[end].component_values,
                names = acqf_list[end].component_names,
                weights = acqf_list[end].ensemble.weights
            )
        end
        
        results[name] = (
            xs = xs,
            ys = ys_values,
            running_min = running_min,
            optimal_point = optimal_point,
            optimal_value = optimal_value,
            error = abs(optimal_value - min_f),
            component_data = component_data
        )
    end
    
    return results
end

# Run the comparison
println("Starting acquisition function comparison...")
results = run_comparison(50)  # Reduced iterations to avoid numerical issues

# Plot convergence comparison
function plot_convergence(results)
    p = plot(title="Acquisition Function Comparison (1D GradBO)",
            xlabel="Function evaluations",
            ylabel=L"|| f(x^*_n) - f^* ||",
            yaxis=:log,
            legend=:topright,
            linewidth=2)
    
    colors = [:blue, :red, :green, :orange, :purple]
    
    for (i, (name, result)) in enumerate(results)
        # Extend running minimum to show same pattern as original
        running_min_extended = collect(Iterators.flatten(fill(x, 2) for x in result.running_min))
        errors = max.(running_min_extended .- min_f, 1e-16)  # Avoid log(0)
        
        plot!(p, (2*n_train):length(errors), errors[(2*n_train):end],
              label=name,
              color=colors[i],
              alpha=0.8)
    end
    
    # Add initial training data region
    vspan!(p, [1, 2*n_train]; color=:gray, alpha=0.2, label="Initial data")
    
    return p
end

# Create and display convergence plot
conv_plot = plot_convergence(results)
display(conv_plot)

# Print summary statistics
println("\n=== SUMMARY ===")
println("Method\t\t\tFinal Error\tOptimal Point\t\tOptimal Value")
println("="^80)
for (name, result) in results
    println("$(rpad(name, 15))\t$(round(result.error, digits=6))\t$(round(result.optimal_point[1], digits=4))\t\t$(round(result.optimal_value, digits=6))")
end

# Plot final GP fit for best performing method
function plot_best_gp_fit(results)
    # Find best method (lowest error)
    best_result, best_name = findmin(results) do result
        result.error
    end
    
    println("\nBest performing method: $(best_name) with error $(round(best_result, digits=6))")
    
    # Create detailed plot of the best method's final GP
    plot_domain = collect(lower[1]:0.01:upper[1])
    
    # For visualization, we'd need to access the final GP state
    # This would require modifying the optimization loop to return the final model
    # For now, just plot the function and found points
    
    p2 = plot(plot_domain, f.(plot_domain),
             label="True function",
             xlim=(lower[1], upper[1]),
             xlabel="x",
             ylabel="f(x)",
             title="Best Method: $(best_name[1]) - Final Result",
             linewidth=2,
             color=:black)
    
    # Plot initial training points
    scatter!(p2, [x[1] for x in x_train], [f(x) for x in x_train],
            label="Initial data", 
            color=:blue, 
            markersize=6)
    
    # Plot optimization trajectory
    scatter!(p2, [x[1] for x in results[best_name].xs[n_train+1:end]], 
            results[best_name].ys[n_train+1:end],
            label="BO samples",
            color=:red,
            markersize=4,
            alpha=0.7)
    
    # Highlight best point
    scatter!(p2, [results[best_name].optimal_point[1]], [results[best_name].optimal_value],
            label="Best found",
            color=:green,
            markersize=8,
            markershape=:star)
    
    return p2
end

best_plot = plot_best_gp_fit(results)
display(best_plot)

println("\nComparison complete!")