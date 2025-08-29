using AbstractGPs
using KernelFunctions
using Plots
using Distributions
using ForwardDiff
using LinearAlgebra
using LaTeXStrings
using QuasiMonteCarlo
import Random
using Optim
using AbstractBayesOpt
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

x_train = [collect(col) for col in eachcol(QuasiMonteCarlo.sample(n_train, lower, upper, SobolSample()))]

# x_train = [lower .+ (upper .- lower) .* rand(d) for _ in 1:n_train]
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
    ucb_acq = UpperConfidenceBound(1.96)

    # 4. Tracked Ensemble of UCB and GradientNormUCB
    ucb_acq_for_ensemble = UpperConfidenceBound(1.96)
    grad_ucb_acq = GradientNormUCB(1.5)
    ensemble_ucb_grad = EnsembleAcquisition([0.9, 0.1], [ucb_acq_for_ensemble, grad_ucb_acq])
    
    # 5. Tracked Ensemble of EI and GradientNormUCB
    ei_for_ensemble = ExpectedImprovement(0.0, best_y)
    grad_ucb_for_ensemble = GradientNormUCB(1.5)
    ensemble_ei_grad = EnsembleAcquisition([0.9, 0.1], [ei_for_ensemble, grad_ucb_for_ensemble])
    
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
        
        # Run optimization
        result, acqf_list, standard_params = AbstractBayesOpt.optimize(problem, hyper_params="all")
        
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
results = run_comparison(30)  # Reduced iterations to avoid numerical issues

# Plot convergence comparison
function plot_convergence(results)
    p = plot(title="Acquisition Function Comparison (1D GradBO)",
            xlabel="Function evaluations",
            ylabel=L"|| f(x^*_n) - f^* ||",
            yaxis=:log,
            legend=:bottomleft,
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