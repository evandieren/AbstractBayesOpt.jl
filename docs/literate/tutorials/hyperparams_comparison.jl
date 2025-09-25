#=
# AbstractBayesOpt Tutorial: Hyperparameter Tuning and Standardisation Comparison
=#

# This tutorial presents the different options for hyperparameter optimisation and standardisation modes available in AbstractBayesOpt.jl.
# We will compare the performance of these configurations on gradient-enhanced GPs on the Himmelblau function

# ## Setup
#
# Loading the necessary packages.
using AbstractBayesOpt
using AbstractGPs
using Plots
using ForwardDiff
using QuasiMonteCarlo
using Random

default(; legend=:outertopright, size=(700, 400)) # hide

Random.seed!(42) # hide
#md nothing # hide

# ## Define the objective function
#
# We will use the Himmelblau function, a well-known multi-modal test function with four global minima.
# The function is defined as: ``f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2``
himmelblau(x::AbstractVector) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
∇himmelblau(x) = ForwardDiff.gradient(himmelblau, x)

# Combined function that returns both value and gradient for our gradient-enhanced GP
f_val_grad(x) = [himmelblau(x); ∇himmelblau(x)]

global_min = 0.0 # hide

d = 2
lower = [-6.0, -6.0]
upper = [6.0, 6.0]
domain = ContinuousDomain(lower, upper)

resolution = 100 # hide
X = range(lower[1], upper[1]; length=resolution) # hide
Y = range(lower[2], upper[2]; length=resolution) # hide
x_mins = [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]] # hide

p1 = contour( # hide
    X, # hide
    Y, # hide
    (x, y) -> himmelblau([x, y]); # hide
    fill=true, # hide
    levels=50, # hide
    c=:coolwarm, # hide
    title="Himmelblau Function", # hide
    xlabel="x₁", # hide
    ylabel="x₂", # hide
) # hide

scatter!( # hide
    [p[1] for p in x_mins], # hide
    [p[2] for p in x_mins]; # hide
    label="Global minima", # hide
    color=:red, # hide
    markersize=5, # hide
    legend=:bottomright, # hide
) # hide

# ## Initial Training Data and Model Setup
#
# We'll use a gradient-enhanced Gaussian Process with an approximate Matérn 5/2 kernel.
# For better space coverage, we generate initial training data using Sobol sampling.
σ² = 1e-12

n_train = 8
x_train = [
    collect(col) for
    col in eachcol(QuasiMonteCarlo.sample(n_train, lower, upper, SobolSample()))
]

# Evaluate function and gradients at training points
y_train = f_val_grad.(x_train)

# Setup the base model that we'll use across all configurations
base_model = GradientGP(ApproxMatern52Kernel(), d+1, σ²)

# ## Configuration Setup
#
# We'll compare different combinations of hyperparameter optimisation strategies and standardisation modes.
# This comprehensive comparison will help us understand how these settings affect optimisation performance.
#
# ### Hyperparameter Optimisation Strategies:
# - **"all"**: Optimise all kernel hyperparameters (length scales, signal variance, etc.)
# - **"length_scale_only"**: Only optimise the length scale parameters
# - **nothing**: Use fixed hyperparameters (no optimisation)
#
# ### Standardisation Modes:
# - **"mean_scale"**: Remove empirical mean and scale by standard deviation (default)
# - **"scale_only"**: Only scale by standard deviation
# - **"mean_only"**: Only remove empirical mean, no scaling
# - **nothing**: No standardisation applied
test_configs = [ # hide
    ("HP:all + MeanScale", "all", "mean_scale"), # hide
    ("HP:all + ScaleOnly", "all", "scale_only"), # hide
    ("HP:all + MeanOnly", "all", "mean_only"), # hide
    ("HP:all + NoStd", "all", nothing), # hide
    ("HP:length + MeanScale", "length_scale_only", "mean_scale"), # hide
    ("HP:length + ScaleOnly", "length_scale_only", "scale_only"), # hide
    ("HP:length + MeanOnly", "length_scale_only", "mean_only"), # hide
    ("HP:length + NoStd", "length_scale_only", nothing), # hide
    ("HP:none + MeanScale", nothing, "mean_scale"), # hide
    ("HP:none + ScaleOnly", nothing, "scale_only"), # hide
    ("HP:none + MeanOnly", nothing, "mean_only"), # hide
    ("HP:none + NoStd", nothing, nothing), # hide
]# hide

# ## Running the Optimisation Comparison
#
# We will run Bayesian optimisation with each configuration and collect performance metrics.
function run_comparison(n_iterations) # hide 
    results = Dict{String,NamedTuple}() # hide 

    for (config_name, hyper_params, standardise_mode) in test_configs # hide
        model = deepcopy(base_model) # hide

        best_y = minimum(first.(y_train)) # hide
        acq_func = ExpectedImprovement(0.0, best_y) # hide

        problem = BOStruct( # hide
            f_val_grad, # hide
            acq_func, # hide
            model, # hide
            domain, # hide
            x_train, # hide
            y_train, # hide
            n_iterations, # hide
            0.0,   # hide
        ) # hide

        start_time = time() # hide

        try # hide
            result, _, standard_params = AbstractBayesOpt.optimize( # hide
                problem;
                hyper_params=hyper_params,
                standardize=standardise_mode, # hide 
            ) # hide

            end_time = time() # hide
            elapsed_time = end_time - start_time # hide

            xs = result.xs # hide
            ys_values = first.(result.ys_non_std) # hide 

            optimal_idx = argmin(ys_values) # hide
            optimal_point = xs[optimal_idx] # hide
            optimal_value = minimum(ys_values) # hide

            all_evals = himmelblau.(xs) # hide
            running_min = accumulate(min, all_evals) # hide

            errors = max.(running_min .- global_min, 1e-16) # hide

            results[config_name] = ( # hide
                xs=xs, # hide
                ys_values=ys_values, # hide
                running_min=running_min, # hide
                errors=errors, # hide
                optimal_point=optimal_point, # hide
                optimal_value=optimal_value, # hide
                error_from_global=abs(optimal_value - global_min), # hide
                elapsed_time=elapsed_time, # hide
                hyper_params=hyper_params, # hide
                standardize=standardise_mode, # hide
                standard_params=standard_params, # hide
                n_evaluations=length(xs), # hide
            ) # hide

        catch e # hide
            @warn "ERROR in configuration $config_name: $e" # hide
            results[config_name] = ( # hide
                error_from_global=Inf, # hide
                elapsed_time=Inf, # hide
                hyper_params=hyper_params, # hide
                standardize=standardise_mode, # hide
                n_evaluations=0, # hide
            ) # hide
        end # hide
    end # hide

    return results # hide
end; # hide

# ### Execute the comparison
#
# Let's run the optimisation with all 12 different configurations. This will take some time
# as we're testing various combinations of hyperparameter optimisation and standardisation settings.
@info "Starting comparison..." # hide
results = run_comparison(30)

# ## Results Analysis and Visualisation
#
function plot_convergence_comparison(results) # hide
    p = plot(; # hide
        title="Himmelblau Optimisation: Hyperparameter & Standardisation Comparison", # hide
        xlabel="Number of iterations", # hide
        ylabel="Error from global minimum", # hide
        yaxis=:log, # hide
        legend=:bottomleft, # hide
        linewidth=2, # hide
        size=(1200, 800), # hide
    ) # hide

    colors = [ # hide
        :blue, # hide
        :lightblue, # hide
        :cyan, # hide
        :gray, # hide
        :red, # hide
        :pink, # hide
        :orange, # hide
        :brown, # hide
        :green, # hide
        :lightgreen, # hide
        :yellow, # hide
        :purple, # hide
    ] # hide
    styles = [ # hide
        :solid, # hide
        :dash, # hide
        :dot, # hide
        :dashdot, # hide
        :solid, # hide
        :dash, # hide
        :dot, # hide
        :dashdot, # hide
        :solid, # hide
        :dash, # hide
        :dot, # hide
        :dashdot, # hide
    ] # hide

    config_names = [ # hide
        "HP:all + MeanScale", # hide
        "HP:all + ScaleOnly", # hide
        "HP:all + MeanOnly", # hide
        "HP:all + NoStd", # hide
        "HP:length + MeanScale", # hide
        "HP:length + ScaleOnly", # hide
        "HP:length + MeanOnly", # hide
        "HP:length + NoStd", # hide
        "HP:none + MeanScale", # hide
        "HP:none + ScaleOnly", # hide
        "HP:none + MeanOnly", # hide
        "HP:none + NoStd", # hide
    ] # hide

    for (i, config_name) in enumerate(config_names) # hide
        if haskey(results, config_name) && haskey(results[config_name], :errors) # hide
            result = results[config_name] # hide
            plot!( # hide
                p, # hide
                1:length(result.errors), # hide
                result.errors; # hide
                label=config_name, # hide
                color=colors[i], # hide
                linestyle=styles[i], # hide
            ) # hide
        end # hide
    end # hide

    vspan!(p, [1, n_train]; color=:gray, alpha=0.2, label="Initial data") # hide

    return p # hide
end; #hide

# ### Create and display the convergence plot
#
# This comprehensive plot shows the optimisation performance of all 12 configurations.
# Each line represents a different combination of hyperparameter optimisation and standardisation.
conv_plot = plot_convergence_comparison(results) # hide
