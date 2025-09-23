```@meta
EditURL = "../../literate/tutorials/nlml_landscape_visualization.jl"
```

# AbstractBayesOpt Tutorial: NLML Landscape Visualisation

This tutorial shows how to visualise the negative log marginal likelihood (NLML) landscape for Gaussian Process models.
The NLML landscape shows how the model likelihood changes as we vary the kernel hyperparameters.
Understanding this landscape is crucial for:
- Choosing appropriate hyperparameter optimisation strategies
- Understanding why some configurations converge faster than others
- Identifying potential issues like local minima or ill-conditioned regions

## Setup

Loading the necessary packages.

````@example nlml_landscape_visualization
using AbstractBayesOpt
using AbstractGPs
using Plots
using ForwardDiff
using QuasiMonteCarlo
using Random

default(; legend=:outertopright, size=(700, 400)) # hide

Random.seed!(42) # hide
nothing # hide
````

## Define the objective function

We'll use the Himmelblau function again, as it provides a good test case with complex structure.

````@example nlml_landscape_visualization
himmelblau(x::AbstractVector) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
∇himmelblau(x::AbstractVector) = ForwardDiff.gradient(himmelblau, x)
f_val_grad(x::AbstractVector) = [himmelblau(x); ∇himmelblau(x)];
nothing #hide
````

## Problem Setup and Training Data

````@example nlml_landscape_visualization
d = 2
lower = [-4.0, -4.0]
upper = [4.0, 4.0]
domain = ContinuousDomain(lower, upper)
σ² = 1e-6
````

Generate training data using Sobol sampling for better coverage

````@example nlml_landscape_visualization
n_train = 75
x_train = [
    collect(col) for
    col in eachcol(QuasiMonteCarlo.sample(n_train, lower, upper, SobolSample()))
]
````

Evaluate function at training points for both model types

````@example nlml_landscape_visualization
y_train_standard = [himmelblau(x) for x in x_train]  # Standard GP: only function values
y_train_gradient = f_val_grad.(x_train);    # Gradient GP: function values + gradients
nothing #hide
````

### Setup Gaussian Process Models

We'll create both standard and gradient-enhanced GP models using the same kernel type
but configured for their respective data structures.

````@example nlml_landscape_visualization
kernel = ApproxMatern52Kernel()

standard_model = StandardGP(kernel, σ²)
gradient_model = GradientGP(kernel, d+1, σ²)
````

Prepare data for NLML computation (this is done under the hood in AbstractBayesOpt.jl)

Standard GP data structure

````@example nlml_landscape_visualization
x_standard = x_train;
y_standard = reduce(vcat, y_train_standard);
nothing #hide
````

Gradient GP data structure

````@example nlml_landscape_visualization
x_gradient = KernelFunctions.MOInputIsotopicByOutputs(x_train, d+1);
y_gradient = vec(permutedims(reduce(hcat, y_train_gradient)));

println("Data shapes for NLML computation:") # hide
println("  Standard GP: x=$(length(x_standard)), y=$(length(y_standard))") # hide
println("  Gradient GP: x=$(length(x_gradient)), y=$(length(y_gradient))") # hide
````

### Define Parameter Ranges for NLML Landscape

We will create a grid of hyperparameter values to evaluate the NLML landscape.
The parameters we will vary are:
- Length scale: Controls how quickly the function varies spatially
- Scale parameter: Controls the overall magnitude of function variations

````@example nlml_landscape_visualization
log_lengthscale_range = range(log(1e-3), log(1e3); length=100) # hide
log_scale_range = range(log(1e-3), log(1e6); length=100) # hide

println("Parameter ranges:") # hide
println( # hide
    "  Lengthscale: $(round(exp(log_lengthscale_range[1]), digits=3)) to $(round(exp(log_lengthscale_range[end]), digits=3))", # hide
) # hide
println( # hide
    "  Scale: $(round(exp(log_scale_range[1]), digits=3)) to $(round(exp(log_scale_range[end]), digits=3))", # hide
) # hide

function compute_nlml_landscape( # hide
    model, x_data, y_data, log_ls_range, log_scale_range, model_name # hide
) # hide
    @info "Computing NLML landscape for $model_name..." # hide

    nlml_values = zeros(length(log_ls_range), length(log_scale_range)) # hide

    total_combinations = length(log_ls_range) * length(log_scale_range) # hide
    completed = 0 # hide

    for (i, log_ls) in enumerate(log_ls_range) # hide
        for (j, log_scale) in enumerate(log_scale_range) # hide
            try # hide
                params = [log_ls, log_scale] # hide
                nlml_val = nlml(model, params, x_data, y_data) # hide
                nlml_values[i, j] = nlml_val # hide

                if !isfinite(nlml_val) || nlml_val > 1e9 # hide
                    @warn "Warning: NLML value out of bounds at (log_ls=$(round(log_ls, digits=2)), log_scale=$(round(log_scale, digits=2))): $nlml_val" # hide
                    nlml_values[i, j] = 1e9 # hide
                end # hide
            catch e # hide
                nlml_values[i, j] = 1e9 # hide
            end # hide

            completed += 1 # hide
            if completed % 500 == 0 # hide
                progress = round(100 * completed / total_combinations; digits=1) # hide
                @info "  Progress: $progress% ($completed/$total_combinations)" # hide
            end # hide
        end # hide
    end # hide

    @info "  Completed NLML landscape computation for $model_name" # hide
    @info "  NLML range: $(round(minimum(nlml_values), digits=2)) to $(round(maximum(nlml_values), digits=2))" # hide

    return nlml_values # hide
end # hide
````

### Compute landscapes for both models

This computation may take several minutes depending on the grid resolution.
We're evaluating 10,000 parameter combinations for each model type.

````@example nlml_landscape_visualization
nlml_standard = compute_nlml_landscape(
    standard_model,
    x_standard,
    y_standard,
    log_lengthscale_range,
    log_scale_range,
    "Standard GP",
)

nlml_gradient = compute_nlml_landscape(
    gradient_model,
    x_gradient,
    y_gradient,
    log_lengthscale_range,
    log_scale_range,
    "Gradient GP",
)
````

This provides a 100x100 grid of NLML values for each model type.

### Optimal Parameters

We approximately provide the hyperparameter combinations that minimise the NLML for each model type.
In AbstractBayesOpt.jl, we optimise the MLML using Optim.jl's BFGS method.

````@example nlml_landscape_visualization
function find_optimal_params(nlml_values, log_ls_range, log_scale_range) # hide
    min_idx = argmin(nlml_values) # hide
    i, j = Tuple(min_idx) # hide
    opt_log_ls = log_ls_range[i] # hide
    opt_log_scale = log_scale_range[j] # hide
    opt_nlml = nlml_values[i, j] # hide

    return ( # hide
        log_lengthscale=opt_log_ls, # hide
        log_scale=opt_log_scale, # hide
        lengthscale=exp(opt_log_ls), # hide
        scale=exp(opt_log_scale), # hide
        nlml=opt_nlml, # hide
    ) # hide
end # hide

opt_standard = find_optimal_params(nlml_standard, log_lengthscale_range, log_scale_range) # hide
opt_gradient = find_optimal_params(nlml_gradient, log_lengthscale_range, log_scale_range) # hide

println("\nOptimal parameters found:") # hide
println("Standard GP:") # hide
println( # hide
    "  Lengthscale: $(round(opt_standard.lengthscale, digits=3)) (log: $(round(opt_standard.log_lengthscale, digits=3)))", # hide
) # hide
println( # hide
    "  Scale: $(round(opt_standard.scale, digits=3)) (log: $(round(opt_standard.log_scale, digits=3)))", # hide
) # hide
println("  NLML: $(round(opt_standard.nlml, digits=3))") # hide

println("\nGradient GP:") # hide
println( # hide
    "  Lengthscale: $(round(opt_gradient.lengthscale, digits=3)) (log: $(round(opt_gradient.log_lengthscale, digits=3)))", # hide
) # hide
println( # hide
    "  Scale: $(round(opt_gradient.scale, digits=3)) (log: $(round(opt_gradient.log_scale, digits=3)))", # hide
) # hide
println("  NLML: $(round(opt_gradient.nlml, digits=3))") # hide

function create_contour_plot( # hide
    nlml_values, log_ls_range, log_scale_range, title_str, optimal_params=nothing # hide
) # hide
    p = contourf( # hide
        log_scale_range, # hide
        log_ls_range, # hide
        log.(nlml_values); # hide
        title=title_str, # hide
        xlabel="log Scale Parameter", # hide
        ylabel="log Lengthscale Parameter", # hide
        color=:coolwarm, # hide
        fill=true, # hide
        levels=50, # hide
        aspect_ratio=:equal, # hide
        size=(600, 500), # hide
    ) # hide

    if optimal_params !== nothing # hide
        scatter!( # hide
            p, # hide
            [log.(optimal_params.scale)], # hide
            [log.(optimal_params.lengthscale)]; # hide
            color=:red, # hide
            markersize=8, # hide
            markershape=:star, # hide
            label="Optimal", # hide
            legend=:bottomright) # hide
    end # hide

    return p # hide
end; # hide
nothing #hide
````

## NLML Landscape Plots

These contour plots show the NLML landscape for both model types. The star indicates
the optimal hyperparameter combination found through the approximate minimisers over the 100x100 grid.
Darker blue regions correspond to lower NLML values (better likelihood).

````@example nlml_landscape_visualization
contour_standard = create_contour_plot( # hide
    nlml_standard, # hide
    log_lengthscale_range, # hide
    log_scale_range, # hide
    "Standard GP log(NLML) Contours", # hide
    opt_standard, # hide
) # hide

contour_gradient = create_contour_plot( # hide
    nlml_gradient, # hide
    log_lengthscale_range, # hide
    log_scale_range, # hide
    "Gradient GP log(NLML) Contours", # hide
    opt_gradient, # hide
) # hide

combined_contours = plot( # hide
    contour_standard, # hide
    contour_gradient; # hide
    layout=(1, 2), # hide
    size=(1200, 500), # hide
    plot_title="NLML Contour Comparison: Standard vs Gradient GP", # hide
) # hide
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

