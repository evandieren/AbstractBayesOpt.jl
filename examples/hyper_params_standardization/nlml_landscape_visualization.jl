# OUT OF DATE

"""
NLML Landscape Visualization for Gradient and Standard GP on Himmelblau Function

This script creates visualizations of the negative log marginal likelihood (NLML) landscape
for both gradient-enhanced and standard Gaussian Process models using the Himmelblau function.
The visualizations help understand the optimization landscape that hyperparameter optimization
algorithms navigate when fitting GP models.

The script generates:
1. 2D heatmaps of NLML over lengthscale and scale parameters
2. 3D surface plots for better visualization of the landscape
3. Side-by-side comparison of gradient vs standard GP landscapes
4. Contour plots with optimization paths if available
"""

using AbstractGPs, KernelFunctions
using Plots
using Distributions
using ForwardDiff
using AbstractBayesOpt
using LinearAlgebra
using LaTeXStrings
using QuasiMonteCarlo
using Random: Random
using Statistics

# Set random seed for reproducibility
Random.seed!(42)

# Define the Himmelblau function and its gradient
himmelblau(x::AbstractVector) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
∇himmelblau(x::AbstractVector) = ForwardDiff.gradient(himmelblau, x)
himmelblau_val_grad(x::AbstractVector) = [himmelblau(x); ∇himmelblau(x)]

# Problem setup
d = 2
lower = [-4.0, -4.0]
upper = [4.0, 4.0]
domain = ContinuousDomain(lower, upper)
σ² = 1e-6  # Noise variance

# Generate training data using Sobol sampling for better coverage
n_train = 120
x_train = [
    collect(col) for
    col in eachcol(QuasiMonteCarlo.sample(n_train, lower, upper, SobolSample()))
]

println("Training points:")
for (i, x) in enumerate(x_train)
    println("  x$i = $x, f(x) = $(round(himmelblau(x), digits=3))")
end

# Evaluate function at training points
y_train_standard = [himmelblau(x) for x in x_train]  # Standard GP: only function values
y_train_gradient = himmelblau_val_grad.(x_train)     # Gradient GP: function values + gradients

println("\nTraining data prepared:")
println("  Standard GP: $(length(y_train_standard)) function evaluations")
println("  Gradient GP: $(length(y_train_gradient)) function+gradient evaluations")

# Setup kernels and models
kernel_constructor = ApproxMatern52Kernel()

# Standard GP model
standard_kernel = 1.0 * (kernel_constructor ∘ ScaleTransform(1.0))
standard_model = StandardGP(standard_kernel, σ²)

# Gradient GP model  
gradient_kernel = 1.0 * (kernel_constructor ∘ ScaleTransform(1.0))
grad_kernel = gradKernel(gradient_kernel)
gradient_model = GradientGP(grad_kernel, d+1, σ²)

# Prepare data for NLML computation
# Standard GP data
x_standard = x_train
y_standard = reduce(vcat, y_train_standard)

# Gradient GP data
x_gradient = KernelFunctions.MOInputIsotopicByOutputs(x_train, d+1)
y_gradient = vec(permutedims(reduce(hcat, y_train_gradient)))

println("\nData shapes for NLML computation:")
println("  Standard GP: x=$(length(x_standard)), y=$(length(y_standard))")
println("  Gradient GP: x=$(length(x_gradient)), y=$(length(y_gradient))")

# Define parameter ranges for NLML landscape

log_lengthscale_range = range(log(1e-3), log(1e3); length=100)
log_scale_range = range(log(1e-3), log(1e6); length=100)

println("\nParameter ranges:")
println(
    "  Lengthscale: $(round(exp(log_lengthscale_range[1]), digits=3)) to $(round(exp(log_lengthscale_range[end]), digits=3))",
)
println(
    "  Scale: $(round(exp(log_scale_range[1]), digits=3)) to $(round(exp(log_scale_range[end]), digits=3))",
)

# Compute NLML landscapes
function compute_nlml_landscape(
    model, x_data, y_data, log_ls_range, log_scale_range, model_name
)
    println("\nComputing NLML landscape for $model_name...")

    nlml_values = zeros(length(log_ls_range), length(log_scale_range))

    total_combinations = length(log_ls_range) * length(log_scale_range)
    completed = 0

    for (i, log_ls) in enumerate(log_ls_range)
        for (j, log_scale) in enumerate(log_scale_range)
            try
                params = [log_ls, log_scale]
                nlml_val = nlml(model, params, x_data, y_data)
                nlml_values[i, j] = nlml_val

                # Handle infinite or very large values
                if !isfinite(nlml_val) || nlml_val > 1e9
                    println(
                        "Warning: NLML value out of bounds at (log_ls=$(round(log_ls, digits=2)), log_scale=$(round(log_scale, digits=2))): $nlml_val",
                    )
                    nlml_values[i, j] = 1e9
                end
            catch e
                # If computation fails, assign a large value
                nlml_values[i, j] = 1e9
            end

            completed += 1
            if completed % 500 == 0
                progress = round(100 * completed / total_combinations; digits=1)
                println("  Progress: $progress% ($completed/$total_combinations)")
            end
        end
    end

    println("  Completed NLML landscape computation for $model_name")
    println(
        "  NLML range: $(round(minimum(nlml_values), digits=2)) to $(round(maximum(nlml_values), digits=2))",
    )

    return nlml_values
end

# Compute landscapes for both models
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

# Find optimal parameters for both models
function find_optimal_params(nlml_values, log_ls_range, log_scale_range)
    min_idx = argmin(nlml_values)
    i, j = Tuple(min_idx)
    opt_log_ls = log_ls_range[i]
    opt_log_scale = log_scale_range[j]
    opt_nlml = nlml_values[i, j]

    return (
        log_lengthscale=opt_log_ls,
        log_scale=opt_log_scale,
        lengthscale=exp(opt_log_ls),
        scale=exp(opt_log_scale),
        nlml=opt_nlml,
    )
end

opt_standard = find_optimal_params(nlml_standard, log_lengthscale_range, log_scale_range)
opt_gradient = find_optimal_params(nlml_gradient, log_lengthscale_range, log_scale_range)

println("\nOptimal parameters found:")
println("Standard GP:")
println(
    "  Lengthscale: $(round(opt_standard.lengthscale, digits=3)) (log: $(round(opt_standard.log_lengthscale, digits=3)))",
)
println(
    "  Scale: $(round(opt_standard.scale, digits=3)) (log: $(round(opt_standard.log_scale, digits=3)))",
)
println("  NLML: $(round(opt_standard.nlml, digits=3))")

println("\nGradient GP:")
println(
    "  Lengthscale: $(round(opt_gradient.lengthscale, digits=3)) (log: $(round(opt_gradient.log_lengthscale, digits=3)))",
)
println(
    "  Scale: $(round(opt_gradient.scale, digits=3)) (log: $(round(opt_gradient.log_scale, digits=3)))",
)
println("  NLML: $(round(opt_gradient.nlml, digits=3))")

# Create visualization functions
function create_2d_heatmap(
    nlml_values, log_ls_range, log_scale_range, title_str, optimal_params=nothing
)
    # Clip extreme values for better visualization
    # clipped_nlml = clamp.(nlml_values, minimum(nlml_values), minimum(nlml_values) + 50)

    p = heatmap(
        log_scale_range,
        log_ls_range,
        log.(nlml_values);
        title=title_str,
        xlabel="log Scale Parameter",
        ylabel="log Lengthscale Parameter",
        color=:viridis,
        aspect_ratio=:equal,
        size=(600, 500),
    )

    # Add optimal point if provided
    if optimal_params !== nothing
        scatter!(
            p,
            [log.(optimal_params.scale)],
            [log.(optimal_params.lengthscale)];
            color=:red,
            markersize=8,
            markershape=:star,
            label="Optimal (NLML=$(round(optimal_params.nlml, digits=1)))",
        )
    end

    return p
end

function create_3d_surface(nlml_values, log_ls_range, log_scale_range, title_str)
    # Clip extreme values for better visualization
    # clipped_nlml = clamp.(nlml_values, minimum(nlml_values), minimum(nlml_values) + 30)

    p = surface(
        log_scale_range,
        log_ls_range,
        log.(nlml_values);
        title=title_str,
        xlabel="log Scale Parameter",
        ylabel="log Lengthscale Parameter",
        zlabel="NLML",
        color=:viridis,
        camera=(45, 60),
        size=(700, 600),
    )

    return p
end

function create_contour_plot(
    nlml_values, log_ls_range, log_scale_range, title_str, optimal_params=nothing
)
    # Create contour levels
    # min_nlml = minimum(nlml_values)
    # max_nlml = min_nlml + 20  # Show contours within reasonable range
    # levels = range(min_nlml, max_nlml, length=15)

    p = contourf(
        log_scale_range,
        log_ls_range,
        log.(nlml_values);
        title=title_str,
        xlabel="log Scale Parameter",
        ylabel="log Lengthscale Parameter",
        color=:viridis,
        fill=true,
        levels=50,
        aspect_ratio=:equal,
        size=(600, 500),
    )

    # Add optimal point if provided
    if optimal_params !== nothing
        scatter!(
            p,
            [log.(optimal_params.scale)],
            [log.(optimal_params.lengthscale)];
            color=:red,
            markersize=8,
            markershape=:star,
            label="Optimal",
        )
    end

    return p
end

# Generate all visualizations
println("\nGenerating visualizations...")

# Contour plots
contour_standard = create_contour_plot(
    nlml_standard,
    log_lengthscale_range,
    log_scale_range,
    "Standard GP log(NLML) Contours",
    opt_standard,
)

contour_gradient = create_contour_plot(
    nlml_gradient,
    log_lengthscale_range,
    log_scale_range,
    "Gradient GP log(NLML) Contours",
    opt_gradient,
)

combined_contours = plot(
    contour_standard,
    contour_gradient;
    layout=(1, 2),
    size=(1200, 500),
    plot_title="NLML Contour Comparison: Standard vs Gradient GP",
)

display(combined_contours)
println("\nSaving visualizations...")
# savefig(combined_contours, "nlml_contours_comparison.png") 
