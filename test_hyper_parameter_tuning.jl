# filepath: /home/vandiere/.julia/dev/BayesOpt/test_hyper_parameter_tuning.jl

"""
Test snippet to analyze the convexity of the NLML (Negative Log Marginal Likelihood) 
function used in hyperparameter optimization.

We manually add a few points from the Himmelblau function and plot the NLML surface
to visualize whether the optimization landscape is convex.
"""

using BayesOpt
using AbstractGPs, KernelFunctions
using Plots
using Statistics
using Random
using LinearAlgebra

# Set random seed for reproducibility
Random.seed!(42)

# Define the Himmelblau function
himmelblau(x::AbstractVector) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

# Manually add a few points from the Himmelblau function
# These are strategic points around different regions of the function
manual_points = [
    [-2.0, -1.0],  # One region
    [0.0, 0.0],    # Center
    [3.0, 2.0],    # Another region  
    [-3.5, -1.8],  # Near a local minimum
    [3.58, -1.85]  # Near another local minimum
]

# Evaluate the function at these points
manual_values = himmelblau.(manual_points)
println("Manual points and their function values:")
for (i, (point, value)) in enumerate(zip(manual_points, manual_values))
    println("Point $i: $point -> $value")
end

# Prepare data for GP
x_train = manual_points
y_train = [[y] for y in manual_values]  # Convert to Vector{Vector{Float64}}

# Standardize the data (zero mean, unit variance)
y_flat = reduce(vcat, y_train)
y_mean = mean(y_flat)
y_std = std(y_flat)
y_train_std = [[(y[1] - y_mean) / y_std] for y in y_train]

println("Data standardization:")
println("Original y range: [$(minimum(y_flat)), $(maximum(y_flat))]")
println("Standardized y range: [$(minimum(reduce(vcat, y_train_std))), $(maximum(reduce(vcat, y_train_std)))]")
println("y_mean = $y_mean, y_std = $y_std")

# Create models for both StandardGP and GradientGP
kernel_constructor = SqExponentialKernel()
noise_var = 1e-6

# StandardGP model
gp_standard = StandardGP(kernel_constructor, noise_var)

# GradientGP model (for 2D problem: function + 2 gradients = 3 outputs)
using ForwardDiff
himmelblau_with_grad(x) = [himmelblau(x); ForwardDiff.gradient(himmelblau, x)]

# Get gradient evaluations at manual points
manual_grad_values = himmelblau_with_grad.(manual_points)
# Standardize gradient data (function values get same standardization, gradients get scaled by y_std)
y_train_grad_std = []
for grad_val in manual_grad_values
    f_val_std = (grad_val[1] - y_mean) / y_std
    grad_vals_std = grad_val[2:end] / y_std  # Gradients scaled by same factor
    push!(y_train_grad_std, [f_val_std; grad_vals_std])
end

gp_gradient = GradientGP(gradKernel(kernel_constructor), 3, noise_var)

# Create parameter ranges to test NLML convexity
# We'll only vary log lengthscale, keeping log scale = 0.0 (scale = 1.0)
log_lengthscale_range = range(-3.0, 2.0, length=100)
fixed_log_scale = 0.0  # This corresponds to scale = 1.0

# Compute NLML over the lengthscale range for both models
nlml_values_standard = zeros(length(log_lengthscale_range))
nlml_values_gradient = zeros(length(log_lengthscale_range))

println("\nComputing NLML over lengthscale range...")

# For StandardGP
println("Computing NLML for StandardGP...")
for (i, log_ℓ) in enumerate(log_lengthscale_range)
    params = [log_ℓ, fixed_log_scale]  # [log_lengthscale, log_scale]
    try
        nlml_val = nlml(gp_standard, params, kernel_constructor, x_train, reduce(vcat, y_train_std))
        nlml_values_standard[i] = nlml_val
    catch e
        nlml_values_standard[i] = 1e6
    end
end

# For GradientGP  
println("Computing NLML for GradientGP...")
for (i, log_ℓ) in enumerate(log_lengthscale_range)
    params = [log_ℓ, fixed_log_scale]  # [log_lengthscale, log_scale]
    try
        # Prepare gradient data for GradientGP
        x_grad = KernelFunctions.MOInputIsotopicByOutputs(x_train, 3)
        y_grad = vec(permutedims(reduce(hcat, y_train_grad_std)))
        nlml_val = nlml(gp_gradient, params, kernel_constructor, x_grad, y_grad)
        nlml_values_gradient[i] = nlml_val
    catch e
        nlml_values_gradient[i] = 1e6
    end
end

# Find the minimum NLML and its location for both models
min_nlml_standard = minimum(nlml_values_standard)
optimal_idx_standard = argmin(nlml_values_standard)
optimal_log_ℓ_standard = log_lengthscale_range[optimal_idx_standard]

min_nlml_gradient = minimum(nlml_values_gradient)
optimal_idx_gradient = argmin(nlml_values_gradient)
optimal_log_ℓ_gradient = log_lengthscale_range[optimal_idx_gradient]

println("\nNLML Analysis Results:")
println("StandardGP:")
println("  Minimum NLML: $min_nlml_standard")
println("  Optimal log lengthscale: $optimal_log_ℓ_standard (lengthscale: $(exp(optimal_log_ℓ_standard)))")
println("  Fixed log scale: $fixed_log_scale (scale: $(exp(fixed_log_scale)))")
println("\nGradientGP:")
println("  Minimum NLML: $min_nlml_gradient") 
println("  Optimal log lengthscale: $optimal_log_ℓ_gradient (lengthscale: $(exp(optimal_log_ℓ_gradient)))")
println("  Fixed log scale: $fixed_log_scale (scale: $(exp(fixed_log_scale)))")

# Create plots to visualize the NLML curves
p1 = plot(log_lengthscale_range, nlml_values_standard,
         xlabel="Log Lengthscale", 
         ylabel="NLML",
         title="NLML vs Log Lengthscale (StandardGP, Scale=1.0)",
         linewidth=2,
         label="StandardGP NLML",
         color=:blue)

# Mark the optimum for StandardGP
scatter!([optimal_log_ℓ_standard], [min_nlml_standard], 
         marker=:star, markersize=10, color=:red, 
         label="StandardGP Optimum")

p2 = plot(log_lengthscale_range, nlml_values_gradient,
         xlabel="Log Lengthscale", 
         ylabel="NLML",
         title="NLML vs Log Lengthscale (GradientGP, Scale=1.0)",
         linewidth=2,
         label="GradientGP NLML",
         color=:green)

# Mark the optimum for GradientGP
scatter!([optimal_log_ℓ_gradient], [min_nlml_gradient], 
         marker=:star, markersize=10, color=:red, 
         label="GradientGP Optimum")

# Comparison plot
p3 = plot(log_lengthscale_range, nlml_values_standard,
         xlabel="Log Lengthscale", 
         ylabel="NLML",
         title="NLML Comparison: StandardGP vs GradientGP",
         linewidth=2,
         label="StandardGP",
         color=:blue)

plot!(log_lengthscale_range, nlml_values_gradient,
      linewidth=2,
      label="GradientGP", 
      color=:green)

scatter!([optimal_log_ℓ_standard], [min_nlml_standard], 
         marker=:star, markersize=8, color=:blue, 
         label="StandardGP Opt")

scatter!([optimal_log_ℓ_gradient], [min_nlml_gradient], 
         marker=:star, markersize=8, color=:green, 
         label="GradientGP Opt")

# Create a zoomed-in plot around the optima
zoom_range = 0.5  # Range around optima to zoom into
zoom_min = min(optimal_log_ℓ_standard, optimal_log_ℓ_gradient) - zoom_range
zoom_max = max(optimal_log_ℓ_standard, optimal_log_ℓ_gradient) + zoom_range
zoom_mask = (log_lengthscale_range .>= zoom_min) .& (log_lengthscale_range .<= zoom_max)

p4 = plot(log_lengthscale_range[zoom_mask], nlml_values_standard[zoom_mask],
         xlabel="Log Lengthscale", 
         ylabel="NLML",
         title="NLML Comparison (Zoomed around Optima)",
         linewidth=2,
         label="StandardGP",
         color=:blue)

plot!(log_lengthscale_range[zoom_mask], nlml_values_gradient[zoom_mask],
      linewidth=2,
      label="GradientGP", 
      color=:green)

scatter!([optimal_log_ℓ_standard], [min_nlml_standard], 
         marker=:star, markersize=8, color=:blue, 
         label="StandardGP Opt")

scatter!([optimal_log_ℓ_gradient], [min_nlml_gradient], 
         marker=:star, markersize=8, color=:green, 
         label="GradientGP Opt")

# Combine all plots
final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))

# Save the plot
savefig(final_plot, "nlml_lengthscale_optimization.png")
println("\nPlot saved as 'nlml_lengthscale_optimization.png'")

# Analyze convexity by examining the second derivative numerically
println("\nConvexity Analysis:")

# Function to evaluate NLML at given lengthscale for StandardGP
nlml_func_standard = (log_ℓ) -> begin
    params = [log_ℓ, fixed_log_scale]
    try
        return nlml(gp_standard, params, kernel_constructor, x_train, reduce(vcat, y_train_std))
    catch
        return 1e6
    end
end

# Function to evaluate NLML at given lengthscale for GradientGP  
nlml_func_gradient = (log_ℓ) -> begin
    params = [log_ℓ, fixed_log_scale]
    try
        x_grad = KernelFunctions.MOInputIsotopicByOutputs(x_train, 3)
        y_grad = vec(permutedims(reduce(hcat, y_train_grad_std)))
        return nlml(gp_gradient, params, kernel_constructor, x_grad, y_grad)
    catch
        return 1e6
    end
end

# Compute numerical second derivatives (convexity test)
δ = 0.01  # Small perturbation for numerical derivatives

# StandardGP second derivative
d2_standard = (nlml_func_standard(optimal_log_ℓ_standard + δ) - 
               2 * nlml_func_standard(optimal_log_ℓ_standard) + 
               nlml_func_standard(optimal_log_ℓ_standard - δ)) / δ^2

# GradientGP second derivative
d2_gradient = (nlml_func_gradient(optimal_log_ℓ_gradient + δ) - 
               2 * nlml_func_gradient(optimal_log_ℓ_gradient) + 
               nlml_func_gradient(optimal_log_ℓ_gradient - δ)) / δ^2

println("Numerical second derivatives at optima:")
println("StandardGP: d²NLML/d(log_ℓ)² = $d2_standard")
println("GradientGP: d²NLML/d(log_ℓ)² = $d2_gradient")

# Check convexity
println("\nConvexity Assessment:")
if d2_standard > 0
    println("✓ StandardGP: NLML is locally convex w.r.t. log lengthscale (second derivative > 0)")
else
    println("✗ StandardGP: NLML is NOT locally convex w.r.t. log lengthscale (second derivative ≤ 0)")
end

if d2_gradient > 0
    println("✓ GradientGP: NLML is locally convex w.r.t. log lengthscale (second derivative > 0)")
else
    println("✗ GradientGP: NLML is NOT locally convex w.r.t. log lengthscale (second derivative ≤ 0)")
end

# Display the plots
display(final_plot)

println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println("• Used $(length(manual_points)) manual points from the Himmelblau function")
println("• Data standardized: mean = $(round(y_mean, digits=3)), std = $(round(y_std, digits=3))")
println("• Fixed scale = 1.0 (log scale = 0.0), optimized only lengthscale")
println("• Analyzed NLML over log lengthscale ∈ [$(log_lengthscale_range[1]), $(log_lengthscale_range[end])]")
println("\nOptimal lengthscales:")
println("• StandardGP:  log ℓ = $(round(optimal_log_ℓ_standard, digits=3)) (ℓ = $(round(exp(optimal_log_ℓ_standard), digits=3)))")
println("• GradientGP:  log ℓ = $(round(optimal_log_ℓ_gradient, digits=3)) (ℓ = $(round(exp(optimal_log_ℓ_gradient), digits=3)))")
println("\nMinimum NLML values:")
println("• StandardGP:  $(round(min_nlml_standard, digits=3))")
println("• GradientGP:  $(round(min_nlml_gradient, digits=3))")
println("\n• Check the plots to visually assess convexity of the lengthscale optimization")
println("• Second derivative analysis provides local convexity information")
println("• GradientGP incorporates derivative information and may have different optimal lengthscale")