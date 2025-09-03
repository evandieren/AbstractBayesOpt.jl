"""
Test to verify that adding a prior mean and doing scale_only is equivalent to doing center_scale.

This test checks that:
ZeroMean + center_scale ≡ PriorMean + scale_only

for both StandardGP and GradientGP.
"""

using AbstractBayesOpt
using AbstractGPs
using KernelFunctions
using Statistics
using Random

Random.seed!(42)

# Test function
f(x) = sin(sum(x)) + 0.5 * sum(x.^2)
∇f(x) = cos(sum(x)) .* ones(length(x)) + x
f_val_grad(x) = [f(x); ∇f(x)]

# Generate test data
dim = 2
n_points = 8
x_test = [[randn(dim)...] for _ in 1:n_points]

# Test for StandardGP
println("=== Testing StandardGP ===")
y_test_standard = [f.(x_test)...]
y_test_standard = [[y] for y in y_test_standard]

# Compute empirical mean for prior
empirical_mean = mean(reduce(vcat, y_test_standard))
println("Empirical mean: $empirical_mean")



println("Testing equivalence of zero-mean + mean_only and prior mean with no standardization")
# Setup 1: ZeroMean + center_scale
kernel = 1*(SqExponentialKernel() ∘ ScaleTransform(1))
model1 = StandardGP(kernel, 1e-12)
bo1 = BOStruct(f, ExpectedImprovement(0.01, minimum(reduce(vcat, y_test_standard))), 
               model1, SqExponentialKernel(), ContinuousDomain([-5.0, -5.0], [5.0, 5.0]), 
               x_test, y_test_standard, 10, 0.0)
# Setup 2: ConstMean(empirical_mean) + scale_only  
model2 = StandardGP(kernel, 1e-12, mean=ConstMean(empirical_mean))
bo2 = BOStruct(f, ExpectedImprovement(0.01, minimum(reduce(vcat, y_test_standard))), 
               model2, SqExponentialKernel(), ContinuousDomain([-5.0, -5.0], [5.0, 5.0]), 
               x_test, y_test_standard, 10, 0.0)

# Apply standardizations
bo1_std, params1 = standardize_problem(bo1, choice="mean_only")
# Checking whether mean_only with zero mean  + \bar{Y} is equivalent to prior mean without doing anything else -> true 
bo2.model = update!(bo2.model,x_test,y_test_standard)
params2 = ([0.0],[1.0])

println(bo1_std.model.gp.kernel)
println(bo2.model.gp.kernel)

println("Setup 1 (ZeroMean + center_scale): μ=$(params1[1]), σ=$(params1[2])")
println("Setup 2 (ConstMean + scale_only): μ=$(params2[1]), σ=$(params2[2])")

# Test points for prediction
x_pred = [[0.5, -0.3], [-1.2, 0.8], [2.1, -1.5]]

# Get predictions from both setups
pred1_mean = [posterior_mean(bo1_std.model, x) for x in x_pred]
pred1_var = [posterior_var(bo1_std.model, x) for x in x_pred]

pred2_mean = [posterior_mean(bo2.model, x) for x in x_pred]
pred2_var = [posterior_var(bo2.model, x) for x in x_pred]

println("\nStandardGP Predictions (un-standardized):")
println("Setup 1 means: $pred1_mean")
println("Setup 2 means: $pred2_mean")
println("Setup 1 vars:  $pred1_var")
println("Setup 2 vars:  $pred2_var")

mean_diff = maximum(abs.(pred1_mean .- pred2_mean))
var_diff = maximum(abs.(pred1_var .- pred2_var))
println("Max mean difference: $mean_diff")
println("Max variance difference: $var_diff")
println("StandardGP equivalence: $(mean_diff < 1e-10 && var_diff < 1e-10)")
@assert (mean_diff < 1e-10 && var_diff < 1e-10)

println("Now testing zero mean + center_scale should be equal to prior mean + scale_only")
# Setup 1: ZeroMean + center_scale
kernel = 1*(SqExponentialKernel() ∘ ScaleTransform(1))
model1 = StandardGP(kernel, 1e-12)
bo1 = BOStruct(f, ExpectedImprovement(0.01, minimum(reduce(vcat, y_test_standard))), 
               model1, SqExponentialKernel(), ContinuousDomain([-5.0, -5.0], [5.0, 5.0]), 
               x_test, y_test_standard, 10, 0.0)
# Setup 2: ConstMean(empirical_mean) + scale_only  
model2 = StandardGP(kernel, 1e-12, mean=ConstMean(empirical_mean))
bo2 = BOStruct(f, ExpectedImprovement(0.01, minimum(reduce(vcat, y_test_standard))), 
               model2, SqExponentialKernel(), ContinuousDomain([-5.0, -5.0], [5.0, 5.0]), 
               x_test, y_test_standard, 10, 0.0)

# Apply standardizations
bo1_std, params1 = standardize_problem(bo1, choice="mean_scale")
bo2_std, params2 = standardize_problem(bo2, choice="scale_only")
# Checking whether mean_only with zero mean  + \bar{Y} is equivalent to prior mean without doing anything else -> true 

println(bo1_std.model.gp.kernel)
println(bo2_std.model.gp.kernel)

println("Setup 1 (ZeroMean + center_scale): μ=$(params1[1]), σ=$(params1[2])")
println("Setup 2 (ConstMean + scale_only): μ=$(params2[1]), σ=$(params2[2])")

# Test points for prediction
x_pred = [[0.5, -0.3], [-1.2, 0.8], [2.1, -1.5]]

# Get predictions from both setups
pred1_mean = [posterior_mean(bo1_std.model, x) for x in x_pred]
pred1_var = [posterior_var(bo1_std.model, x) for x in x_pred]

pred2_mean = [posterior_mean(bo2.model, x) for x in x_pred]
pred2_var = [posterior_var(bo2.model, x) for x in x_pred]

println("\nStandardGP Predictions (un-standardized):")
println("Setup 1 means: $pred1_mean")
println("Setup 2 means: $pred2_mean")
println("Setup 1 vars:  $pred1_var")
println("Setup 2 vars:  $pred2_var")



mean_diff = maximum(abs.(pred1_mean .- pred2_mean))
var_diff = maximum(abs.(pred1_var .- pred2_var))
println("Max mean difference: $mean_diff")
println("Max variance difference: $var_diff")
println("StandardGP equivalence: $(mean_diff < 1e-10 && var_diff < 1e-10)")
@assert (mean_diff < 1e-10 && var_diff < 1e-10)

pred1_unstd_mean = [(m * params1[2][1]) for m in pred1_mean]
pred1_unstd_var = [v .* (params1[2][1].^2) for v in pred1_var]
pred2_unstd_mean = [(m * params2[2][1]) for m in pred2_mean]
pred2_unstd_var = [v .* (params2[2][1].^2) for v in pred2_var]

mean_diff = maximum(abs.(pred1_unstd_mean .- pred2_unstd_mean))
var_diff = maximum(abs.(pred1_unstd_var .- pred2_unstd_var))

println("Unstandardized means comparison:")
println("Setup 1 unstd means: $pred1_unstd_mean")
println("Setup 2 unstd means: $pred2_unstd_mean")
println("Setup 1 unstd vars: $pred1_unstd_var")
println("Setup 2 unstd vars: $pred2_unstd_var")

@assert (mean_diff < 1e-10 && var_diff < 1e-10)

println("Okay, so both are equivalent")


# Test for GradientGP
println("\n=== Testing GradientGP ===")
y_test_gradient = f_val_grad.(x_test)

# Compute empirical mean for prior (only for function values, zero for gradients)
empirical_mean_grad = mean(hcat(y_test_gradient...)[1, :])
prior_mean_vector = [empirical_mean_grad; zeros(dim)]
println("Empirical mean for gradients: $prior_mean_vector")


println("Testing equivalence of gradConstMean(zeros) + center_scale and gradConstMean(prior) with no standardization")
# Setup 1: gradConstMean([0,0,0]) + center_scale
grad_kernel = gradKernel(kernel)
model1_grad = GradientGP(grad_kernel, dim+1, 1e-12)
bo1_grad = BOStruct(f_val_grad, ExpectedImprovement(0.01, minimum(hcat(y_test_gradient...)[1, :])), 
                    model1_grad, SqExponentialKernel(), ContinuousDomain([-5.0, -5.0], [5.0, 5.0]), 
                    x_test, y_test_gradient, 10, 0.0)

# Setup 2: gradConstMean([empirical_mean, 0, 0]) + scale_only
model2_grad = GradientGP(grad_kernel, dim+1, 1e-12, mean=gradConstMean(prior_mean_vector))
bo2_grad = BOStruct(f_val_grad, ExpectedImprovement(0.01, minimum(hcat(y_test_gradient...)[1, :])), 
                    model2_grad, SqExponentialKernel(), ContinuousDomain([-5.0, -5.0], [5.0, 5.0]), 
                    x_test, y_test_gradient, 10, 0.0)

# Apply standardizations
bo1_grad_std, params1_grad = standardize_problem(bo1_grad, choice="mean_only")

bo2_grad.model = update!(bo2_grad.model,x_test,y_test_gradient)

# Get gradient predictions from both setups
pred1_grad_mean = [posterior_grad_mean(bo1_grad_std.model, x) for x in x_pred]
pred1_grad_var = [posterior_grad_var(bo1_grad_std.model, x) for x in x_pred]

pred2_grad_mean = [posterior_grad_mean(bo2_grad.model, x) for x in x_pred]
pred2_grad_var = [posterior_grad_var(bo2_grad.model, x) for x in x_pred]



println("\nGradientGP Predictions (standardized):")
println("Setup 1 means: $pred1_grad_mean")
println("Setup 2 means: $pred2_grad_mean")
println("Setup 1 vars:  $pred1_grad_var")
println("Setup 2 vars:  $pred2_grad_var")

mean_diff_grad = maximum([maximum(abs.(m1 .- m2)) for (m1, m2) in zip(pred1_grad_mean, pred2_grad_mean)])
var_diff_grad = maximum([maximum(abs.(v1 .- v2)) for (v1, v2) in zip(pred1_grad_var, pred2_grad_var)])
println("Max mean difference: $mean_diff_grad")
println("Max variance difference: $var_diff_grad")
println("GradientGP equivalence: $(mean_diff_grad < 1e-10 && var_diff_grad < 1e-10)")

@assert (mean_diff_grad < 1e-10 && var_diff_grad < 1e-10)






println("Now testing gradConstMean(zeros) + center_scale should be equal to gradConstMean(prior) + scale_only")

# Setup 1: gradConstMean([0,0,0]) + center_scale
grad_kernel = gradKernel(kernel)
model1_grad = GradientGP(grad_kernel, dim+1, 1e-12)
bo1_grad = BOStruct(f_val_grad, ExpectedImprovement(0.01, minimum(hcat(y_test_gradient...)[1, :])), 
                    model1_grad, SqExponentialKernel(), ContinuousDomain([-5.0, -5.0], [5.0, 5.0]), 
                    x_test, y_test_gradient, 10, 0.0)

# Setup 2: gradConstMean([empirical_mean, 0, 0]) + scale_only
model2_grad = GradientGP(grad_kernel, dim+1, 1e-12, mean=gradConstMean(prior_mean_vector))
bo2_grad = BOStruct(f_val_grad, ExpectedImprovement(0.01, minimum(hcat(y_test_gradient...)[1, :])), 
                    model2_grad, SqExponentialKernel(), ContinuousDomain([-5.0, -5.0], [5.0, 5.0]), 
                    x_test, y_test_gradient, 10, 0.0)

# Apply standardizations
bo1_grad_std, params1_grad = standardize_problem(bo1_grad, choice="mean_scale")
bo2_grad_std, params2_grad = standardize_problem(bo2_grad, choice="scale_only")

println("Setup 1 (gradConstMean(zeros) + center_scale): μ=$(params1_grad[1]), σ=$(params1_grad[2])")
println("Setup 2 (gradConstMean(prior) + scale_only): μ=$(params2_grad[1]), σ=$(params2_grad[2])")

# Get gradient predictions from both setups
pred1_grad_mean = [posterior_grad_mean(bo1_grad_std.model, x) for x in x_pred]
pred1_grad_var = [posterior_grad_var(bo1_grad_std.model, x) for x in x_pred]

pred2_grad_mean = [posterior_grad_mean(bo2_grad_std.model, x) for x in x_pred]
pred2_grad_var = [posterior_grad_var(bo2_grad_std.model, x) for x in x_pred]



println("\nGradientGP Predictions (standardized):")
println("Setup 1 means: $pred1_grad_mean")
println("Setup 2 means: $pred2_grad_mean")
println("Setup 1 vars:  $pred1_grad_var")
println("Setup 2 vars:  $pred2_grad_var")

mean_diff_grad = maximum([maximum(abs.(m1 .- m2)) for (m1, m2) in zip(pred1_grad_mean, pred2_grad_mean)])
var_diff_grad = maximum([maximum(abs.(v1 .- v2)) for (v1, v2) in zip(pred1_grad_var, pred2_grad_var)])
@assert (mean_diff_grad < 1e-10 && var_diff_grad < 1e-10)
println("Max mean difference: $mean_diff_grad")
println("Max variance difference: $var_diff_grad")


# Un-standardize predictions to compare
pred1_grad_unstd_mean = [(m .* params1_grad[2]) .+ params1_grad[1] for m in pred1_grad_mean]
pred1_grad_unstd_var = [v .* (params1_grad[2].^2) for v in pred1_grad_var]

pred2_grad_unstd_mean = [(m .* params2_grad[2]) .+ params2_grad[1] for m in pred2_grad_mean]
pred2_grad_unstd_var = [v .* (params2_grad[2].^2) for v in pred2_grad_var]

println("\nGradientGP Predictions (un-standardized):")
println("Setup 1 means: $pred1_grad_unstd_mean")
println("Setup 2 means: $pred2_grad_unstd_mean")

mean_diff_grad = maximum([maximum(abs.(m1 .- m2)) for (m1, m2) in zip(pred1_grad_unstd_mean, pred2_grad_unstd_mean)])
var_diff_grad = maximum([maximum(abs.(v1 .- v2)) for (v1, v2) in zip(pred1_grad_unstd_var, pred2_grad_unstd_var)])
println("Max mean difference: $mean_diff_grad")
println("Max variance difference: $var_diff_grad")
println("GradientGP equivalence: $(mean_diff_grad < 1e-10 && var_diff_grad < 1e-10)")

@assert (mean_diff_grad < 1e-10 && var_diff_grad < 1e-10)
