"""
Simple debug test for standardization equivalence
"""

using AbstractBayesOpt
using AbstractGPs
using KernelFunctions
using Statistics
using Random

Random.seed!(42)

println("=== Debug Test for StandardGP ===")

# Simple 1D test
x_test = [[-1.0], [0.0], [1.0]]
y_test = [[2.0], [3.0], [4.0]]  # Mean = 3.0, Std ≈ 1.0

empirical_mean = mean(reduce(vcat, y_test))
println("Empirical mean: $empirical_mean")

# Setup 1: ZeroMean + center_scale
kernel = 1*(SqExponentialKernel() ∘ ScaleTransform(1))
model1 = StandardGP(kernel, 1e-6)
bo1 = BOStruct(x -> [x[1]^2], ExpectedImprovement(0.01, 2.0), 
               model1, SqExponentialKernel(), ContinuousDomain([-2.0], [2.0]), 
               x_test, y_test, 10, 0.0)

# Setup 2: ConstMean(empirical_mean) + scale_only  
model2 = StandardGP(kernel, 1e-6, mean=ConstMean(empirical_mean))
bo2 = BOStruct(x -> [x[1]^2], ExpectedImprovement(0.01, 2.0), 
               model2, SqExponentialKernel(), ContinuousDomain([-2.0], [2.0]), 
               x_test, y_test, 10, 0.0)

println("Before standardization:")
println("Model 1 mean: $(bo1.model.gp.mean)")
println("Model 2 mean: $(bo2.model.gp.mean)")

# Apply standardizations
println("\nApplying standardizations...")
bo1_std, params1 = standardize_problem(bo1, choice="center_scale")
bo2_std, params2 = standardize_problem(bo2, choice="scale_only")

println("After standardization:")
println("Model 1 mean: $(bo1_std.model.gp.mean)")
println("Model 2 mean: $(bo2_std.model.gp.mean)")

println("Setup 1 (ZeroMean + center_scale): μ=$(params1[1]), σ=$(params1[2])")
println("Setup 2 (ConstMean + scale_only): μ=$(params2[1]), σ=$(params2[2])")

# Test a prediction point
x_pred = [0.5]
pred1_mean = posterior_mean(bo1_std.model, x_pred)
pred2_mean = posterior_mean(bo2_std.model, x_pred)

println("Standardized predictions at x=$x_pred:")
println("Setup 1 mean: $pred1_mean")
println("Setup 2 mean: $pred2_mean")

# Un-standardize
pred1_unstd = (pred1_mean * params1[2][1]) + params1[1][1]
pred2_unstd = (pred2_mean * params2[2][1]) + params2[1][1]

println("Un-standardized predictions:")
println("Setup 1: $pred1_unstd")
println("Setup 2: $pred2_unstd")
println("Difference: $(abs(pred1_unstd - pred2_unstd))")



# Test GradientGP equivalence
f(x) = x[1]^2
∇f(x) = [2 * x[1]]
f_val_grad(x) = [f(x); ∇f(x)]

y_test_gradient = f_val_grad.(x_test)
empirical_mean_grad = mean(hcat(y_test_gradient...)[1, :])
prior_mean_vector = [empirical_mean_grad; zeros(1)]

grad_kernel = gradKernel(kernel)
model1_grad = GradientGP(grad_kernel, 2, 1e-6)
bo1_grad = BOStruct(f_val_grad, ExpectedImprovement(0.01, minimum(hcat(y_test_gradient...)[1, :])), 
                    model1_grad, SqExponentialKernel(), ContinuousDomain([-2.0], [2.0]), 
                    x_test, y_test_gradient, 10, 0.0)

model2_grad = GradientGP(grad_kernel, 2, 1e-6, mean=gradConstMean(prior_mean_vector))
bo2_grad = BOStruct(f_val_grad, ExpectedImprovement(0.01, minimum(hcat(y_test_gradient...)[1, :])), 
                    model2_grad, SqExponentialKernel(), ContinuousDomain([-2.0], [2.0]), 
                    x_test, y_test_gradient, 10, 0.0)

bo1_grad_std, params1_grad = standardize_problem(bo1_grad, choice="center_scale")
bo2_grad_std, params2_grad = standardize_problem(bo2_grad, choice="scale_only")

pred1_grad_mean = posterior_grad_mean(bo1_grad_std.model, x_pred)
pred2_grad_mean = posterior_grad_mean(bo2_grad_std.model, x_pred)

pred1_grad_unstd = (pred1_grad_mean .* params1_grad[2]) .+ params1_grad[1]
pred2_grad_unstd = (pred2_grad_mean .* params2_grad[2]) .+ params2_grad[1]

isapprox(pred1_grad_unstd, pred2_grad_unstd; atol=1e-10)