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
