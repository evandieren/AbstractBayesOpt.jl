
# JET analysis for StandardGP_1D tutorial
using JET
using AbstractBayesOpt
using AbstractGPs
using ForwardDiff
using Plots
using Random

# Define the objective function from the tutorial
f(x) = sum(x .- 2)^2 + sin(3*sum(x))

# Test JET on key functions from the tutorial
function test_bayesian_optimization()
    Random.seed!(42)
    d = 1
    domain = ContinuousDomain([0.0], [5.0])
    
    # Initialize surrogate
    noise_var = 1e-9
    surrogate = StandardGP(SqExponentialKernel(), noise_var)
    
    # Generate training data
    n_train = 5
    x_train = [domain.lower .+ (domain.upper .- domain.lower) .* rand(d) for _ in 1:n_train]
    y_train = f.(x_train)
    
    # Set up acquisition function
    ξ = 0.0
    acq = ExpectedImprovement(ξ, minimum(y_train))
    
    # Create BO structure
    bo_struct = BOStruct(
        f,
        acq,
        surrogate,
        domain,
        x_train,
        y_train,
        10,
        0.0,
    )
    
    # Run optimization
    result, acq_list, standard_params = AbstractBayesOpt.optimize(bo_struct; standardize=nothing)
    
    return result
end

function test_gradient_optimization()
    Random.seed!(42)
    d = 1
    domain = ContinuousDomain([0.0], [5.0])
    
    # Initialize gradient surrogate
    noise_var = 1e-9
    grad_surrogate = GradientGP(SqExponentialKernel(), d+1, noise_var)
    
    # Generate training data
    n_train = 5
    x_train = [domain.lower .+ (domain.upper .- domain.lower) .* rand(d) for _ in 1:n_train]
    
    ∇f(x) = ForwardDiff.gradient(f, x)
    f_val_grad(x) = [f(x); ∇f(x)]
    y_train_grad = f_val_grad.(x_train)
    
    # Set up acquisition function
    ξ = 0.0
    acq = ExpectedImprovement(ξ, minimum(reduce(vcat, y_train_grad)))
    
    # Create BO structure
    bo_struct_grad = BOStruct(
        f_val_grad,
        acq,
        grad_surrogate,
        domain,
        x_train,
        y_train_grad,
        10,
        0.0,
    )
    
    # Run optimization
    result_grad, acq_list_grad, standard_params_grad = AbstractBayesOpt.optimize(bo_struct_grad)
    
    return result_grad
end

println("JET analysis for Bayesian Optimization functions:")
println("\n1. Testing standard BO:")
@report_opt test_bayesian_optimization()

# println("\n2. Testing gradient-enhanced BO:")
# @report_opt test_gradient_optimization()

# println("\n3. Testing with detailed type analysis:")
# # Use @report_call macro for more detailed output
# @report_opt test_bayesian_optimization()

# println("\n4. Testing individual optimization call:")
# function test_single_optimize()
#     domain = ContinuousDomain([0.0], [5.0])
#     surrogate = StandardGP(SqExponentialKernel(), 1e-9)
#     x_test = [[1.0]]
#     y_test = [f([1.0])]
#     bo_struct = BOStruct(f, ExpectedImprovement(0.0, 0.0), surrogate, domain, x_test, y_test, 1, 0.0)
#     return AbstractBayesOpt.optimize(bo_struct; standardize=nothing)
# end

# @JET.report_call test_single_optimize()

# println("\n5. Testing objective function:")
# @JET.report_call f([1.0])

# println("\nJET analysis complete. Check output above for any reported issues.")