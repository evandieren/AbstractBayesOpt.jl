"""
This short example shows a 2D optimization of a function using the Bayesian Optimization framework.

f : R² → R
"""

using AbstractGPs
using KernelFunctions
using Plots
using Distributions
using LaTeXStrings
using LinearAlgebra
using AbstractBayesOpt

using Random: Random
Random.seed!(555)

# Objective Function 
# Branin
function branin(x::AbstractVector)
    x1 = x[1]
    x2 = x[2]
    b = 5.1 / (4*pi^2);
    c = 5/pi;
    r = 6;
    a = 1;
    s = 10;
    t = 1 / (8*pi);
    term1 = a * (x2 - b*x1^2 + c*x1 - r)^2;
    term2 = s*(1-t)*cos(x1);
    y = term1 + term2 + s;
end

# Rosenbrock
rosenbrock(x::AbstractVector) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

# Himmelblau
himmelblau(x::AbstractVector) = (x[1]^2 + x[2] - 11)^2 + (x[1]+x[2]^2-7)^2

f(x) = himmelblau(x)

problem_dim = 2
lower = [-6, -6.0]
upper = [6.0, 6.0]
domain = ContinuousDomain(lower, upper)
σ² = 1e-12

kernel = ApproxMatern52Kernel()

# Generate uniform random samples
n_train = 10
x_train = [lower .+ (upper .- lower) .* rand(problem_dim) for _ in 1:n_train]
y_train = f.(x_train) #+ sqrt(σ²).* randn(n_train)

y_train = map(x -> [x], y_train)

model = StandardGP(kernel, σ²) # Instantiates the StandardGP (gives it the prior).

# Conditioning: no need if true
# We are conditionning the GP, returning GP|X,y where y can be noisy (but supposed fixed)
# model = update!(model, x_train, y_train)

# Init of the acquisition function
ξ = 0.0
acqf = ExpectedImprovement(ξ, minimum(y_train)[1])

# This maximises the function
bo_struct = BOStruct(f, acqf, model, domain, copy(x_train), copy(y_train), 50, 0.0)

print_info(bo_struct)

@info "Starting Bayesian Optimization..."

choice = "mean_scale"

@time result, acq_list, std_params = AbstractBayesOpt.optimize(
    bo_struct, standardize=choice
)
xs = result.xs
ys = result.ys_non_std
# ys = (reduce(vcat,result.ys).*y_std) .+ y_mean
println("Optimal point: ", xs[argmin(ys)])
println("Optimal value: ", minimum(ys))

# xs_nothing = copy(xs)
# acq_nothing = copy(acq_list)

# xs_scale_only = copy(xs)
# acq_scale_only = copy(acq_list .* std_params[2][1])

# xs_mean_scale = copy(xs)
# acq_mean_scale = copy(acq_list.*std_params[2][1])

# xs_mean_only = copy(xs)
# acq_mean_only = copy(acq_list)

running_min = accumulate(min, f.(xs))

Plots.plot(
    n_train:length(running_min),
    norm.(running_min)[n_train:end];
    yaxis=:log,
    title="Error w.r.t true minimum (2D BO)",
    xlabel="Function evaluations",
    ylabel=L"|| f(x^*_n) - f^* ||",
    label="BO",
    xlims=(1, length(running_min)),
)
Plots.vspan!([1, n_train]; color=:blue, alpha=0.2, label="")

# Plots.plot(n_train:length(acq_nothing), acq_nothing[n_train:end] .+ eps(), label="standardize = nothing", xlabel="Iteration",
#         ylabel="Acquisition value", title="Acquisition value over iterations (1D BO)", yaxis=:log)
# Plots.plot!(n_train:length(acq_scale_only), acq_scale_only[n_train:end] .+ eps(), label="standardize = scale_only",ls=:dash)

# Plots.plot(n_train:length(acq_mean_only), acq_mean_only[n_train:end] .+ eps(), label="standardize = mean_only",ls=:dot,yaxis=:log)
# Plots.plot!(n_train:length(acq_mean_scale), acq_mean_scale[n_train:end] .+ eps(), label="standardize = mean_scale",ls=:dashdot)

# Plots.plot(n_train:length(xs_nothing), f.(xs_nothing)[n_train:end], label="standardize = nothing", xlabel="Iteration", ylabel="f(x)", title="Value of f at sampled points (1D BO)")
# Plots.plot!(n_train:length(xs_scale_only), f.(xs_scale_only)[n_train:end], label="standardize = scale_only",ls=:dash)

# Plots.plot(n_train:length(xs_mean_only), f.(xs_mean_only)[n_train:end], label="standardize = mean_only",ls=:dot)
# Plots.plot!(n_train:length(xs_mean_scale), f.(xs_mean_scale)[n_train:end], label="standardize = mean_scale",ls=:dashdot)
