"""
This short example shows a 1D optimization of a function using the Bayesian Optimization framework.

f : R → R
"""

using AbstractGPs
using KernelFunctions
using Plots
using Distributions
using AbstractBayesOpt
using Optim
using LinearAlgebra
# using LaTeXStrings
using Random: Random
Random.seed!(555)

# Objective Function
f(x) = sin(sum(x .+ 1)) + sin((10.0 / 3.0) * sum(x .+ 1))
min_f = −1.988699758534925
dim = 1
lower = [-10.0]
upper = [10.0]
domain = ContinuousDomain(lower, upper)

σ² = 1e-12 # 1e-10

# Generate uniform random samples
n_train = 10
x_train = [lower .+ (upper .- lower) .* rand(dim) for _ in 1:n_train]

y_train = f.(x_train) #+ sqrt(σ²).* randn(n_train);
y_train = map(x -> [x], y_train)

#prior_mean = ConstMean(mean(reduce(vcat,y_train))) # set prior mean to empirical mean of data

kernel_constructor = ApproxMatern52Kernel()
kernel = 1 * (kernel_constructor ∘ ScaleTransform(1)) # needed because I need to do MLE
model = StandardGP(kernel, σ²)#, mean=prior_mean)

# # Conditioning:  no need if standardize == true
# model = update!(model, x_train, y_train)

# Init of the acquisition function
ξ = 0.0
acqf = ExpectedImprovement(ξ, minimum(reduce(vcat, y_train)))

# This maximises the function
bo_struct = BOStruct(f, acqf, copy(model), domain, x_train, y_train, 100, 0.0)

print_info(bo_struct)

@info "Starting Bayesian Optimization..."

choice = "mean_only"

result, acq_list, standard_params = AbstractBayesOpt.optimize(bo_struct; standardize=choice)
xs = reduce(vcat, result.xs)
ys = reduce(vcat, result.ys_non_std)

# xs_nothing = copy(xs)
# acq_nothing = copy(acq_list)

# xs_scale_only = copy(xs)
# acq_scale_only = copy(acq_list .* standard_params[2][1])

# xs_mean_scale = copy(xs)
# acq_mean_scale = copy(acq_list.*standard_params[2][1])

# xs_mean_only = copy(xs)
# acq_mean_only = copy(acq_list)

println("Optimal point: ", xs[argmin(ys)])
println("Optimal value: ", minimum(ys))

running_min = accumulate(min, f.(xs))

p = Plots.plot(
    n_train:length(running_min),
    running_min[n_train:end] .- min_f;
    yaxis=:log,
    title="Error w.r.t true minimum (1D BO)",
    xlabel="Function evaluations",
    ylabel=L"|| f(x^*_n) - f^* ||",
    label="BO",
    xlims=(1, length(running_min)),
)
Plots.vspan!([1, n_train]; color=:blue, alpha=0.2, label="")
Plots.display(p)

# Plots.plot(n_train:length(acq_nothing), acq_nothing[n_train:end] .+ eps(), label="standardize = nothing", xlabel="Iteration",
#         ylabel="Acquisition value", title="Acquisition value over iterations (1D BO)", yaxis=:log)
# Plots.plot!(n_train:length(acq_scale_only), acq_scale_only[n_train:end] .+ eps(), label="standardize = scale_only",ls=:dash)

# Plots.plot(n_train:length(acq_mean_only), acq_mean_only[n_train:end] .+ eps(), label="standardize = mean_only",ls=:dot,yaxis=:log)
# Plots.plot!(n_train:length(acq_mean_scale), acq_mean_scale[n_train:end] .+ eps(), label="standardize = mean_scale",ls=:dashdot)

# Plots.plot(n_train:length(xs_nothing), f.(xs_nothing)[n_train:end], label="standardize = nothing", xlabel="Iteration", ylabel="f(x)", title="Value of f at sampled points (1D BO)")
# Plots.plot!(n_train:length(xs_scale_only), f.(xs_scale_only)[n_train:end], label="standardize = scale_only",ls=:dash)

# Plots.plot(n_train:length(xs_mean_only), f.(xs_mean_only)[n_train:end], label="standardize = mean_only",ls=:dot)
# Plots.plot!(n_train:length(xs_mean_scale), f.(xs_mean_scale)[n_train:end], label="standardize = mean_scale",ls=:dashdot)
