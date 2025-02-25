"""
This short example shows a 1D optimization of a function using the Bayesian Optimization framework.

f : R² → R
"""

using AbstractGPs
using KernelFunctions
using Plots
using Distributions

using BayesOpt

import Random
Random.seed!(1234)

# Objective Function
f(x) = sin(sum(x.+1)) + sin((10.0 / 3.0) * sum(x .+1))

problem_dim = 1
lower = [-10.0]
upper = [10.0]
domain = ContinuousDomain(lower, upper)

kernel = Matern32Kernel()
prior_gp = AbstractGPs.GP(kernel) # Creates GP(0,k) for the prior
model = StandardGP(prior_gp) # Instantiates the StandardGP (gives it the prior).

# Generate uniform random samples
n_train = 10
x_train = [lower .+ (upper .- lower) .* rand(problem_dim) for _ in 1:n_train]

println(x_train)

σ² = 1e-3 # 1e-10
y_train = f.(x_train) + σ².* randn(n_train);
println(y_train)
# Conditioning: 
# We are conditionning the GP, returning GP|X,y where y can be noisy (but supposed fixed anyway)
model = update!(model, x_train, y_train, σ²)

# Init of the acquisition function
ξ = 1e-1
acqf = ExpectedImprovement(ξ, minimum(y_train))

# This maximises the function
problem = BOProblem(
                    f,
                    domain,
                    model,
                    x_train,
                    y_train,
                    acqf,
                    30,
                    σ²
                    )

print_info(problem)

@info "Starting Bayesian Optimization..."
result = optimize(problem)
xs = reduce(vcat,result.xs)
ys = result.ys

println("Optimal point: ",xs[argmin(ys)])
println("Optimal value: ",minimum(ys))

plot_domain = collect(lower[1]:0.01:upper[1])

post_mean, post_var = mean_and_var(result.gp.gpx(plot_domain))
post_var[post_var .< 0] .= 0

plot(plot_domain, f.(plot_domain),
        label="target function",
        xlim=(lower[1], upper[1]),
        xlabel="x",
        ylabel="y",
        title="BayesOpt, EI ξ=$(ξ), σ²=$(σ²)",
        legend=:outertopright)
plot!(plot_domain, post_mean; label="GP", ribbon=sqrt.(post_var),color="green")
scatter!(
    xs[1:n_train],
    ys[1:n_train];
    label="Train Data"
)
scatter!(
    xs[n_train+1:end],
    ys[n_train+1:end];
    label="Candidates"
)
scatter!(
    [xs[argmin(ys)]],
    [minimum(ys)];
    label="Best candidate"
)