"""
This short example shows a 1D optimization of a function using the Bayesian Optimization framework.
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

# function branin(x)
#     x1 = x[1]
#     x2 = x[2]
#     b = 5.1 / (4*pi^2);
#     c = 5/pi;
#     r = 6;
#     a = 1;
#     s = 10;
#     t = 1 / (8*pi);
#     term1 = a * (x2 - b*x1^2 + c*x1 - r)^2;
#     term2 = s*(1-t)*cos(x1);
#     y = term1 + term2 + s;
# end

lower = [-10.0]
upper = [10.0]
domain = ContinuousDomain(lower, upper)

kernel = Matern32Kernel()
prior_gp = AbstractGPs.GP(kernel) # Creates GP(0,k)
model = StandardGP(prior_gp) # Instantiates the StandardGP (gives it the prior).

n_train = 10
x_train = sort(rand(Uniform(lower[1], upper[1]),n_train))
println(x_train)

σ² = 1e-10
y_train = f.(x_train) + σ².* randn(n_train);
# Conditioning: 
# We are conditionning the GP, returning GP|X,y where y can be noisy (but supposed fixed anyway)
model = update!(model, x_train, y_train, σ²)

acqf = ExpectedImprovement(1e-1, maximum(-y_train))

# This maximises the function
problem = BOProblem(
                    f,
                    domain,
                    model,
                    x_train,
                    y_train,
                    acqf,
                    100,
                    σ²
                    )

print_info(problem)

@info "Starting Bayesian Optimization..."
result = optimize(problem)
xs = vec(result.xs)
ys = vec(result.ys)

println("Optimal point: ",xs[argmin(ys)])
println("Optimal value: ",minimum(ys))

plot_domain = collect(lower[1]:0.01:upper[1])

scatter(
    xs,
    ys;
    xlim=(lower[1], upper[1]),
    xlabel="x",
    ylabel="y",
    title="posterior (default parameters)",
    label="Train Data",
)

post_mean, post_var = mean_and_var(result.gp.gpx(plot_domain))
post_var[post_var .< 0] .= 0

plot!(plot_domain, f.(plot_domain), label="target function")
plot!(plot_domain, post_mean; label="GP", ribbon=sqrt.(post_var))