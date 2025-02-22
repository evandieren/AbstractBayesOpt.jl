using AbstractGPs
using KernelFunctions

using BayesOpt

import Random
Random.seed!(1234)

# 1. Objective Function (Branin)
function branin(x)
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

lower = [-5.0, 0.0]
upper = [10.0, 15.0]
domain = ContinuousDomain(lower, upper)

kernel = 0.5 * Matern32Kernel() ∘ ScaleTransform(0.5) + 0.5 * SqExponentialKernel()
prior_gp = AbstractGPs.GP(kernel)
X_init = Matrix{Float64}(undef, 2, 0)  # Empty initial inputs (2D)
posterior_fx = prior_gp(X_init)         # Initial FiniteGP
model = StandardGP(prior_gp, posterior_fx)

acqf = ExpectedImprovement(0.1, Inf)

problem = BOProblem(
    branin,
    domain,
    model,
    acqf,
    max_iter=30,
    noise=0.0
)

@info "Starting Bayesian Optimization..."
result = optimize!(problem)

println(result.xs)

println(result.ys)

println("Optimal point",xs[argmin(ys)])
println("Optimal value",min(ys))