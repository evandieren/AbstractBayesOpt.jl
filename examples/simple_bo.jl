using AbstractGPs
using KernelFunctions

using BayesOpt

import Random
Random.seed!(1234)

# Objective Function
f(x) = sin(sum(abs, x)) + sum(abs2, x) * cos(sum(abs, x))

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

lower = [-5.0]
upper = [10.0]
domain = ContinuousDomain(lower, upper)

kernel = Matern32Kernel()
prior_gp = AbstractGPs.GP(kernel)
model = StandardGP(prior_gp)
x_train = [[0.0], [1.5], [2.5], [3.5]]
y_train = f.(x_train)
model = update!(model, x_train, y_train, 0.0)

println(x_train)
println(y_train)

acqf = ExpectedImprovement(0.1, minimum(y_train))


test = acqf(model,x_train[1])
println(test)

problem = BOProblem(
    f,
    domain,
    model,
    acqf,
    30,
    0.0
)

print_info(problem)

#@info "Starting Bayesian Optimization..."
#result = optimize(problem)

#println(result.xs)

#println(result.ys)

#println("Optimal point",xs[argmin(ys)])
#println("Optimal value",min(ys))