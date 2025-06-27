"""
This short example shows a 1D optimization of a function using the Bayesian Optimization framework.

f : R → R
"""

using AbstractGPs
using KernelFunctions
using Plots
using Distributions
using BayesOpt
using Optim
using LinearAlgebra
using LaTeXStrings
import Random
Random.seed!(555)

# Objective Function
f(x) = sin(sum(x.+1)) + sin((10.0 / 3.0) * sum(x .+1))
min_f = −1.988699758534924
problem_dim = 1
lower = [-10.0]
upper = [10.0]
domain = ContinuousDomain(lower, upper)

σ² = 1e-6 # 1e-10

kernel = Matern52Kernel()
model = StandardGP(kernel, σ²) # Instantiates the StandardGP (gives it the prior).

# Generate uniform random samples
n_train = 10
x_train = [lower .+ (upper .- lower) .* rand(problem_dim) for _ in 1:n_train]

y_train = f.(x_train) #+ sqrt(σ²).* randn(n_train);
y_train = map(x -> [x], y_train)

# Initial log-parameters: log(lengthscale), log(magnitude)
initial_params = [log(1.0), log(1.0)]

# Optimize with BFGS
res = Optim.optimize(x -> nlml(x,kernel,x_train,y_train,σ²), initial_params, Optim.Newton())

# Extract optimized values
opt_params = Optim.minimizer(res)
ell_opt = exp(opt_params[1])
scale_opt = exp(opt_params[2])

println("Optimized lengthscale: ", ell_opt)
println("Optimized magnitude: ", scale_opt)

kernel = scale_opt *(kernel ∘ ScaleTransform(ell_opt))
model = StandardGP(kernel, σ²)

# Conditioning: 
model = update!(model, x_train, y_train)


# Init of the acquisition function
ξ = 1e-3
acqf = ExpectedImprovement(ξ, minimum(reduce(vcat,y_train)))
acqf2 = ExpectedImprovement(ξ, minimum(reduce(vcat,y_train)))

# This maximises the function
problem = BOProblem(
                    f,
                    domain,
                    model,
                    copy(x_train),
                    copy(y_train),
                    acqf,
                    90,
                    0.0
                    )

problem2 = BOProblem(
                    f,
                    domain,
                    model,
                    copy(x_train),
                    copy(y_train),
                    acqf2,
                    90,
                    0.0
                    )

print_info(problem)

@info "Starting Bayesian Optimization..."
result, acq_list = BayesOpt.optimize(problem)
xs = reduce(vcat,result.xs)
ys = reduce(vcat,result.ys)

println("Optimal point: ",xs[argmin(ys)])
println("Optimal value: ",minimum(ys))


@info "Starting Bayesian Optimization..."
result2,acq_list2 = BayesOpt.optimize(problem2)
xs2 = reduce(vcat,result2.xs)
ys2 = reduce(vcat,result2.ys)

println("Optimal point: ",xs2[argmin(ys2)])
println("Optimal value: ",minimum(ys2))


plot(max.(acq_list,1e-15),yaxis=:log)
plot!(max.(acq_list2,1e-15))