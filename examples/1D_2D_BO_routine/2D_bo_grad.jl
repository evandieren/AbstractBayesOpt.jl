"""
This short example shows a 2D optimization of a function using the Bayesian Optimization framework.

f : R² → R
"""

using AbstractGPs
using KernelFunctions
using Plots
using Distributions
using ForwardDiff
using LaTeXStrings
using AbstractBayesOpt

import Random
Random.seed!(123456)

# Objective Function 

# Rosenbrock
rosenbrock(x::AbstractVector) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2

# Himmelblau
himmelblau(x::AbstractVector) = (x[1]^2 + x[2] -11)^2 + (x[1]+x[2]^2-7)^2

f(x) = himmelblau(x)

∇f(x) = ForwardDiff.gradient(f, x)

f_val_grad(x) = [f(x); ∇f(x)]

d = 2
lower = [-6,-6.0]
upper = [6.0,6.0]
domain = ContinuousDomain(lower, upper)
σ² = 1e-12


# Generate uniform random samples
n_train = 10
x_train = [lower .+ (upper .- lower) .* rand(d) for _ in 1:n_train]


val_grad = f_val_grad.(x_train)
# Create flattened output
y_train = [val_grad[i] for i = eachindex(val_grad)]



kernel = ApproxMatern52Kernel()
model = GradientGP(kernel,d+1,σ²)

# Init of the acquisition function
ξ = 0.0
acqf = ExpectedImprovement(ξ, minimum(hcat(y_train...)[1,:]))


# This maximises the function
bo_struct = BOStruct(
                    f_val_grad, # because we probe both the function value and its gradients.
                    acqf,
                    model,
                    domain,
                    copy(x_train),
                    copy(y_train),
                    30,
                    σ²
                    )

print_info(bo_struct)

@info "Starting Bayesian Optimization..."
result, acq_list, standard_params = AbstractBayesOpt.optimize(bo_struct)
xs = result.xs
ys = result.ys_non_std

println("Optimal point: ",xs[argmin(ys)])
println("Optimal value: ",minimum(ys))

running_min = accumulate(min, f.(xs))

running_min = collect(Iterators.flatten(fill(x, 3) for x in (running_min)))

p = Plots.plot((3*n_train):length(running_min),running_min[3*n_train:end],yaxis=:log, title="Error w.r.t true minimum (2D GradBO)",
            xlabel="Function evaluations",ylabel=L"|| f(x^*_n) - f^* ||",
            label="GradBO",xlims=(1,length(running_min)))
Plots.vspan!([1,3*n_train]; color=:blue,alpha=0.2, label="")
Plots.display(p)


x_train_prepped = prep_input(result.model,x_train)
post_mean, post_var = unstandardized_mean_and_var(result.model,x_train_prepped, standard_params)
println("Posterior at training points:")
for i in eachindex(x_train)
    println("x: ", x_train[i], " f: ", post_mean[i,1], " ± ", sqrt(post_var[i,1] .+ 5e-10), " | ∇f: ", post_mean[i,2:end], " ± ", sqrt.(post_var[i,2:end] .+ 5e-10))
end 