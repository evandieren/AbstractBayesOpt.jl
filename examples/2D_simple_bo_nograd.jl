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

import Random
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
himmelblau(x::AbstractVector) = (x[1]^2 + x[2] -11)^2 + (x[1]+x[2]^2-7)^2

f(x) = himmelblau(x)

problem_dim = 2
lower = [-6,-6.0]
upper = [6.0,6.0]
domain = ContinuousDomain(lower, upper)
σ² = 1e-12

kernel_constructor = ApproxMatern52Kernel()

# Generate uniform random samples
n_train = 10
x_train = [lower .+ (upper .- lower) .* rand(problem_dim) for _ in 1:n_train]
y_train = f.(x_train) #+ sqrt(σ²).* randn(n_train)

y_train = map(x -> [x], y_train)

mean_y = mean(y_train)

# f̃(x) = (himmelblau(x)-y_mean)/y_std

kernel = 1 *(kernel_constructor ∘ ScaleTransform(1))
model = StandardGP(kernel,σ²) # Instantiates the StandardGP (gives it the prior).

# Conditioning: no need if true
# We are conditionning the GP, returning GP|X,y where y can be noisy (but supposed fixed)
# model = update!(model, x_train, y_train)

# Init of the acquisition function
ξ = 0.0
acqf = ExpectedImprovement(ξ, minimum(y_train)[1])

# This maximises the function
bo_struct = BOStruct(
                    f,
                    acqf,
                    model,
                    kernel_constructor,
                    domain,
                    copy(x_train),
                    copy(y_train),
                    100,
                    0.0
                    )

print_info(bo_struct)

@info "Starting Bayesian Optimization..."
@time result,acq_list, std_params = AbstractBayesOpt.optimize(bo_struct,scale_only=true)
xs = result.xs
ys = result.ys_non_std 
# ys = (reduce(vcat,result.ys).*y_std) .+ y_mean

println("Optimal point: ",xs[argmin(ys)])
println("Optimal value: ",minimum(ys))

running_min = accumulate(min, f.(xs))

Plots.plot(n_train:length(running_min),norm.(running_min)[n_train:end],yaxis=:log, title="Error w.r.t true minimum (2D BO)",
            xlabel="Function evaluations",ylabel=L"|| f(x^*_n) - f^* ||",
            label="BO",xlims=(1,length(running_min)))
Plots.vspan!([1,n_train]; color=:blue,alpha=0.2, label="")