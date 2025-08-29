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
using LaTeXStrings
import Random
using ForwardDiff
Random.seed!(555)

# Objective Function
f(x) = sin(sum(x.+1)) + sin((10.0 / 3.0) * sum(x .+1))
min_f = −1.988699758534925
problem_dim = 1
lower = [-10.0]
upper = [10.0]
domain = ContinuousDomain(lower, upper)

σ² = 1e-12 # 1e-10

# Generate uniform random samples
n_train = 10
x_train = [lower .+ (upper .- lower) .* rand(problem_dim) for _ in 1:n_train]

y_train = f.(x_train) #+ sqrt(σ²).* randn(n_train);
y_train = map(x -> [x], y_train)

kernel_constructor = ApproxMatern52Kernel()
kernel = 1 *(kernel_constructor ∘ ScaleTransform(1)) # needed because I need to do MLE
model = StandardGP(kernel, σ²,mean=mean(y_train)[1])

# # Conditioning:  no need if standardize == true
# model = update!(model, x_train, y_train)


# Init of the acquisition function
ξ = 0.0
acqf = ExpectedImprovement(ξ, minimum(reduce(vcat,y_train)))

# This maximises the function
problem = BOStruct(
                    f,
                    acqf,
                    model,
                    kernel_constructor,
                    domain,
                    copy(x_train),
                    copy(y_train),
                    210,
                    0.0
                    )

print_info(problem)

obj = p -> nlml(model, p, kernel_constructor, x_train, reduce(vcat,y_train),mean=mean(y_train)[1])

obj([0.0,0.0])

ForwardDiff.gradient(obj,[0.0,0.0])