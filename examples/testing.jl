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
min_f = −1.988699758534925
problem_dim = 1
lower = [-10.0]
upper = [10.0]
domain = ContinuousDomain(lower, upper)

σ² = 1e-1 # 1e-10

# Generate uniform random samples
n_train = 10
x_train = [lower .+ (upper .- lower) .* rand(problem_dim) for _ in 1:n_train]

y_train = f.(x_train) #+ sqrt(σ²).* randn(n_train);
y_train = map(x -> [x], y_train)

kernel_constructor = Matern52Kernel()
kernel = 1 *(kernel_constructor ∘ ScaleTransform(1)) # needed because I need to do MLE
model = StandardGP(kernel, σ²)

# # Conditioning:  no need if standardize == true
# model = update!(model, x_train, y_train)


# Init of the acquisition function
ξ = 0.0
acqf = ExpectedImprovement(ξ, minimum(reduce(vcat,y_train)))

# This maximises the function
problem = BOProblem(
                    f,
                    domain,
                    model,
                    kernel_constructor,
                    copy(x_train),
                    copy(y_train),
                    acqf,
                    210,
                    0.0
                    )

print_info(problem)


old_params = log.([get_lengthscale(problem.gp)[1],get_scale(problem.gp)[1]])

obj = p -> nlml(model, p, kernel, x_train, reduce(vcat,y_train), model.noise_var,mean=ZeroMean())

nlml(model,old_params,kernel,x_train,reduce(vcat,y_train),model.noise_var)

new_model = optimize_hyperparameters(model, x_train, y_train, kernel_constructor,old_params,true,num_restarts=25)

obj_lengthscale = p -> nlml(new_model,[p,0.0],kernel,x_train,reduce(vcat,y_train), model.noise_var,mean=ZeroMean())

ℓ_logvals = log(0.7):0.01:log(1.40) 

nlml_ℓ = obj_lengthscale.(ℓ_logvals)

ℓ_opt = get_lengthscale(new_model)
nlml_opt = nlml(new_model,[log(ℓ_opt[1]),0.0],kernel,x_train,reduce(vcat,y_train),model.noise_var)

plot(ℓ_logvals,nlml_ℓ)
scatter!(log.(ℓ_opt),[nlml_opt])

#nlml(model,[log(1.1527522390407372),0.0],kernel,x_train,reduce(vcat,y_train),model.noise_var)
# using ImageMagick, FileIO

# # Load frames into an array
# frames = [load("./examples/plots/1D_iter_$(i).png") for i in 0:49]

# # Save as GIF (set delay between frames in seconds)
# save("my_animation_1D.gif", cat(frames...; dims=3), fps=0.5)