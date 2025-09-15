#=
# AbstractBayesOpt Tutorial: Basic 1D Optimization
=#

# ## Setup
#
# Loading the necessary packages.
using AbstractBayesOpt
using AbstractGPs

using Plots
default(; legend=:outertopright, size=(700, 400))

using Random
Random.seed!(42)  # setting the seed for reproducibility of this notebook
#md nothing #hide

# ## Define the objective function
# We will optimise a simple 1D function: f(x) = (x[1]-2)^2 + sin(3*x[1])
f(x) = (x[1]-2)^2 + sin(3*x[1])

d = 1
domain = ContinuousDomain([0.0], [5.0])

# ## Initialize the surrogate model
# We'll use a standard Gaussian Process surrogate with a Matérn 5/2 kernel. We add a small jitter term for numerical stability of 1e-12.
noise_var = 1e-12
surrogate = StandardGP(Matern52Kernel(), noise_var)



# Generate uniform random samples x_train
n_train = 10
x_train = [domain.lower .+ (domain.upper .- domain.lower) .* rand(d) for _ in 1:n_train]

y_train = f.(x_train) 
y_train = map(x -> [x], y_train) # make y_train a vector of vectors, usual format for AbstractBayesOpt


# ## Choose an acquisition function
# We'll use the Expected Improvement acquisition function with an exploration parameter ξ = 0.0.
ξ = 0.0
acq = ExpectedImprovement(ξ, minimum(reduce(vcat,y_train)))


# ## Set up the Bayesian Optimization structure
bo_struct = BOStruct(f,
                     acq,
                     copy(surrogate),
                     domain,
                     x_train,
                     y_train,
                     50,  # number of iterations
                     0.0  # Actual noise level (0.0 for noiseless)
                     )

print_info(bo_struct)


@info "Starting Bayesian Optimization..."
result, acq_list, standard_params = AbstractBayesOpt.optimize(bo_struct)

# ## Results
# The optimization result is stored in `result`. We can print the best found input and its corresponding function value.
xs = reduce(vcat,result.xs)
ys = reduce(vcat,result.ys_non_std)

println("Optimal point: ",xs[argmin(ys)])
println("Optimal value: ",minimum(ys))