# Loading the necessary packages.
using AbstractBayesOpt
using AbstractGPs

using Random
Random.seed!(1234)  # setting the seed for reproducibility of this notebook
#md nothing #hide

# ## Define the objective function
# We will optimise a simple 1D function: f(x) = (x[1]-2)^2 + sin(3*x[1])
f(x) = sum(x .- 2)^2 + sin(3*sum(x))

min_f = -0.8494048256167165

d = 1
domain = ContinuousDomain([0.0], [5.0])

noise_var = 1e-2
surrogate = StandardGP(Matern52Kernel(), noise_var)

n_train = 5
x_train = [domain.lower .+ (domain.upper .- domain.lower) .* rand(d) for _ in 1:n_train]
y_train = f.(x_train)
y_train = map(x -> [x], y_train) # make y_train a vector of vectors, usual format for AbstractBayesOpt

# ## Choose an acquisition function
# We'll use the Expected Improvement acquisition function with an exploration parameter ξ = 0.0.
ξ = 0.0
acq = UpperConfidenceBound(0.0)# minimum(reduce(vcat, y_train)))

surrogate = update(surrogate, x_train, y_train)

@time x_cand = optimize_acquisition(acq, surrogate, domain)