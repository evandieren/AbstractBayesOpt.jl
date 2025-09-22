# Loading the necessary packages.
using AbstractBayesOpt
using AbstractGPs
using ForwardDiff

using Random
Random.seed!(42)  # setting the seed for reproducibility of this notebook
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

y_train_prepped = reduce(vcat, y_train)
obj = p -> AbstractBayesOpt.nlml(surrogate, p, x_train, y_train_prepped)

obj([0.0; 0.0])

ForwardDiff.gradient(obj, [0.0; 0.0])

using Zygote
grads = Zygote.gradient(p -> nlml(surrogate, p, x_train, y_train_prepped), [0.0, 0.0]) # This is fine

function debug_gradient(obj, x0)
    ε = sqrt(eps(Float64))
    n = length(x0)
    grad = zeros(n)

    for i in 1:n
        x_plus = copy(x0)
        x_minus = copy(x0)
        x_plus[i] += ε
        x_minus[i] -= ε

        f_plus = obj(x_plus)
        f_minus = obj(x_minus)

        println("Param $i: f(x+ε) = $f_plus, f(x-ε) = $f_minus")

        grad[i] = (f_plus - f_minus) / (2ε)
    end

    return grad
end

debug_gradient(obj, [0.0, 0.0]) # This matches Zygote

optimize_hyperparameters(surrogate,
    x_train,
    y_train,
    [0.0; 0.0],
    true,
    domain = domain)

# ## Choose an acquisition function
# We'll use the Expected Improvement acquisition function with an exploration parameter ξ = 0.0.
ξ = 0.0
acq = ExpectedImprovement(ξ, minimum(reduce(vcat, y_train)))

# ## Set up the Bayesian Optimization structure
# We use BOStruct to bundle all components needed for the optimization. Here, we set the number of iterations to 5 and the actual noise level to 0.0 (since our function is noiseless).
# We then run the optimize function to perform the Bayesian Optimization.
bo_struct = BOStruct(
    f,
    acq,
    surrogate,
    domain,
    x_train,
    y_train,
    10,  # number of iterations
    0.0  # Actual noise level (0.0 for noiseless)
)

print_info(bo_struct)
