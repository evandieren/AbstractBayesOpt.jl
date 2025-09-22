#=
# AbstractBayesOpt Tutorial: Basic 2D Optimization
=#

# ## Setup
#
# Loading the necessary packages.
using AbstractBayesOpt
using AbstractGPs
using ForwardDiff

using Plots
default(; legend=:outertopright, size=(700, 400))

using Random
Random.seed!(42)  # setting the seed for reproducibility of this notebook
#md nothing #hide

# ## Define the objective function
d = 2
f(x) = (x[1]^2 + x[2] - 11)^2 + (x[1]+x[2]^2-7)^2

min_f = 0.0

domain = ContinuousDomain([-6.0, -6.0], [6.0, 6.0]) #hide

X = domain.lower[1]:0.01:domain.upper[1] #hide
Y = domain.lower[2]:0.01:domain.upper[2] #hide

p1 = contour(
    X,
    Y,
    (x, y) -> f([x, y]);
    fill=true,
    levels=50,
    c=:viridis,
    title="Target function : Himmelblau",
    xlabel="x₁",
    ylabel="x₂",
) # hide

x_mins = [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]] #hide

# Scatter them on the contour plot
scatter!(
    [p[1] for p in x_mins],
    [p[2] for p in x_mins];
    label="Minima",
    color=:red,
    markersize=5,
    legend=:bottomright,
)

# ## Initialize the surrogate model
# We'll use a standard Gaussian Process surrogate with a Matérn 5/2 kernel. We add a small jitter term for numerical stability of 1e-12.
noise_var = 1e-9
surrogate = StandardGP(SqExponentialKernel(), noise_var)

# Generate uniform random samples x_train
n_train = 5
x_train = [domain.lower .+ (domain.upper .- domain.lower) .* rand(d) for _ in 1:n_train]

y_train = f.(x_train)

# ## Choose an acquisition function
# We'll use the Expected Improvement acquisition function with an exploration parameter ξ = 0.0.
ξ = 0.0
acq = ExpectedImprovement(ξ, minimum(y_train))

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
    50,  # number of iterations
    0.0,  # Actual noise level (0.0 for noiseless)
)

print_info(bo_struct)

surrogate = update(surrogate, x_train, y_train)

acq(surrogate, [[0.0, 0.0]])

@info "Starting Bayesian Optimization..."
result, acq_list, standard_params = AbstractBayesOpt.optimize(
    bo_struct; standardize=nothing
)

# ## Results
# The optimization result is stored in `result`. We can print the best found input and its corresponding function value.
xs = result.xs
ys = result.ys_non_std

println("Optimal point: ", xs[argmin(ys)])
println("Optimal value: ", minimum(ys))

# ## Plotting of running minimum over iterations
# The running minimum is the best function value found up to each iteration.
running_min = accumulate(min, f.(xs))

p = Plots.plot(
    n_train:length(running_min),
    running_min[n_train:end] .- min_f;
    yaxis=:log,
    title="Error w.r.t true minimum (2D BO)",
    xlabel="Function evaluations",
    label="BO",
    xlims=(1, length(running_min)),
)
Plots.vspan!([1, n_train]; color=:blue, alpha=0.2, label="")

# ## Gradient-enhanced GPs
# Now, let's see how to use gradient information to improve the optimization. We'll use the same function but now also provide its gradient.
# We define a new surrogate model that can handle gradient information, specifically a `GradientGP`.
grad_surrogate = GradientGP(SqExponentialKernel(), d + 1, noise_var)

ξ = 0.0
acq = ExpectedImprovement(ξ, minimum(y_train))

∇f(x) = ForwardDiff.gradient(f, x)
f_val_grad(x) = [f(x); ∇f(x)]

# Generate value and gradients at random samples
y_train_grad = f_val_grad.(x_train)

# Set up the Bayesian Optimization structure
bo_struct_grad = BOStruct(
    f_val_grad,
    acq,
    grad_surrogate,
    domain,
    x_train,
    y_train_grad,
    50,  # number of iterations
    0.0,  # Actual noise level (0.0 for noiseless)
)

print_info(bo_struct_grad)

@info "Starting Bayesian Optimization..."
result_grad, acq_list_grad, standard_params_grad = AbstractBayesOpt.optimize(bo_struct_grad)

# ## Results
# The optimization result is stored in `result`. We can print the best found input and its corresponding function value.
xs_grad = result_grad.xs
ys_grad = first.(result_grad.ys_non_std)

println("Optimal point (GradBO): ", xs_grad[argmin(ys_grad)])
println("Optimal value (GradBO): ", minimum(ys_grad))

# ## Plotting of running minimum over iterations
# The running minimum is the best function value found up to each iteration.
running_min_grad = accumulate(min, f.(xs_grad))
# Double function evaluations due to gradients
running_min_grad = collect(Iterators.flatten(fill(x, 2) for x in (running_min_grad)))

p = Plots.plot(
    (2 * n_train):length(running_min_grad),
    running_min_grad[(2 * n_train):end] .- min_f;
    yaxis=:log,
    title="Error w.r.t true minimum (2D GradBO)",
    xlabel="Function evaluations",
    label="gradBO",
    xlims=(1, length(running_min_grad)),
)
Plots.vspan!([1, 2 * n_train]; color=:blue, alpha=0.2, label="")

# ## Plotting the surrogate model
# We can visualize the surrogate model's mean and uncertainty along with the true function and the evaluated

plot_domain = collect(domain.lower[1]:0.01:domain.upper[1])

# TODO: finish this plotting part to show the surrogate mean with the points etc.
# TODO: Do a slice of the 2D function and plot the surrogate on that slice too.
