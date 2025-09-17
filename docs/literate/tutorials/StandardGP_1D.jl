#=
# AbstractBayesOpt Tutorial: Basic 1D Optimization
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
# We will optimise a simple 1D function: ``f(x) = (x-2)^2 + \sin(3*x)``
f(x) = sum(x .- 2)^2 + sin(3*sum(x))

min_f = -0.8494048256167165

d = 1


domain = ContinuousDomain([0.0], [5.0]) #hide

plot_domain = domain.lower[1]:0.01:domain.upper[1] #hide
ys = f.(plot_domain) #hide

plot(                                           #hide
    plot_domain,                                #hide
    ys;                                          #hide
    xlim=(domain.lower[1], domain.upper[1]),    #hide
    label="f(x)",               #hide
    xlabel="x",                 #hide
    ylabel="f(x)",             #hide
    legend=:outertopright,              #hide
) #hide

x_min = plot_domain[argmin(ys)] #hide

scatter!([x_min], [minimum(ys)]; label="Minimum", color=:red, markersize=5) #hide




# ## Initialize the surrogate model
# We'll use a standard Gaussian Process surrogate with a Matérn 5/2 kernel. We add a small jitter term for numerical stability of 1e-12.
noise_var = 1e-9
surrogate = StandardGP(Matern52Kernel(), noise_var)

# Generate uniform random samples x_train
n_train = 5
x_train = [domain.lower .+ (domain.upper .- domain.lower) .* rand(d) for _ in 1:n_train]

y_train = f.(x_train)
y_train = map(x -> [x], y_train) # make y_train a vector of vectors, usual format for AbstractBayesOpt

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
    0.0,  # Actual noise level (0.0 for noiseless)
)

print_info(bo_struct)

@info "Starting Bayesian Optimization..."
result, acq_list, standard_params = AbstractBayesOpt.optimize(
    bo_struct; standardize=nothing
)

# ## Results
# The optimization result is stored in `result`. We can print the best found input and its corresponding function value.
xs = reduce(vcat, result.xs)
ys = reduce(vcat, result.ys_non_std)

println("Optimal point: ", xs[argmin(ys)])
println("Optimal value: ", minimum(ys))

# ## Plotting of running minimum over iterations
# The running minimum is the best function value found up to each iteration.
running_min = accumulate(min, f.(xs))

p = Plots.plot(
    n_train:length(running_min),
    running_min[n_train:end] .- min_f;
    yaxis=:log,
    title="Error w.r.t true minimum (1D BO)",
    xlabel="Function evaluations",
    label="BO",
    xlims=(1, length(running_min)),
)
Plots.vspan!([1, n_train]; color=:blue, alpha=0.2, label="")

# ## Gradient-enhanced GPs
# Now, let's see how to use gradient information to improve the optimization. We'll use the same function but now also provide its gradient.
# We define a new surrogate model that can handle gradient information, specifically a `GradientGP`.
grad_surrogate = GradientGP(ApproxMatern52Kernel(), d+1, noise_var)

ξ = 0.0
acq = ExpectedImprovement(ξ, minimum(reduce(vcat, y_train)))

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
    10,  # number of iterations
    0.0,  # Actual noise level (0.0 for noiseless)
)

print_info(bo_struct_grad)

@info "Starting Bayesian Optimization..."
result_grad, acq_list_grad, standard_params_grad = AbstractBayesOpt.optimize(bo_struct_grad)

# ## Results
# The optimization result is stored in `result`. We can print the best found input and its corresponding function value.
xs_grad = reduce(vcat, result_grad.xs)
ys_grad = hcat(result_grad.ys_non_std...)[1, :]

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
    title="Error w.r.t true minimum (1D GradBO)",
    xlabel="Function evaluations",
    label="gradBO",
    xlims=(1, length(running_min_grad)),
)
Plots.vspan!([1, 2*n_train]; color=:blue, alpha=0.2, label="")

# ## Plotting the surrogate model
# We can visualize the surrogate model's mean and uncertainty along with the true function and the evaluated

plot_domain = collect(domain.lower[1]:0.01:domain.upper[1])

plot_x = map(x -> [x], plot_domain)
plot_x = prep_input(grad_surrogate, plot_x)
post_mean, post_var = unstandardized_mean_and_var(
    result_grad.model, plot_x, standard_params_grad
)

post_mean = reshape(post_mean, :, d+1)[:, 1] # This returns f(x) to match the StandardGP
post_var = reshape(post_var, :, d+1)[:, 1]
post_var[post_var .< 0] .= 0

plot(
    plot_domain,
    f.(plot_domain);
    label="target function",
    xlim=(domain.lower[1], domain.upper[1]),
    xlabel="x",
    ylabel="y",
    title="AbstractBayesOpt",
    legend=:outertopright,
)
plot!(
    plot_domain,
    post_mean;
    label="gradGP",
    ribbon=sqrt.(post_var),
    ribbon_scale=2,
    color="green",
)
scatter!(xs_grad[1:n_train], ys_grad[1:n_train]; label="Train Data")
scatter!(xs_grad[(n_train + 1):end], ys_grad[(n_train + 1):end]; label="Candidates")
scatter!([xs_grad[argmin(ys_grad)]], [minimum(ys_grad)]; label="Best candidate")
