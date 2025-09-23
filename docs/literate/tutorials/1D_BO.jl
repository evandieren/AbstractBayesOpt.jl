#=
# AbstractBayesOpt Tutorial: 1D Bayesian Optimisation
=#

# ## Setup
#
# Loading the necessary packages.
using AbstractBayesOpt
using AbstractGPs
using ForwardDiff
using Plots
default(; legend=:outertopright, size=(700, 400)) # hide

using Random # hide
Random.seed!(42) # hide
#md nothing # hide

# ## Define the objective function
# We will optimise a simple 1D function: ``f(x) = (x-2)^2 + \sin(3x)``
f(x) = (x - 2)^2 + sin(3x)
min_f = -0.8494048256167165 # hide
d = 1
domain = ContinuousDomain([0.0], [5.0])

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

# ## Standard GPs
# We'll use a standard Gaussian Process surrogate with a Matérn 5/2 kernel. We add a small jitter term for numerical stability of ``10^{-12}``.
noise_var = 1e-12
surrogate = StandardGP(Matern52Kernel(), noise_var)

# Generate uniform random samples `x_train` and evaluate the function at these points to get `y_train`.
n_train = 5
x_train = first.([
    domain.lower .+ (domain.upper .- domain.lower) .* rand(d) for _ in 1:n_train
])

y_train = f.(x_train)

# ### Choose an acquisition function
# We'll use the Expected Improvement acquisition function with an exploration parameter ξ = 0.0.
ξ = 0.0
acq = ExpectedImprovement(ξ, minimum(y_train))

# ### Set up the Bayesian Optimisation structure
# We use BOStruct to bundle all components needed for the optimisation. Here, we set the number of iterations to 5 and the actual noise level to 0.0 (since our function is noiseless).
# We then run the optimize function to perform the Bayesian optimisation.
bo_struct = BOStruct(
    f,
    acq,
    surrogate,
    domain,
    x_train,
    y_train,
    30,  # number of iterations
    0.0,  # Actual noise level (0.0 for noiseless)
)

@info "Starting Bayesian ..."
result, acq_list, standard_params = AbstractBayesOpt.optimize(
    bo_struct; standardize="mean_only"
);

# ### Results
# The  result is stored in `result`. We can print the best found input and its corresponding function value.
xs = result.xs # hide
ys = result.ys_non_std # hide

println("Optimal point: ", xs[argmin(ys)]) # hide
println("Optimal value: ", minimum(ys)) # hide

# ### Plotting of running minimum over iterations
# The running minimum is the best function value found up to each iteration.
running_min = accumulate(min, f.(xs)) # hide

p = Plots.plot(
    n_train:length(running_min),
    running_min[n_train:end] .- min_f;
    yaxis=:log,
    title="Error w.r.t true minimum (1D BO)",
    xlabel="Function evaluations",
    label="BO",
    xlims=(1, length(running_min)),
) # hide
Plots.vspan!([1, n_train]; color=:blue, alpha=0.2, label="training GP") # hide

# ## Gradient-enhanced GPs
# Now, let's see how to use gradient information to improve the optimisation. We'll use the same function but now also provide its gradient.
# We define a new surrogate model that can handle gradient information, specifically a [`GradientGP`](@ref).
grad_surrogate = GradientGP(ApproxMatern52Kernel(), d + 1, noise_var)

ξ = 0.0
acq = ExpectedImprovement(ξ, minimum(y_train))

∂f(x) = ForwardDiff.derivative(f, x)
f_∂f(x) = [f(x); ∂f(x)];

# Generate value and gradients at random samples
y_train_grad = f_∂f.(x_train)

# Set up the Bayesian Optimisation structure
bo_struct_grad = BOStruct(
    f_∂f,
    acq,
    grad_surrogate,
    domain,
    x_train,
    y_train_grad,
    10,  # number of iterations
    0.0,  # Actual noise level (0.0 for noiseless)
)

@info "Starting Bayesian Optimisation..." # hide
result_grad, acq_list_grad, standard_params_grad = AbstractBayesOpt.optimize(
    bo_struct_grad; standardize="mean_only"
);

# ### Results
# The  result is stored in `result_grad`. We can print the best found input and its corresponding function value.
xs_grad = reduce(vcat, result_grad.xs) # hide
ys_grad = hcat(result_grad.ys_non_std...)[1, :] # hide

println("Optimal point (GradBO): ", xs_grad[argmin(ys_grad)]) # hide
println("Optimal value (GradBO): ", minimum(ys_grad)) # hide

# ### Plotting of running minimum over iterations
# The running minimum is the best function value found up to each iteration.
# Since each evaluation provides both a function value and a 1D gradient, we duplicate the running minimum values to reflect the number of function evaluations.
running_min_grad = accumulate(min, f.(xs_grad)) # hide
running_min_grad = collect(Iterators.flatten(fill(x, 2) for x in (running_min_grad))) # hide

p = Plots.plot( # hide
    (2 * n_train):length(running_min_grad), # hide
    running_min_grad[(2 * n_train):end] .- min_f; # hide
    yaxis=:log, # hide
    title="Error w.r.t true minimum (1D GradBO)", # hide
    xlabel="Function evaluations", # hide
    label="gradBO", # hide
    xlims=(1, length(running_min_grad)), # hide
) # hide
Plots.vspan!([1, 2 * n_train]; color=:blue, alpha=0.2, label="") # hide

# ### Plotting the surrogate model
# We can visualize the surrogate model's mean and uncertainty along with the true function and the evaluated

plot_domain = collect(domain.lower[1]:0.01:domain.upper[1]) # hide

plot_x = map(x -> [x], plot_domain) # hide
plot_x = prep_input(grad_surrogate, plot_x) # hide
post_mean, post_var = unstandardized_mean_and_var( # hide
    result_grad.model,
    plot_x,
    standard_params_grad, # hide
) # hide

post_mean = reshape(post_mean, :, d + 1)[:, 1] # hide
post_var = reshape(post_var, :, d + 1)[:, 1] # hide
post_var[post_var .< 0] .= 0 # hide

plot(
    plot_domain, # hide
    f.(plot_domain); # hide
    label="target function", # hide
    xlim=(domain.lower[1], domain.upper[1]), # hide
    xlabel="x", # hide
    ylabel="y", # hide
    title="AbstractBayesOpt", # hide
    legend=:outertopright, # hide
) # hide
plot!(
    plot_domain, # hide
    post_mean; # hide
    label="gradGP", # hide
    ribbon=sqrt.(post_var), # hide
    ribbon_scale=2, # hide
    color="green", # hide
) # hide
scatter!(xs_grad[1:n_train], ys_grad[1:n_train]; label="Train Data") # hide
scatter!(xs_grad[(n_train + 1):end], ys_grad[(n_train + 1):end]; label="Candidates") # hide
scatter!([xs_grad[argmin(ys_grad)]], [minimum(ys_grad)]; label="Best candidate") # hide
