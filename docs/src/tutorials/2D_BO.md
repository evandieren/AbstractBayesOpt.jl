```@meta
EditURL = "../../literate/tutorials/2D_BO.jl"
```

# AbstractBayesOpt Tutorial: Basic 2D Optimisation

## Setup

Loading the necessary packages.

````@example 2D_BO
using AbstractBayesOpt
using AbstractGPs
using ForwardDiff
using Plots
default(; legend=:outertopright, size=(700, 400)) # hide

using Random # hide
Random.seed!(42) # hide
nothing #hide
````

## Define the objective function

````@example 2D_BO
f(x) = (x[1]^2 + x[2] - 11)^2 + (x[1]+x[2]^2-7)^2
min_f = 0.0 # hide
d = 2
domain = ContinuousDomain([-6.0, -6.0], [6.0, 6.0])

resolution = 100 #hide
X = range(domain.lower[1], domain.upper[1]; length=resolution) # hide
Y = range(domain.lower[2], domain.upper[2]; length=resolution) # hide

p1 = contour( # hide
    X, # hide
    Y, # hide
    (x, y) -> f([x, y]); # hide
    fill=true, # hide
    levels=50, # hide
    c=:coolwarm, # hide
    title="Target function : Himmelblau", # hide
    xlabel="x₁", # hide
    ylabel="x₂", # hide
) # hide

x_mins = [[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]] #hide
````

Scatter them on the contour plot

````@example 2D_BO
scatter!( # hide
    [p[1] for p in x_mins], # hide
    [p[2] for p in x_mins]; # hide
    label="Minima", # hide
    color=:red, # hide
    markersize=5, # hide
    legend=:bottomright, # hide
) # hide
````

## Standard GPs
We'll use a standard Gaussian Process surrogate with a squared-exponential kernel. We add a small jitter term for numerical stability of ``10^{-9}``.

````@example 2D_BO
noise_var = 1e-9
surrogate = StandardGP(SqExponentialKernel(), noise_var)
````

Generate uniform random samples x_train

````@example 2D_BO
n_train = 5
x_train = [domain.lower .+ (domain.upper .- domain.lower) .* rand(d) for _ in 1:n_train]

y_train = f.(x_train)
````

### Choose an acquisition function
We'll use the Expected Improvement acquisition function with an exploration parameter ξ = 0.0.

````@example 2D_BO
ξ = 0.0
acq = ExpectedImprovement(ξ, minimum(y_train))
````

### Set up the Bayesian Optimisation structure
We use BOStruct to bundle all components needed for the optimization. Here, we set the number of iterations to 5 and the actual noise level to 0.0 (since our function is noiseless).
We then run the optimize function to perform the Bayesian Optimisation.

````@example 2D_BO
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

@info "Starting Bayesian Optimisation..."
result, acq_list, standard_params = AbstractBayesOpt.optimize(
    bo_struct; standardize="mean_only"
);
nothing #hide
````

### Results
The optimization result is stored in `result`. We can print the best found input and its corresponding function value.

````@example 2D_BO
xs = result.xs # hide
ys = result.ys_non_std # hide

println("Optimal point: ", xs[argmin(ys)]) # hide
println("Optimal value: ", minimum(ys)) # hide
````

### Plotting of running minimum over iterations
The running minimum is the best function value found up to each iteration.

````@example 2D_BO
running_min = accumulate(min, f.(xs)) # hide

p = Plots.plot( # hide
    n_train:length(running_min), # hide
    running_min[n_train:end] .- min_f; # hide
    yaxis=:log, # hide
    title="Error w.r.t true minimum (2D BO)", # hide
    xlabel="Function evaluations", # hide
    label="BO", # hide
    xlims=(1, length(running_min)), # hide
) # hide
Plots.vspan!([1, n_train]; color=:blue, alpha=0.2, label="") # hide
````

## Gradient-enhanced GPs
Now, let's see how to use gradient information to improve the optimization. We'll use the same function but now also provide its gradient.
We define a new surrogate model that can handle gradient information, specifically a `GradientGP`.

````@example 2D_BO
grad_surrogate = GradientGP(SqExponentialKernel(), d + 1, noise_var)

ξ = 0.0
acq = ExpectedImprovement(ξ, minimum(y_train))

∇f(x) = ForwardDiff.gradient(f, x)
f_val_grad(x) = [f(x); ∇f(x)];
nothing #hide
````

Generate value and gradients at random samples

````@example 2D_BO
y_train_grad = f_val_grad.(x_train)
````

Set up the Bayesian Optimisation structure

````@example 2D_BO
bo_struct_grad = BOStruct(
    f_val_grad,
    acq,
    grad_surrogate,
    domain,
    x_train,
    y_train_grad,
    20,  # number of iterations
    0.0,  # Actual noise level (0.0 for noiseless)
)

@info "Starting Bayesian Optimisation..." # hide
result_grad, acq_list_grad, standard_params_grad = AbstractBayesOpt.optimize(bo_struct_grad);
nothing #hide
````

### Results
The optimization result is stored in `result_grad`. We can print the best found input and its corresponding function value.

````@example 2D_BO
xs_grad = result_grad.xs # hide
ys_grad = first.(result_grad.ys_non_std) # hide

x_min_grad = xs_grad[argmin(ys_grad)] # hide
y_min_grad = minimum(ys_grad) # hide

println("Optimal point (GradBO): ", x_min_grad) # hide
println("Optimal value (GradBO): ", y_min_grad) # hide
````

### Plotting of running minimum over iterations
The running minimum is the best function value found up to each iteration.
Since each evaluation provides both a function value and a 2D gradient, we duplicate the running minimum values 3x to reflect the number of function evaluations.

````@example 2D_BO
running_min_grad = accumulate(min, f.(xs_grad)); # hide
running_min_grad = collect(Iterators.flatten(fill(x, 3) for x in (running_min_grad))) # hide

p = Plots.plot( # hide
    (3 * n_train):length(running_min_grad), # hide
    running_min_grad[(3 * n_train):end] .- min_f; # hide
    yaxis=:log, # hide
    title="Error w.r.t true minimum (2D GradBO)", # hide
    xlabel="Function evaluations", # hide
    label="gradBO", # hide
    xlims=(1, length(running_min_grad)), # hide
) # hide
Plots.vspan!([1, 3 * n_train]; color=:blue, alpha=0.2, label="") # hide
````

We observe that the gradient information does not necessarily lead to a better optimisation path in terms of function evaluations.

### Plotting the surrogate model
We can visualize the surrogate model's mean and uncertainty along with the true function and the evaluated

````@example 2D_BO
zipped_grid = [ # hide
    [x1, x2] for # hide
    (x1, x2) in zip(vec(repeat(X', resolution, 1)), vec(repeat(Y, 1, resolution))) # hide
] # hide

μ, σ² = unstandardized_mean_and_var(result_grad.model, zipped_grid, standard_params_grad) # hide
μ_function_grid = reshape(μ[:, 1], resolution, resolution) # hide

p1 = contour( # hide
    X, # hide
    Y, # hide
    (x, y) -> f([x, y]); # hide
    fill=true, # hide
    levels=50, # hide
    c=:coolwarm, # hide
    title="Target function : Himmelblau", # hide
) # hide

scatter!( # hide
    [p[1] for p in x_mins], # hide
    [p[2] for p in x_mins]; # hide
    label="", # hide
    color=:green, # hide
    markershape=:diamond, # hide
    markersize=5, # hide
) # hide

p2 = contour( # hide
    X, # hide
    Y, # hide
    μ_function_grid; # hide
    fill=true, # hide
    levels=50, # hide
    c=:coolwarm, # hide
    title="Surrogate mean - GradBO", # hide
) # hide
scatter!( # hide
    p2, # hide
    [x[1] for x in x_train], # hide
    [x[2] for x in x_train]; # hide
    label="", # hide
    color=:black, # hide
    markershape=:x, # hide
    markersize=5,# hide
) # hide

scatter!( # hide
    p2, # hide
    [x[1] for x in xs_grad[(n_train + 1):end]], # hide
    [x[2] for x in xs_grad[(n_train + 1):end]]; # hide
    label="", # hide
    color=:orange, # hide
    markersize=5, # hide
) # hide
scatter!( # hide
    p2, # hide
    [x_min_grad[1]], # hide
    [x_min_grad[2]]; # hide
    label="", # hide
    color=:red, # hide
    markershape=:star5, # hide
    markersize=8, # hide
) # hide

p_legend = plot(; legend=:bottom, grid=false, axis=false, ticks=false, legend_columns=4) # hide

scatter!( # hide
    p_legend, # hide
    [NaN], # hide
    [NaN]; # hide
    label="Training points", # hide
    color=:black, # hide
    markershape=:x, # hide
    markersize=5, # hide
) # hide
scatter!(p_legend, [NaN], [NaN]; label="Candidate points", color=:orange, markersize=5) # hide
scatter!( # hide
    p_legend, # hide
    [NaN], # hide
    [NaN]; # hide
    label="Best candidate", # hide
    color=:red, # hide
    markershape=:star5, # hide
    markersize=8, # hide
) # hide
scatter!( # hide
    p_legend, # hide
    [NaN], # hide
    [NaN]; # hide
    label="Function minima", # hide
    color=:green, # hide
    markershape=:diamond, # hide
    markersize=5, # hide
) # hide

p = plot(p1, p2) # hide
finalplot = plot(p, p_legend; layout=@layout([a{0.9h}; b{0.1h}]), size=(1000, 400)) # hide
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

