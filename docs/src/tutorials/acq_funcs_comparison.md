```@meta
EditURL = "../../literate/tutorials/acq_funcs_comparison.jl"
```

# AbstractBayesOpt Tutorial: Acquisition Functions Comparison with gradient-enhanced GPs

## Setup

Loading the necessary packages.

````@example acq_funcs_comparison
using AbstractBayesOpt
using AbstractGPs
using Plots
using ForwardDiff
using QuasiMonteCarlo
using Random

default(; legend=:outertopright, size=(700, 400)) # hide

Random.seed!(555) # hide
nothing # hide
````

## Define the objective function

We will compare different acquisition functions on a 1D function with multiple local minima:
``f(x) = \sin(x + 1) + \sin(\frac{10}{3}(x + 1))``

````@example acq_funcs_comparison
f(x) = sin(x + 1) + sin((10.0 / 3.0) * (x + 1))
∂f(x) = ForwardDiff.derivative(f, x)
min_f = −1.988699758534924 # hide
f_∂f(x) = [f(x); ∂f(x)];

d = 1
lower = [-10.0]
upper = [10.0]
domain = ContinuousDomain(lower, upper)

plot_domain = lower[1]:0.01:upper[1] # hide
ys = f.(plot_domain) # hide

plot( # hide
    plot_domain, # hide
    ys; # hide
    xlim=(lower[1], upper[1]), # hide
    label="f(x)", # hide
    xlabel="x", # hide
    ylabel="f(x)", # hide
    title="Objective Function: sin(x+1) + sin(10(x+1)/3)", # hide
    legend=:outertopright, # hide
) # hide

approx_min = [-8.9981; 9.8514] # hide
f_min = f.(approx_min) # hide
scatter!([approx_min], [f_min]; label="Global minima", color=:red, markersize=5) # hide
````

### Initial Training Data

We will use a gradient-enhanced Gaussian Process ([`GradientGP`](@ref)) with a Matérn 5/2 kernel.
We add a small noise variance for numerical stability.

````@example acq_funcs_comparison
σ² = 1e-12;
nothing #hide
````

Generate initial training data using Sobol sampling for better space coverage

````@example acq_funcs_comparison
n_train = 5
x_train = vec(QuasiMonteCarlo.sample(n_train, lower, upper, SobolSample()))
y_train = f_∂f.(x_train)
````

Setup the gradient-enhanced GP model, using in-house [`ApproxMatern52Kernel`](@ref) for AD compatibility.

````@example acq_funcs_comparison
model = GradientGP(ApproxMatern52Kernel(), d+1, σ²)
````

## Acquisition Functions Setup

We will compare five different acquisition functions:
1. **Expected Improvement (EI)**: Balances exploitation and exploration by considering both the magnitude and probability of improvement (see [`ExpectedImprovement`](@ref))
2. **Probability of Improvement (PI)**: Focuses on the probability of finding a better point (see [`ProbabilityImprovement`](@ref))
3. **Upper Confidence Bound (UCB)**: Uses lower bound estimates to guide exploration (see [`UpperConfidenceBound`](@ref))
4. **UCB + Gradient UCB Ensemble**: Combines standard UCB with gradient norm information (see [`GradientNormUCB`](@ref) and [`EnsembleAcquisition`](@ref))
5. **EI + Gradient UCB Ensemble**: Combines Expected Improvement with gradient norm information (see [`GradientNormUCB`](@ref) and [`EnsembleAcquisition`](@ref))

We show below the function to setup the acquisition functions, and run the tests.
You can skip to the results analysis and visualisation section if you want to see the outcomes directly.

````@example acq_funcs_comparison
function setup_acquisition_functions(y_train)
    best_y = minimum(first.(y_train))

    ei_acq = ExpectedImprovement(0.0, best_y)

    pi_acq = ProbabilityImprovement(0.0, best_y)

    ucb_acq = UpperConfidenceBound(1.96)

    ucb_acq_for_ensemble = UpperConfidenceBound(1.96)
    grad_ucb_acq = GradientNormUCB(1.5)
    ensemble_ucb_grad = EnsembleAcquisition(
        [0.9, 0.1], [ucb_acq_for_ensemble, grad_ucb_acq]
    )

    ei_for_ensemble = ExpectedImprovement(0.0, best_y)
    grad_ucb_for_ensemble = GradientNormUCB(1.5)
    ensemble_ei_grad = EnsembleAcquisition(
        [0.9, 0.1], [ei_for_ensemble, grad_ucb_for_ensemble]
    )

    return [
        ("EI", ei_acq),
        ("PI", pi_acq),
        ("UCB", ucb_acq),
        ("UCB+GradUCB", ensemble_ucb_grad),
        ("EI+GradUCB", ensemble_ei_grad),
    ]
end
````

### Running the Optimisation Comparison

Now we will run Bayesian optimisation with each acquisition function and compare their performance.

````@example acq_funcs_comparison
function run_comparison(n_iterations=30)
    results = Dict{String, Any}()

    for (name, acq_func) in setup_acquisition_functions(y_train)
        @info "\n=== Running optimisation with $name ==="

        problem = BOStruct(
            f_∂f,
            acq_func,
            model,
            domain,
            x_train,
            y_train,
            n_iterations,
            0.0,  # Actual noise level (0.0 for noiseless)
        )

        result, _, _ = AbstractBayesOpt.optimize(problem);

        xs = result.xs
        ys = first.(result.ys_non_std)

        optimal_point = xs[argmin(ys)]
        optimal_value = minimum(ys)

        @info "Optimal point: $optimal_point"
        @info "Optimal value: $optimal_value"
        @info "Error from true minimum: $(abs(optimal_value - min_f))"

        running_min = accumulate(min, f.(xs));

        results[name] = (
            xs=xs,
            ys=ys,
            running_min=running_min,
            optimal_point=optimal_point,
            optimal_value=optimal_value,
            error=abs(optimal_value - min_f),
        )
    end

    return results
end;
nothing #hide
````

### Execute the comparison

Let's run the optimisation with each acquisition function for 30 iterations.

````@example acq_funcs_comparison
@info "Starting acquisition function comparison..." # hide
results = run_comparison(30)
````

### Results Analysis and Visualisation

We will create a convergence plot showing how each acquisition function performs over time.
The plot shows the error relative to the true minimum on a logarithmic scale.

````@example acq_funcs_comparison
function plot_convergence(results) # hide
    p = plot(; # hide
        title="Acquisition Function Comparison (1D GradBO)", # hide
        xlabel="Function evaluations", # hide
        yaxis=:log, # hide
        legend=:bottomleft, # hide
        linewidth=2, # hide
    ) # hide

    colors = [:blue, :red, :green, :orange, :purple] # hide

    for (i, (name, result)) in enumerate(results) # hide
        running_min_extended = collect( # hide
            Iterators.flatten(fill(x, 2) for x in result.running_min) # hide
        ) # hide
        errors = max.(running_min_extended .- min_f, 1e-16)  # Avoid log(0) # hide

        plot!( # hide
            p, # hide
            (2 * n_train):length(errors), # hide
            errors[(2 * n_train):end]; # hide
            label=name, # hide
            color=colors[i], # hide
            alpha=0.8, # hide
        ) # hide
    end # hide

    vspan!(p, [1, 2*n_train]; color=:gray, alpha=0.2, label="Initial data") # hide

    return p # hide
end; # hide
nothing #hide
````

### Create and display the convergence plot

This plot shows how quickly each acquisition function converges to the global minimum.
The ensemble methods that combine multiple acquisition functions often show improved performance.

````@example acq_funcs_comparison
conv_plot = plot_convergence(results) # hide
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

