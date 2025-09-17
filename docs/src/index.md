# AbstractBayesOpt.jl

`AbstractBayesOpt` is a Julia library for Bayesian Optimisation (BO), which relies on abstract classes for surrogate models, acquisition functions and domain definitions.

The library is designed to solve **minimisation problems** of the form:

$$\min_{x \in \mathcal{X}} f(x)$$

where $f: \mathcal{X} \to \mathbb{R}$ is the objective function, which can be expensive to evaluate, non-differentiable, or noisy. The optimisation domain $\mathcal{X} \subseteq \mathbb{R}^d$ can be continuous, bounded, and possibly multi-dimensional.  

The library uses **Bayesian Optimisation (BO)** to iteratively propose evaluation points \( x \) in the domain by:

1. Modeling the objective function \( f \) with a **surrogate model** (e.g., Gaussian Process).  
2. Using an **acquisition function** to select the next query point that balances exploration and exploitation.  
3. Updating the surrogate with new observations and repeating until a **stopping criterion** is met.

## Abstract Interfaces

We currently have three main abstract interfaces that work with our BO loop:

- `AbstractAcquisition`: Interface to implement for an acquisition function to be used in `AbstractBayesOpt`.
- `AbstractSurrogate`: Interface to implement for a surrogate to be used in `AbstractBayesOpt`.
- `AbstractDomain`: Interface to implement for the optimisation domain to be used in `AbstractBayesOpt`.

AbstractBayesOpt.jl defines the core abstractions for building Bayesian optimization
algorithms. To add a new surrogate model, acquisition function, or domain, implement
the following interfaces:

---

### Surrogates

Subtype [`AbstractSurrogate`](@ref) and implement:

- `update(model::AbstractSurrogate, xs::AbstractVector, ys::AbstractVector)`:  
  Update the surrogate with new data `(xs, ys)`.

- `posterior_mean(model::AbstractSurrogate, x::AbstractVector)`:  
  Return the posterior mean at point `x`.

- `posterior_var(model::AbstractSurrogate, x::AbstractVector)`:  
  Return the posterior variance at point `x`.

- `nlml(model::AbstractSurrogate, params::AbstractVector, xs::AbstractVector, ys::AbstractVector)`:  
  Compute the negative log marginal likelihood given hyperparameters and data.

---

### Acquisition Functions

Subtype [`AbstractAcquisition`](@ref) and implement:

- `(acq::AbstractAcquisition)(model::AbstractSurrogate, x, x_buf=nothing)`:  
  Evaluate the acquisition function at `x`.  
  Optionally use a buffer `x_buf` for caching.

- `update(acq::AbstractAcquisition, ys::AbstractVector, model::AbstractSurrogate)`:  
  Update acquisition state given new observations.

- `Base.copy(acq::AbstractAcquisition)`:  
  Return a copy of the acquisition function.

---

### Domains

Subtype [`AbstractDomain`](@ref) and implement:

Concrete implementations should subtype this and define the necessary properties:
- `lower`: The lower bounds of the domain.
- `upper`: The upper bounds of the domain.

as well as creating its constructor.
---

Concrete implementations may add additional methods as needed, but these are the
minimum required for compatibility with the Bayesian optimization loop.

## What abstract subtypes are currently implemented in AbstractBayesOpt

### Acquisition functions
- `ExpectedImprovement`: Standard expected improvement acquisition function for balancing exploration and exploitation.
- `UpperConfidenceBound`: Acquisition function using a confidence bound to guide optimisation.
- `GradientNormUCB`: Gradient-based variant of the Upper Confidence Bound acquisition function.
- `ProbabilityImprovement`: Probability of improvement acquisition function.
- `EnsembleAcquisition`: Combines multiple acquisition functions into an ensemble to leverage complementary strategies.

### Surrogates
- `StandardGP`: Gaussian Process surrogate model with standard mean and covariance functions.
- `GradientGP`: Gaussian Process surrogate model supporting gradient information. 

### Domains
- `ContinuousDomain`: Represents a continuous optimisation domain, defining bounds and dimensionality for optimisation problems.