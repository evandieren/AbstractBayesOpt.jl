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

- `AbstractAcquisition`: Interface to implement for an acquisition function to be used in `AbstractBayesOpt`.
- `AbstractSurrogate`: Interface to implement for a surrogate to be used in `AbstractBayesOpt`.
- `AbstractDomain`: Interface to implement for the optimisation domain to be used in `AbstractBayesOpt`.

These define the abstract types and required methods for surrogates, acquisitions, and domains. The BO loop only relies on these abstract classes.

## What is currently implemented

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