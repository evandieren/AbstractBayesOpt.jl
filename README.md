# AbstractBayesOpt
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://evandieren.github.io/AbstractBayesOpt.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://evandieren.github.io/AbstractBayesOpt.jl/dev)
[![Build Status](https://github.com/evandieren/AbstractBayesOpt.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/evandieren/AbstractBayesOpt.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/evandieren/AbstractBayesOpt.jl/graph/badge.svg?token=ZU1TJCHT4O)](https://codecov.io/gh/evandieren/AbstractBayesOpt.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)



AbstractBayesOpt.jl is a general framework for Bayesian Optimisation in Julia. It relies on abstract classes for surrogate models, acquisition functions and domain definitions.
The codebase is entirely written in Julia.

The library is designed to solve **minimisation problems** of the form:

$$\min_{x \in \mathcal{X}} f(x)$$

where $f: \mathcal{X} \to \mathbb{R}$ is the objective function, which can be expensive to evaluate, non-differentiable, or noisy. The optimisation domain $\mathcal{X} \subseteq \mathbb{R}^d$ can be continuous, bounded, and possibly multi-dimensional.  

The library uses **Bayesian Optimisation (BO)** to iteratively propose evaluation points $x$ in the domain by:

1. Modelling the objective function with a **surrogate model** (e.g., Gaussian Process).  
2. Using an **acquisition function** to select the next query point that balances exploration and exploitation.  
3. Updating the surrogate with new observations and repeating until a **stopping criterion** is met.

## Installation

You can install AbstractBayesOpt.jl directly from this repository via the Julia General Registry

```
using Pkg
Pkg.add("AbstractBayesOpt")
```
## What does it provide?

We currently have three main abstract interfaces that work with our BO loop:

- `AbstractAcquisition`: Interface to implement for an acquisition function to be used in AbstractBayesOpt.
- `AbstractSurrogate`: Interface to implement for a surrogate to be used in AbstractBayesOpt.
- `AbstractDomain`: Interface to implement for the optimisation domain to be used in AbstractBayesOpt.

AbstractBayesOpt.jl defines the core abstractions for building Bayesian optimisation
algorithms. To add a new surrogate model, acquisition function, or domain, implement
the following interfaces.

We refer to the [documentation](https://evandieren.github.io/AbstractBayesOpt.jl/) for an extensive description of the features of the library, including the different 
subtypes we have implemented, as well as some tutorials and the public API.

## Issues and Contributing

We follow the practices from ColPrac, and we warmly welcome any contribution to fix bugs, or implement new features. Please [submit an issue](https://github.com/evandieren/AbstractBayesOpt.jl/issues) to start working on AbstractBayesOpt.jl !
