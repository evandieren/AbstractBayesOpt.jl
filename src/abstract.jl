"""
    Abstract

An abstract type for surrogate models and acquisition functions used in Bayesian optimization.

Use abstraction to keep the types used in AbstractBayesOpt.jl 
separated from their implementation (e.g. using AbstractGP.jl and KernelFunctions.jl) in folders acquisition/ and surrogates/
"""
# AbstractBayesOpt.jl: Abstract types and update/optimize functions

"""
    AbstractSurrogate

Abstract type for surrogate models used in Bayesian optimization.

Concrete implementation should subtype this and implement the following methods:
- `update(model::AbstractSurrogate, xs::AbstractVector, ys::AbstractVector)`: 
    Update the surrogate model with new data points `xs` and corresponding observations `ys`.
- `posterior_mean(surrogate::AbstractSurrogate, x::AbstractVector)`: 
    Compute the posterior mean of the surrogate model at point `x`.
- `posterior_var(surrogate::AbstractSurrogate, x::AbstractVector)`: 
    Compute the posterior variance of the surrogate model at point `x`.
- `nlml(surrogate::AbstractSurrogate, params::AbstractVector, xs::AbstractVector, ys::AbstractVector)`: 
    Compute the negative log marginal likelihood of the surrogate model given hyperparameters `params`, input data `xs`, and observations `ys`.


Other methods can be added as needed depending on the use case.
"""
abstract type AbstractSurrogate end


"""
    AbstractAcquisition

Abstract type for acquisition functions used in Bayesian optimization.

Concrete implementation should subtype this and implement the following methods:
- `(acq::AbstractAcquisition)(surrogate::AbstractSurrogate, x, x_buf=nothing)`: 
    Evaluate the acquisition function at point `x` using the surrogate model. Optionally, a buffer `x_buf` can be provided for caching purposes.
- `update(acq::AbstractAcquisition, ys::AbstractVector, model::AbstractSurrogate)`: 
    Update the acquisition function with new observations `ys` and the current surrogate model.
- `Base.copy(acq::AbstractAcquisition)`: 
    Create a copy of the acquisition function.
"""
abstract type AbstractAcquisition end



""" 
    AbstractDomain

An abstract type for defining the domain over which the optimization is performed.

Concrete implementations should subtype this and define the necessary properties:
- `lower`: The lower bounds of the domain.
- `upper`: The upper bounds of the domain.

as well as creating its constructor.

Other methods can be added as needed depending on the use case.
"""
abstract type AbstractDomain end