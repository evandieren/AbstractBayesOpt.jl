"""
    Abstract

An abstract type for surrogate models and acquisition functions used in Bayesian optimization.

Use abstraction to keep the types used in AbstractBayesOpt.jl
separated from their implementation (e.g. using AbstractGP.jl and KernelFunctions.jl) in folders acquisition/ and surrogates/
"""

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

If you wish to standardize the outputs, you can also implement:
- `std_y(model::AbstractSurrogate)`:
    Get the standard deviation used for standardizing the outputs in the surrogate model.
- `get_mean_std(model::AbstractSurrogate)`:
    Get the mean and standard deviation used for standardizing the outputs in the surrogate model.

Other methods can be added as needed depending on the use case, and we refer to the impelementations of [`StandardGP`](@ref) and [`GradientGP`](@ref) for examples.
"""
abstract type AbstractSurrogate end

"""
    AbstractAcquisition

Abstract type for acquisition functions used in Bayesian optimization.

Concrete implementation should subtype this and implement the following methods:
- `(acq::AbstractAcquisition)(surrogate::AbstractSurrogate, x::AbstractVector)`:
    Evaluate the acquisition function at point `x` using the surrogate model. 
    This should also work for a single real input `x::Real` if working in 1D, in which case it is treated as a one-dimensional input vector. via the abstract method defined below.
- `update(acq::AbstractAcquisition, ys::AbstractVector, model::AbstractSurrogate)`:
    Update the acquisition function with new observations `ys` and the current surrogate model.
- `Base.copy(acq::AbstractAcquisition)`:
    Create a copy of the acquisition function.
"""
abstract type AbstractAcquisition end

"""
    (acq::AbstractAcquisition)(surrogate::AbstractSurrogate, x::AbstractVector)

Evaluate the acquisition function at a given input point if Real using the surrogate model. 


This is a wrapper to allow for 1D optimization where the input is a single real number.

Arguments:
- `acq::AbstractAcquisition`: The acquisition function to be evaluated.
- `surrogate::AbstractSurrogate`: The surrogate model used in the acquisition function.
- `x::AbstractVector`: The input point where the acquisition function is to be evaluated.

returns:
- `Real`: The value of the acquisition function at the input point.
"""
function (acq::AbstractAcquisition)(surrogate::AbstractSurrogate, x::X) where {X<:Real}
    acq(surrogate, [x])[1]
end

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
