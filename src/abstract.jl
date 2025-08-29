"""
    Abstract

An abstract type for surrogate models and acquisition functions used in Bayesian optimization.

Use abstraction to cover the future addition of gradients, and keep the types used in AbstractBayesOpt.jl 
seperated from their implementation (e.g. using AbstractGP.jl and kernelfunctions.jl) in folders acquisition/ and surrogates/
"""

# Abstract Acquisition function
abstract type AbstractAcquisition end
abstract type AbstractSurrogate <: AbstractGPs.AbstractGP end
abstract type AbstractDomain end

function update! end
function optimize_acquisition! end
