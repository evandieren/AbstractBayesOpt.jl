"""
    Abstract

An abstract type for surrogate models and acquisition functions used in Bayesian optimization.

Use abstraction to cover the future addition of gradients, and keep the types used in AbstractBayesOpt.jl 
seperated from their implementation (e.g. using AbstractGP.jl and kernelfunctions.jl) in folders acquisition/ and surrogates/
"""
# AbstractBayesOpt.jl: Abstract types and update/optimize functions

# Abstract Acquisition function
abstract type AbstractAcquisition end
abstract type AbstractSurrogate <: AbstractGPs.AbstractGP end
abstract type AbstractDomain end

# Update functions for models, acquisition functions, and BO structure to implement 
function update(model::AbstractSurrogate, xs::AbstractVector, ys::AbstractVector) end

function update(acq::AbstractAcquisition, ys::AbstractVector, model::AbstractSurrogate) end

# Optimization functions for acquisition functions
function optimize_acquisition end
