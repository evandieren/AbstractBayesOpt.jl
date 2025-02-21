# src/Abstract.jl
module Abstract
"""
    Abstract

An abstract type for surrogate models and acquisition functions used in Bayesian optimization.

Reason: Use abstraction to cover the future addition of gradients, and keep the types used in BayesOpt.jl 
seperated from their implementation (e.g. using AbstractGP.jl and kernelfunctions.jl) in folders acquisition/ and surrogates/
"""


# Abstract Acquisition function
abstract type AbstractAcquisition end

#function acquire end  # Generic acquisition interface
# we either do acquire or we call the function directly, but need to require a constructor

# Abstract Surrogates
abstract type AbstractSurrogate <: AbstractGPs.AbstractGP end
function update_model! end  # Model update

end  # module Abstract