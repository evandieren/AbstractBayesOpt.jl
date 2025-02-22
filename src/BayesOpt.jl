module BayesOpt

using Reexport
using AbstractGPs, KernelFunctions
using ElasticArrays
using ForwardDiff
using Optim
using FillArrays
using Statistics
using SpecialFunctions

# Interface definitions
include("abstract.jl")
export AbstractAcquisition, AbstractSurrogate, AbstractDomain

# Domain implementations
include("domains.jl")
export ContinuousDomain

# Surrogate models
export StandardGP, posterior_mean, posterior_var
include("surrogates/StandardGP.jl")
# include("surrogates/GradientGP.jl") # not implemented yet

# Acquisition functions
export ExpectedImprovement, ei, ProbabilityImprovement, pi
include("acquisition/ExpectedImprovement.jl")
include("acquisition/ProbabilityImprovement.jl")

# Core Bayesian Optimization framework
export optimize, print_info
include("bayesian_opt.jl")

# Optimization tools
include("optimizer.jl")

# Public API exports
export BOProblem, optimize!, update!

export optimize_acquisition!

end
