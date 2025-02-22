module BayesOpt

using Reexport
using AbstractGPs, KernelFunctions
using ElasticArrays
using ForwardDiff
using Optim


# Interface definitions
include("Abstract.jl")

# Domain implementations
include("domains.jl")

# Surrogate models
include("surrogates/StandardGP.jl")
# include("surrogates/GradientGP.jl") # not implemented yet

# Acquisition functions
include("acquisition/ExpectedImprovement.jl")
include("acquisition/ProbabilityImprovement.jl")

# Core Bayesian Optimization framework
include("bayesian_opt.jl")

# Optimization tools
include("optimizer.jl")

# Public API exports
export BOProblem, bayesian_optimize!, update!
export posterior_mean, posterior_var
export optimize_acquisition!

# Re-export key dependencies
@reexport using .Abstract: AbstractSurrogate, AbstractAcquisition
@reexport using .Domains: ContinuousDomain, CategoricalDomain
@reexport using .Surrogates: StandardGP, GradientGP
@reexport using .Acquisition: ExpectedImprovement, ProbabilityImprovement


# Dependency interface exports (for extensibility)
export AbstractGPs, KernelFunctions, AbstractSurrogate, AbstractAcquisition

end
