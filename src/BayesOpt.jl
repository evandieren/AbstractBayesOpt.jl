module BayesOpt

using AbstractGPs, KernelFunctions
using ElasticArrays
using ForwardDiff
using Optim
using FillArrays
using Statistics
using SpecialFunctions
using Plots
using Distributions

# Interface definitions
include("abstract.jl")
export AbstractAcquisition, AbstractSurrogate, AbstractDomain

# Domain implementations
include("domains.jl")
export ContinuousDomain

# Surrogate models
include("surrogates/StandardGP.jl")
export StandardGP, posterior_mean, posterior_var
# include("surrogates/GradientGP.jl") # not implemented yet

# Acquisition functions
include("acquisition/ExpectedImprovement.jl")
include("acquisition/ProbabilityImprovement.jl")
export ExpectedImprovement, ei, ProbabilityImprovement, pi

# Core Bayesian Optimization framework
include("bayesian_opt.jl")
export optimize, print_info, update!, BOProblem

# Optimization tools
include("optimizer.jl")

end
