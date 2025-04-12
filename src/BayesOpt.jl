module BayesOpt

using AbstractGPs, KernelFunctions
using ForwardDiff
using Optim
using FillArrays
using Statistics
using SpecialFunctions
using Plots
using Distributions
#using GLMakie
using Distances

# Interface definitions
include("abstract.jl")
export AbstractAcquisition, AbstractSurrogate, AbstractDomain

# Domain implementations
include("domains.jl")
export ContinuousDomain

# Surrogate models
include("surrogates/surrogates_utils.jl")
include("surrogates/StandardGP.jl")
export StandardGP, prep_input, posterior_mean, posterior_var

include("surrogates/GradientGP.jl") # not implemented yet
export GradientGP, ApproxMatern52Kernel, gradMean, gradKernel, prep_input, posterior_mean, posterior_var

# Acquisition functions
include("acquisition/acq_utils.jl")
include("acquisition/ExpectedImprovement.jl")
include("acquisition/ProbabilityImprovement.jl")
include("acquisition/KnowledgeGradient.jl")
export normcdf, normpdf, optimize_mean!, ExpectedImprovement, ei, ProbabilityImprovement, pi, KnowledgeGradient, kg

# Core Bayesian Optimization framework
include("bayesian_opt.jl")
export optimize, print_info, update!, BOProblem

end
