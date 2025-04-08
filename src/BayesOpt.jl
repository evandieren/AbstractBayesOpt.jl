module BayesOpt

using AbstractGPs, KernelFunctions
using ForwardDiff
using Optim
using FillArrays
using Statistics
using SpecialFunctions
using Plots
using Distributions
using GLMakie
using Distances

# Interface definitions
include("abstract.jl")
export AbstractAcquisition, AbstractSurrogate, AbstractDomain

# Domain implementations
include("domains.jl")
export ContinuousDomain

# Surrogate models
include("surrogates/StandardGP.jl")
export StandardGP, prep_input, posterior_mean, posterior_var

include("surrogates/GradientGP.jl") # not implemented yet
export GradientGP, ApproxMatern52Kernel, gradKernel, prep_input, posterior_mean, posterior_var

# Acquisition functions
include("acquisition/acq_utils.jl")
include("acquisition/ExpectedImprovement.jl")
include("acquisition/ProbabilityImprovement.jl")
export normcdf, normpdf, ExpectedImprovement, ei, ProbabilityImprovement, pi

# Core Bayesian Optimization framework
include("bayesian_opt.jl")
export optimize, print_info, update!, BOProblem

# Optimization tools
include("optimizer.jl")

end
