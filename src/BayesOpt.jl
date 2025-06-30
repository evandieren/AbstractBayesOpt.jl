module BayesOpt

using AbstractGPs, KernelFunctions
using ForwardDiff
using Optim
using FillArrays
using Statistics
using SpecialFunctions
using Plots
using Distributions
using LinearAlgebra
using Random
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
export StandardGP, prep_input, posterior_mean, posterior_var, nlml

include("surrogates/GradientGP.jl") # not implemented yet
export GradientGP, ApproxMatern52Kernel, gradMean, gradKernel, prep_input, posterior_mean, posterior_var, posterior_grad_mean, posterior_grad_cov, nlml_grad 

# Acquisition functions
include("acquisition/acq_utils.jl")
export normcdf, normpdf,optimize_mean!,optimize_acquisition!
include("acquisition/ExpectedImprovement.jl")
export ExpectedImprovement, ei
include("acquisition/UpperConfidenceBound.jl")
export UpperConfidenceBound, ucb
include("acquisition/gradNormUCB.jl")
export GradientNormUCB, gradUCB
#include("acquisition/ProbabilityDescent.jl")
include("acquisition/ProbabilityImprovement.jl")
export ProbabilityImprovement, pi
#include("acquisition/KnowledgeGradient.jl")
include("acquisition/Ensemble_Acq.jl")
export EnsembleAcquisition, ea

# Core Bayesian Optimization framework
include("bayesian_opt.jl")
export optimize, print_info, update!, BOProblem, stop_criteria


include("plotting.jl")
export plot_state

end
