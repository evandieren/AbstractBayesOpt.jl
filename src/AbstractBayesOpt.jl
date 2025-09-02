module AbstractBayesOpt

using AbstractGPs, KernelFunctions
using ForwardDiff
using Optim
using Statistics
using SpecialFunctions
using Plots
using Distributions
using LinearAlgebra
using Random
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
export StandardGP, prep_input, posterior_mean, posterior_var, nlml,nlml_ls, get_mean_std, rescale_y ,unstandardized_mean_and_var, get_lengthscale, get_scale

include("surrogates/GradientGP.jl")
export GradientGP, ApproxMatern52Kernel, gradConstMean, gradKernel, prep_input, posterior_mean, posterior_var, posterior_grad_mean, posterior_grad_cov,posterior_grad_var, nlml, standardize_y,unstandardized_mean_and_var, get_lengthscale, get_scale

# Acquisition functions
include("acquisition/acq_utils.jl")
export normcdf, normpdf,optimize_mean!,optimize_acquisition!#, sample_gp_function
include("acquisition/ExpectedImprovement.jl")
export ExpectedImprovement
include("acquisition/UpperConfidenceBound.jl")
export UpperConfidenceBound
include("acquisition/gradNormUCB.jl")
export GradientNormUCB
include("acquisition/ProbabilityImprovement.jl")
export ProbabilityImprovement
#include("acquisition/KnowledgeGradient.jl")
# export KnowledgeGradient, KG
include("acquisition/EnsembleAcq.jl")
export EnsembleAcquisition

# include("acquisition/ThompsonSampling.jl")
# export ThompsonSampling, TS

# Core Bayesian Optimization framework
include("bayesian_opt.jl")
export optimize, print_info, update!, BOStruct, stop_criteria, optimize_hyperparameters, standardize_problem, rescale_output


# include("plotting.jl")
# export plot_state

end
