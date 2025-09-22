module AbstractBayesOpt

using AbstractGPs
using KernelFunctions
using ForwardDiff
using Optim
using Statistics
using SpecialFunctions
using Distributions
using LinearAlgebra
using Random
using Distances
using DifferentiationInterface

# Interface definitions
include("abstract.jl")
export AbstractAcquisition, AbstractSurrogate, AbstractDomain

# Domain implementations
include("domains.jl")
export ContinuousDomain

# Surrogate models
include("surrogates/surrogates_utils.jl")
include("surrogates/StandardGP.jl")
export StandardGP,
    prep_input,
    posterior_mean,
    posterior_var,
    nlml,
    nlml_ls,
    get_mean_std,
    std_y,
    unstandardized_mean_and_var,
    get_lengthscale,
    get_scale,
    get_kernel_constructor,
    rescale_model

include("surrogates/GradientGP.jl")
export GradientGP,
    ApproxMatern52Kernel,
    gradConstMean,
    gradKernel,
    prep_input,
    posterior_mean,
    posterior_var,
    posterior_grad_mean,
    posterior_grad_cov,
    posterior_grad_var,
    nlml,
    get_mean_std,
    std_y,
    unstandardized_mean_and_var,
    get_lengthscale,
    get_scale,
    get_kernel_constructor,
    rescale_model

# Acquisition functions
include("acquisition/acq_utils.jl")
export normcdf, normpdf, optimize_acquisition
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
export optimize, update, BOStruct, stop_criteria, optimize_hyperparameters

# Utility functions
include("BO_utils.jl")
export print_info, rescale_output, standardize_problem, lengthscale_bounds

# include("plotting.jl")
# export plot_state

end
