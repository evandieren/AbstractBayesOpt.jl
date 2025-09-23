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
using ArgCheck

# Interface definitions
include("abstract.jl")
export AbstractAcquisition, AbstractSurrogate, AbstractDomain

# Domain implementations
include("domains/ContinuousDomain.jl")
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
include("acquisition_functions/acq_utils.jl")
export optimize_acquisition

include("acquisition_functions/ExpectedImprovement.jl")
export ExpectedImprovement

include("acquisition_functions/UpperConfidenceBound.jl")
export UpperConfidenceBound

include("acquisition_functions/gradNormUCB.jl")
export GradientNormUCB

include("acquisition_functions/ProbabilityImprovement.jl")
export ProbabilityImprovement

include("acquisition_functions/EnsembleAcq.jl")
export EnsembleAcquisition

# Core Bayesian Optimization framework
include("bayesian_opt.jl")
export optimize, update, BOStruct, optimize_hyperparameters

# Utility functions
include("BO_utils.jl")
export print_info, rescale_output, standardize_problem

end
