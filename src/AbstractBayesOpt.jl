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
using QuasiMonteCarlo
using ArgCheck

# Interface definitions
include("abstract.jl")
export AbstractAcquisition, AbstractSurrogate, AbstractDomain

# Surrogate models
include("surrogates/StandardGP.jl")
include("surrogates/GradientGP.jl")
include("surrogates/surrogates_utils.jl")

## Models
export StandardGP, GradientGP

## Methods
export posterior_mean, posterior_var, nlml
export posterior_grad_mean, posterior_grad_var, posterior_grad_cov # gradient-enhanced GP methods

## Safe AD Matern 5/2
export ApproxMatern52Kernel

## Gradient kernel and mean functions
export gradConstMean, gradKernel

## Surrogate utils
export prep_input, unstandardized_mean_and_var

# Acquisition functions
include("acquisition_functions/ExpectedImprovement.jl")
include("acquisition_functions/UpperConfidenceBound.jl")
include("acquisition_functions/ProbabilityImprovement.jl")
include("acquisition_functions/gradNormUCB.jl")
include("acquisition_functions/EnsembleAcq.jl")
include("acquisition_functions/acq_utils.jl")

export ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityImprovement,
    GradientNormUCB,
    EnsembleAcquisition,
    optimize_acquisition

# Domain definitions
include("domains/ContinuousDomain.jl")
export ContinuousDomain

# Core Bayesian Optimization framework
include("bayesian_opt.jl")
export BOStruct
export optimize

# Utility functions
include("BO_utils.jl")

end
