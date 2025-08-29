using AbstractBayesOpt
using Test
using AbstractGPs, KernelFunctions
using ForwardDiff
using Optim
using FillArrays
using Statistics
using SpecialFunctions
using Distributions
using LinearAlgebra
using Random

# Set random seed for reproducible tests
Random.seed!(42)

@testset "AbstractBayesOpt.jl" begin
    include("test_domains.jl")
    include("test_surrogates.jl")
    include("test_acquisition.jl")
    include("test_bayesian_opt.jl")
end
