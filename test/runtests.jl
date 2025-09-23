using AbstractBayesOpt

using AbstractBayesOpt:
    update,
    optimize_hyperparameters,
    get_lengthscale,
    get_scale,
    std_y,
    standardize_problem,
    get_mean_std,
    print_info,
    rescale_output

using Test
using AbstractGPs
using ForwardDiff
using Random

# Set random seed for reproducible tests
Random.seed!(42)

@testset "AbstractBayesOpt.jl" begin
    include("test_domains.jl")
    include("test_surrogates.jl")
    include("test_acquisition.jl")
    include("test_bayesian_opt.jl")
end
