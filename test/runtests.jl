using Pkg
Pkg.activate(dirname(@__DIR__))

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
    rescale_output,
    lengthscale_bounds,
    _unitvec,
    _scalar_mean,
    _build_noise,
    _prep_input,
    prep_output,
    make_linear_operator_mean,
    get_kernel_constructor


using Test
using AbstractGPs
using ForwardDiff
using Random

# Set random seed for reproducible tests
Random.seed!(42)

@testset "AbstractBayesOpt.jl" begin
    include("test_domains.jl")
    include("test_surrogates.jl")
    include("test_kernels.jl")
    include("test_acquisition.jl")
    include("test_bayesian_opt.jl")
end
