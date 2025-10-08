## Utility functions for Bayesian Optimization

## BOStruct and related functions

function _make_info(BO::BOStruct)
    info = "== BOStruct Information ==\n"
    info *= "Target function: $(BO.func)\n"
    info *= "Domain: $(BO.domain)\n"
    info *= "Number of data points: $(length(BO.xs))\n"
    # info *= "Surrogate model: $(BO.model)\n"
    info *= "Acquisition function: $(BO.acq)\n"
    info *= "Max iterations: $(BO.max_iter)\n"
    info *= "Noise level: $(BO.noise)\n"
    info *= "========================="
    return info
end

function print_info(BO::BOStruct)
    return println(_make_info(BO))
end

function Base.show(io::IO, BO::BOStruct)
    return print(io, _make_info(BO))
end

## Standardization functions

"""
    standardize_problem(BO::BOStruct, choice::String)

Standardize the output values of the BOStruct and update the GP and acquisition function accordingly.

Arguments:
- `BO::BOStruct`: The Bayesian Optimization problem to standardize.
- `choice::String`: Standardization mode:
    - "mean_scale": remove empirical mean and scale by empirical std.
    - "scale_only": only scale by empirical std (no centering). If the GP has a non-zero prior mean, it is rescaled accordingly for consistency.
    - "mean_only": only remove the empirical mean (no scaling).

returns:
- `BO::BOStruct`: The updated BOStruct with standardized outputs and updated model/acquisition.
- `params::Tuple`: A tuple containing the mean and standard deviation used for standardization (vectors matching the output dimension).
"""
function standardize_problem(BO::BOStruct, choice::String)
    @argcheck choice in ["mean_scale", "scale_only", "mean_only"] "choice must be one of: 'mean_scale', 'scale_only', 'mean_only'"

    # Attention: here it is the standard deviation, need to square for kernel scaling
    μ, σ = get_mean_std(BO.model, BO.ys_non_std, choice) # μ should be the type of Y

    @info "Standardization choice: $choice"
    @info "Standardization parameters: μ=$μ, σ=$σ"

    # Need to update original kernel scale if scale_only or mean_scale
    if choice in ["scale_only", "mean_scale"]
        BO.model = rescale_model(BO.model, σ) # Update the kernel scale / σ² and mean by σ
    end

    # Need to standardize the outputs too:
    BO.ys = std_y(BO.model, BO.ys_non_std, μ, σ)
    BO.model = update(BO.model, BO.xs, BO.ys)
    BO.acq = update(BO.acq, BO.ys, BO.model)

    return BO, (μ, σ)
end

"""
    lengthscale_bounds(
        X_train::AbstractVector,
        domain_lower::AbstractVector,
        domain_upper::AbstractVector;
        min_frac::Float64=0.1,
        max_frac::Float64=1.0,
    )

Compute sensible per-dimension lower and upper bounds for GP kernel lengthscales
using approximate fill distances that include domain boundaries.

Arguments:
- `X_train`: vector of n points
- `domain_lower`: vector of length d, lower bounds of domain.
- `domain_upper`: vector of length d, upper bounds of domain.
- `min_frac`: fraction of fill distance for minimum lengthscale. defaults to 0.1.
- `max_frac`: fraction of domain size for maximum lengthscale. defaults to 1.0.

Returns:
- `(ℓ_lower, ℓ_upper)` — vectors of length d.
"""
function lengthscale_bounds(
    X_train::AbstractVector,
    domain_lower::AbstractVector,
    domain_upper::AbstractVector;
    min_frac::Float64=0.1,
    max_frac::Float64=1.0,
    n_samples::Int=10_000,
)
    n = length(X_train)
    d = length(domain_lower)

    # if d = 1, X_train will be a vector of scalars, otherwise a vector of vectors
    if d > 1
        @assert all(length(x) == d for x in X_train) """
                                                All points in X_train must have 
                                                dimension $d
                                                """
    end

    # ℓ_upper is a fraction of the domain diameter
    ℓ_upper = max_frac .* (domain_upper .- domain_lower)

    if d > 1
        # Multidimensional Monte Carlo fill distance
        h_fill = monte_carlo_fill_distance(
            X_train, domain_lower, domain_upper; n_samples=n_samples
        )
        ℓ_lower = fill(max(min_frac * h_fill, 1e-12), d)
    else
        # 1D axis-wise computation (including domain edges)
        @assert length(X_train) == n "X_train must be a vector of scalars for d=1"

        sort!(X_train)
        gaps = diff([domain_lower[1]; X_train; domain_upper[1]])
        h_fill = maximum(gaps)
        ℓ_lower = [max(min_frac * h_fill, 1e-12)]
    end

    return ℓ_lower, ℓ_upper
end

"""
    monte_carlo_fill_distance(X_train, domain_lower, domain_upper;
                              n_samples=10_000)

Estimate the fill distance h_X,D = sup_{x in D} min_{x_j in X} ||x - x_j|| 
using Monte Carlo sampling. Used for d>1.

Arguments:
- `X_train`: vector of points (each a vector of length d)
- `domain_lower`, `domain_upper`: vectors of length d
- `n_samples`: number of random points to sample

returns
- `h_max`: estimated fill distance
"""
function monte_carlo_fill_distance(
    X_train::AbstractVector{<:AbstractVector},
    domain_lower::AbstractVector,
    domain_upper::AbstractVector;
    n_samples::Int=10_000,
)
    d = length(domain_lower)

    # convert training points to d × n matrix
    X = hcat(X_train...)

    h_max = 0.0
    for _ in 1:n_samples
        x_sample = domain_lower .+ rand(d) .* (domain_upper .- domain_lower)
        dists = sqrt.(sum((X .- x_sample) .^ 2; dims=1))
        min_dist = minimum(dists)
        if min_dist > h_max
            h_max = min_dist
        end
    end

    return h_max
end

"""
    rescale_output(ys::AbstractVector, params::Tuple)

Rescale the standardized output values back to the original scale.

Arguments:
- `ys::AbstractVector`: A vector of standardized function values.
- `params::Tuple`: A tuple containing the mean and standard deviation used 
for standardization.

returns:
- `ys_rescaled`: A vector of rescaled function values.
"""
function rescale_output(ys::AbstractVector, params::Tuple)
    μ, σ = params

    if isnothing(μ) || isnothing(σ)
        return [y for y in ys]
    else
        return [(y .* σ) .+ μ for y in ys]
    end
end

"""
    _noise_like(y::AbstractFloat; σ=1.0)

Generate Gaussian noise with standard deviation `σ` for a single float output.

Arguments:
- `y::AbstractFloat`: A single float output value.
- `σ::Float64`: Standard deviation of the noise (default is 1.0).

returns:
- `noise::Float64`: A random noise value drawn from a normal distribution with mean.
"""
_noise_like(y::AbstractFloat; σ=1.0) = σ * randn()

"""
    _noise_like(y::AbstractVector{T}; σ=1.0) where {T<:AbstractFloat}

Generate Gaussian noise with standard deviation `σ` for a vector of float outputs.

Arguments:
- `y::AbstractVector{T}`: A vector of float output values.
- `σ::Float64`: Standard deviation of the noise (default is 1.0

returns:
- `noise::Vector{T}`: A vector of random noise values drawn from a normal 
distribution with mean.
"""
_noise_like(y::AbstractVector{T}; σ=1.0) where {T<:AbstractFloat} = σ * randn(length(y))

"""
    _noise_like(y::AbstractVector{T}; σ::AbstractVector{T}) where {T}

Generate Gaussian noise with per-dimension standard deviations for a single float output.

Arguments:
- `y::AbstractVector{T}`: One output (vector of type T).
- `σ::AbstractVector{T}`: A vector of standard deviations for each dimension. 

returns:
- `noise::Vector{T}`: A vector of random noise values drawn from a normal distribution 
with mean.
"""
_noise_like(y::AbstractVector{T}; σ::AbstractVector{T}) where {T} = σ .* randn(length(y))
