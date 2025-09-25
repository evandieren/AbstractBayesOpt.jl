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
        x_train::AbstractMatrix,
        domain_lower::AbstractVector,
        domain_upper::AbstractVector;
        min_frac::Float64=0.1,
        max_frac::Float64=1.0,
    )

Computes sensible per-dimension lower and upper bounds for GP kernel lengthscales using
nearest-neighbor fill distances and domain extents.

Arguments:
- `x_train`: n×d matrix of training points (rows = points, columns = dimensions)
- `domain_lower`: vector of length d, lower bound of domain per dimension
- `domain_upper`: vector of length d, upper bound of domain per dimension
- `min_frac`: fraction of fill distance for minimum lengthscale (default 0.05)
- `max_frac`: fraction of domain size for maximum lengthscale (default 2.0)

Returns:
- `(ℓ_lower, ℓ_upper)` vectors of length d suitable for setting log-space bounds.
"""
function lengthscale_bounds(
    x_train::AbstractMatrix,
    domain_lower::AbstractVector,
    domain_upper::AbstractVector;
    min_frac::Float64=0.1,
    max_frac::Float64=1.0,
)
    n, d = size(x_train)
    ℓ_lower = zeros(d)
    ℓ_upper = zeros(d)

    for i in 1:d
        xi = view(x_train, :, i)
        domain_size = domain_upper[i] - domain_lower[i]

        # approximate fill distance along this dimension: max nearest-neighbor distance
        if n < 2
            h_i = domain_size
        else
            max_min_dist = 0.0
            for j in 1:n
                min_d = Inf
                xj = xi[j]
                for k in 1:n
                    if k == j
                        continue
                    end
                    dk = abs(xj - xi[k])
                    if dk < min_d
                        min_d = dk
                    end
                end
                if min_d > max_min_dist
                    max_min_dist = min_d
                end
            end
            h_i = max_min_dist
        end

        ℓ_lower[i] = max(min_frac * h_i, 1e-12)
        ℓ_upper[i] = max_frac * domain_size
    end

    return ℓ_lower, ℓ_upper
end

"""
    lengthscale_bounds(x_train::AbstractVector{<:AbstractVector}, domain::ContinuousDomain;
                       min_frac::Float64=0.05, max_frac::Float64=2.0)

Convenience overload accepting a vector-of-vectors of points and a `ContinuousDomain`.
Returns the same as the matrix method.
"""
function lengthscale_bounds(
    x_train::AbstractVector,
    domain::ContinuousDomain;
    min_frac::Float64=0.1,
    max_frac::Float64=1.0,
)
    X = permutedims(reduce(hcat, x_train)) # n × d
    return lengthscale_bounds(
        X, domain.lower, domain.upper; min_frac=min_frac, max_frac=max_frac
    )
end

"""
    rescale_output(ys::AbstractVector, params::Tuple)

Rescale the standardized output values back to the original scale.

Arguments:
- `ys::AbstractVector`: A vector of standardized function values.
- `params::Tuple`: A tuple containing the mean and standard deviation used for standardization.

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
- `noise::Vector{T}`: A vector of random noise values drawn from a normal distribution with mean.
"""
_noise_like(y::AbstractVector{T}; σ=1.0) where {T<:AbstractFloat} = σ * randn(length(y))

"""
    _noise_like(y::AbstractVector{T}; σ::AbstractVector{T}) where {T}

Generate Gaussian noise with per-dimension standard deviations for a single float output.

Arguments:
- `y::AbstractVector{T}`: One output (vector of type T).
- `σ::AbstractVector{T}`: A vector of standard deviations for each dimension. (size must match output dimension)

returns:
- `noise::Vector{T}`: A vector of random noise values drawn from a normal distribution with mean.
"""
_noise_like(y::AbstractVector{T}; σ::AbstractVector{T}) where {T} = σ .* randn(length(y))
