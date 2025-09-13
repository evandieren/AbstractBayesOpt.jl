## Utility functions for Bayesian Optimization



## BOStruct and related functions
"""
    Prints information about the BOStruct

Arguments:
- `BO::BOStruct`: The BOStruct instance to print information about.

returns:
- Nothing, just prints to the console.
"""
function print_info(BO::BOStruct)
    println("== Printing information about the BOStruct ==")
    println("Target function: ",BO.func)
    println("Domain: ",BO.domain)
    println("xs: ",BO.xs)
    println("ys: ",BO.ys)
    println("Surrogate: ",BO.model)
    println("ACQ: ",BO.acq)
    println("max_iter: ",BO.max_iter)
    println("noise: ",BO.noise)
    println("=============================================")
end


## Standardization functions


"""
    standardize_problem(BO::BOStruct; choice="mean_scale")

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
function standardize_problem(BO::BOStruct; choice="mean_scale")
    @assert choice in ["mean_scale", "scale_only", "mean_only"] "choice must be one of: 'mean_scale', 'scale_only', 'mean_only'"

    ys_non_std = BO.ys_non_std
    p = length(BO.ys[1])

    # Attention: here it is the standard deviation, need to square for kernel scaling
    μ::Vector{Float64}, σ::Vector{Float64} = get_mean_std(BO.model,ys_non_std)

    # Taking into account the choice of the user
    if choice == "scale_only"
        μ = zeros(p)
    elseif choice == "mean_only"
        σ = ones(p)
    end

    println("Standardization choice: $choice")
    println("Standardization parameters: μ=$μ, σ=$σ")


    # Need to update original kernel scale if scale_only or mean_scale
    if choice in ["scale_only", "mean_scale"]
        BO.model = rescale_model(BO.model, σ) # Update the kernel scale / σ² and mean by σ
    end 

    # Need to standardize the outputs too:
    BO.ys = std_y(BO.model, ys_non_std,μ, σ)
    BO.model = update(BO.model, BO.xs, BO.ys)
    BO.acq = update(BO.acq, BO.ys, BO.model)
    
    return BO, (μ, σ)
end


"""
Compute sensible per-dimension lower and upper bounds for GP kernel lengthscales using
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
function lengthscale_bounds(x_train::AbstractMatrix, domain_lower::AbstractVector, domain_upper::AbstractVector;
                            min_frac::Float64=0.1, max_frac::Float64=1.0)

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
                    if k == j; continue; end
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
function lengthscale_bounds(x_train::AbstractVector{<:AbstractVector}, domain::ContinuousDomain;
                            min_frac::Float64=0.1, max_frac::Float64=1.0)
    X = permutedims(reduce(hcat, x_train)) # n × d
    return lengthscale_bounds(X, domain.lower, domain.upper; min_frac=min_frac, max_frac=max_frac)
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