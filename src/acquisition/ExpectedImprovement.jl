"""
    ExpectedImprovement(ξ, best_y)

Expected Improvement acquisition function.

Arguments:
- `ξ::Float64`: Exploration parameter
- `best_y::Float64`: Best observed objective value

returns:
- `EI::ExpectedImprovement`: Expected Improvement acquisition function instance

References:
[Jones et al., 1998](https://link.springer.com/article/10.1023/A:1008306431147)
"""
struct ExpectedImprovement <: AbstractAcquisition
    ξ::Float64
    best_y::Float64
end

function (EI::ExpectedImprovement)(surrogate::AbstractSurrogate, x, x_buf=nothing)

    # Allocate buffer if not provided
    if x_buf === nothing
        if surrogate isa GradientGP
            x_buf = [(copy(x), 1)]
        else
            x_buf = [copy(x)]
        end
    else
        # Reuse buffer
        if surrogate isa GradientGP
            x_buf[1][1] .= x  # copy x into the tuple buffer
        else
            x_buf[1] .= x  # copy into 1×d matrix
        end
    end

    μ = posterior_mean(surrogate, x_buf)
    σ² = posterior_var(surrogate, x_buf)
    Δ = (EI.best_y - EI.ξ) - μ # we are substracting ξ because we are minimising.

    if σ² <= 1e-12
        return max(Δ, 0.0)
    end

    σ = sqrt(σ²)

    z = Δ/σ

    return Δ * cdf(Normal(0, 1), z) + σ * pdf(Normal(0, 1), z)
end


"""
    update!(acqf::ExpectedImprovement, ys::AbstractVector, surrogate::AbstractSurrogate)

Update the Expected Improvement acquisition function with new array of observations.

Arguments:
- `acqf::ExpectedImprovement`: Current Expected Improvement acquisition function
- `ys::AbstractVector`: Array of updated observations
- `surrogate::AbstractSurrogate`: Surrogate model

returns:
- `new_acqf::ExpectedImprovement`: Updated Expected Improvement acquisition function
"""
function update!(acqf::ExpectedImprovement,ys::AbstractVector, surrogate::AbstractSurrogate)
    if (length(ys[1])==1) # we are in 1d
        ExpectedImprovement(acqf.ξ, minimum(reduce(vcat,ys)))
    else 
        ExpectedImprovement(acqf.ξ, minimum(hcat(ys...)[1,:]))
    end
end