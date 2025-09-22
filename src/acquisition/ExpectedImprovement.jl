"""
Expected Improvement acquisition function.

Arguments:
- `ξ::Float64`: Exploration parameter
- `best_y::Float64`: Best observed objective value

returns:
- `EI::ExpectedImprovement`: Expected Improvement acquisition function instance

References:
[Jones et al., 1998](https://link.springer.com/article/10.1023/A:1008306431147)
"""
struct ExpectedImprovement{Y} <: AbstractAcquisition
    ξ::Y
    best_y::Y
end

Base.copy(EI::ExpectedImprovement) = ExpectedImprovement(EI.ξ, EI.best_y)

function (EI::ExpectedImprovement)(surrogate::AbstractSurrogate, x::AbstractVector)

    μ = posterior_mean(surrogate, x)
    σ² = posterior_var(surrogate, x)
    Δ = (EI.best_y - EI.ξ) .- μ # we are substracting ξ because we are minimising.
    return _single_input_ei.(Δ, σ²)
end

function _single_input_ei(Δ, σ²)
    if σ² <= 1e-12
        return max(Δ, 0.0)
    end
    σ = sqrt(σ²)
    z = Δ / σ
    return Δ * cdf(Normal(0, 1), z) + σ * pdf(Normal(0, 1), z)
end

"""
Update the Expected Improvement acquisition function with new array of observations.

Arguments:
- `acqf::ExpectedImprovement`: Current Expected Improvement acquisition function
- `ys::AbstractVector`: Array of updated observations
- `surrogate::AbstractSurrogate`: Surrogate model

returns:
- `new_acqf::ExpectedImprovement`: Updated Expected Improvement acquisition function
"""
function update(acq::ExpectedImprovement, ys::AbstractVector, surrogate::AbstractSurrogate)
    if (length(ys[1]) == 1) # we are in 1d
        ExpectedImprovement(acq.ξ, minimum(reduce(vcat, ys)))
    else
        ExpectedImprovement(acq.ξ, minimum(hcat(ys...)[1, :]))
    end
end
