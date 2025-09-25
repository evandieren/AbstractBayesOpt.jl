"""
    ExpectedImprovement{Y}(ξ::Y, best_y::Y) <: AbstractAcquisition

Expected Improvement acquisition function.

Attributes:
- `ξ::Y`: Exploration parameter
- `best_y::Y`: Best observed objective value

References:
[Jones et al., 1998](https://link.springer.com/article/10.1023/A:1008306431147)
"""
struct ExpectedImprovement{Y} <: AbstractAcquisition
    ξ::Y
    best_y::Y
end

"""
    Base.copy(EI::ExpectedImprovement)

Creates a copy of the ExpectedImprovement instance.

returns:
- `new_EI::ExpectedImprovement`: A new instance of ExpectedImprovement with copied
"""
Base.copy(EI::ExpectedImprovement) = ExpectedImprovement(EI.ξ, EI.best_y)

"""
    (EI::ExpectedImprovement)(surrogate::AbstractSurrogate, x::AbstractVector)

Evaluate the Expected Improvement acquisition function at a given set of points.

Arguments:
- `surrogate::AbstractSurrogate`: The surrogate model used by the acquisition function.
- `x::AbstractVector`: Vector of input points where the acquisition function is evaluated.

returns:
- `value::AbstractVector`: The evaluated acquisition values at the given points.
"""
function (EI::ExpectedImprovement)(surrogate::AbstractSurrogate, x::AbstractVector)
    μ = posterior_mean(surrogate, x)
    σ² = posterior_var(surrogate, x)
    Δ = (EI.best_y - EI.ξ) .- μ # we are substracting ξ because we are minimising.
    return _single_input_ei.(Δ, σ²)
end

"""
    _single_input_ei(Δ, σ²)

Helper function to compute Expected Improvement for a single input.

Arguments:
- `Δ::Float64`: Difference between best observed value and predicted mean
- `σ²::Float64`: Predicted variance

returns:
- `ei::Float64`: Expected Improvement value
"""
function _single_input_ei(Δ, σ²)
    if σ² <= 1e-12
        return max(Δ, 0.0)
    end
    σ = sqrt(σ²)
    z = Δ / σ
    return Δ * cdf(Normal(0, 1), z) + σ * pdf(Normal(0, 1), z)
end

"""
    update(acq::ExpectedImprovement, ys::AbstractVector, surrogate::AbstractSurrogate)

Update the Expected Improvement acquisition function with new array of observations.

Arguments:
- `acqf::ExpectedImprovement`: Current Expected Improvement acquisition function
- `ys::AbstractVector`: Array of updated observations
- `surrogate::AbstractSurrogate`: Surrogate model

returns:
- `new_acqf::ExpectedImprovement`: Updated Expected Improvement acquisition function
"""
function update(acq::ExpectedImprovement, ys::AbstractVector, surrogate::AbstractSurrogate)
    ExpectedImprovement(acq.ξ, _get_minimum(surrogate, ys))
end
