"""
    ProbabilityImprovement{Y}(ξ::Y, best_y::Y) <: AbstractAcquisition

Attributes:
- `ξ::Y`: Exploration parameter
- `best_y::Y`: Best observed objective value

References:
[Kushner, 1964](https://asmedigitalcollection.asme.org/fluidsengineering/article/86/1/97/392213/A-New-Method-of-Locating-the-Maximum-Point-of-an)
"""
struct ProbabilityImprovement{Y} <: AbstractAcquisition
    ξ::Y
    best_y::Y
end

"""
    Base.copy(PI::ProbabilityImprovement)

Creates a copy of the ProbabilityImprovement instance.

returns:
- `new_PI::ProbabilityImprovement`: A new instance of ProbabilityImprovement with copied parameters.
"""
Base.copy(PI::ProbabilityImprovement) = ProbabilityImprovement(PI.ξ, PI.best_y)


"""
    (PI::ProbabilityImprovement)(surrogate::AbstractSurrogate, x::AbstractVector)

Evaluate the Probability of Improvement acquisition function at a given set of points.

Arguments:
- `surrogate::AbstractSurrogate`: The surrogate model used by the acquisition function.
- `x::AbstractVector`: Vector of input points where the acquisition function is evaluated.

returns:
- `value::AbstractVector`: The evaluated acquisition values at the given points.
"""
function (PI::ProbabilityImprovement)(surrogate::AbstractSurrogate, x::AbstractVector)
    μ = posterior_mean(surrogate, x)
    σ² = posterior_var(surrogate, x)
    Δ = (PI.best_y - PI.ξ) .- μ # we are substracting ξ because we are minimising.
    return _single_input_pi.(Δ, σ²)
end

"""
    _single_input_pi(Δ, σ²)

Helper function to compute Probability of Improvement for a single input.

Arguments:
- `Δ`: Difference between best observed value and predicted mean
- `σ²`: Predicted variance

returns:
- `pi`: Probability of Improvement at the given input.
"""
function _single_input_pi(Δ, σ²)
    if σ² <= 1e-12
        return max(Δ, 0.0)
    end
    σ = sqrt(σ²)
    return cdf(Normal(0, 1), Δ / σ)
end

"""
    update(acq::ProbabilityImprovement, ys::AbstractVector, surrogate::AbstractSurrogate)

Update the Probability of Improvement acquisition function with new array of observations.

Arguments:
- `acqf::ProbabilityImprovement`: Current Probability of Improvement acquisition function
- `ys::AbstractVector`: Array of updated observations
- `surrogate::AbstractSurrogate`: Surrogate model

returns:
- `PI::ProbabilityImprovement`: Updated Probability of Improvement acquisition function
"""
function update(
    acq::ProbabilityImprovement, ys::AbstractVector, surrogate::AbstractSurrogate
)
    return ProbabilityImprovement(acq.ξ, _get_minimum(surrogate, ys))
end
