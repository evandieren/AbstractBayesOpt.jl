"""
Probability of Improvement acquisition function.

Arguments:
- `ξ::Float64`: Exploration parameter
- `best_y::Float64`: Best observed objective value

returns:
- `PI::ProbabilityImprovement`: Probability of Improvement acquisition function instance

References:
[Kushner, 1964](https://asmedigitalcollection.asme.org/fluidsengineering/article/86/1/97/392213/A-New-Method-of-Locating-the-Maximum-Point-of-an)
"""
struct ProbabilityImprovement <: AbstractAcquisition
    ξ::Float64
    best_y::Float64
end

Base.copy(PI::ProbabilityImprovement) = ProbabilityImprovement(PI.ξ, PI.best_y)

function (PI::ProbabilityImprovement)(surrogate::AbstractSurrogate, x::AbstractVector)
    μ = posterior_mean(surrogate, x)
    σ² = posterior_var(surrogate, x)
    Δ = (PI.best_y - PI.ξ) .- μ # we are substracting ξ because we are minimising.
    return _single_input_pi.(Δ, σ²)

    # max(σ², 0) == 0 && return max(Δ, 0.0)

    # σ = sqrt(σ²)
    # @evandieren was z not useful or is that a typo?
    # z = (PI.best_y .- μ) ./ σ
    # return normcdf(Δ / σ, 1)
end

function _single_input_pi(Δ, σ²)
    if σ² <= 0
        return max(Δ, 0.0)
    end
    σ = sqrt(σ²)
    return normcdf(Δ / σ, 1)
end

"""
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
    if isa(ys[1], Float64) # we are in 1d
        ProbabilityImprovement(acq.ξ, minimum(reduce(vcat, ys)))
    else
        ProbabilityImprovement(acq.ξ, minimum(hcat(ys...)[1, :]))
    end
end
