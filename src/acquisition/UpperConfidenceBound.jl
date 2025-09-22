"""
Upper Confidence Bound (UCB) acquisition function.

Arguments:
- `β::Float64`: Exploration-exploitation balance parameter

returns:
- `UCB::UpperConfidenceBound`: Upper Confidence Bound acquisition function instance

References:
[Srinivas et al., 2012](https://ieeexplore.ieee.org/document/6138914)
"""
struct UpperConfidenceBound <: AbstractAcquisition
    β::Float64  # exploration-exploitation balance parameter
end

Base.copy(UCB::UpperConfidenceBound) = UpperConfidenceBound(UCB.β)

function (UCB::UpperConfidenceBound)(surrogate::AbstractSurrogate, x::AbstractVector)
    μ = posterior_mean(surrogate, x)
    σ² = posterior_var(surrogate, x)

    σ²_safe = max.(σ², 0.0)  # Ensure non-negative variance

    return -μ .+ UCB.β .* sqrt.(σ²_safe)
end

"""
Update the Upper Confidence Bound acquisition function with new array of observations.

Arguments:
- `acqf::UpperConfidenceBound`: Current Upper Confidence Bound acquisition function
- `ys::AbstractVector`: Array of updated observations
- `surrogate::AbstractSurrogate`: Surrogate model

returns:
- `UCB::UpperConfidenceBound`: Updated Upper Confidence Bound acquisition function
"""
function update(acq::UpperConfidenceBound, ys::AbstractVector, surrogate::AbstractSurrogate)
    acq
end
