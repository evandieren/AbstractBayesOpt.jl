"""
    UpperConfidenceBound{Y}(β::Y) <: AbstractAcquisition

Upper Confidence Bound (UCB) acquisition function.

Attributes:
- `β::Y`: Exploration-exploitation balance parameter

References:
[Srinivas et al., 2012](https://ieeexplore.ieee.org/document/6138914)
"""
struct UpperConfidenceBound{Y} <: AbstractAcquisition
    β::Y  # exploration-exploitation balance parameter
end


"""
    Base.copy(UCB::UpperConfidenceBound)

Creates a copy of the UpperConfidenceBound instance.

returns:
- `new_UCB::UpperConfidenceBound`: A new instance of UpperConfidenceBound.
"""
Base.copy(UCB::UpperConfidenceBound) = UpperConfidenceBound(UCB.β)


"""
    (UCB::UpperConfidenceBound)(surrogate::AbstractSurrogate, x::AbstractVector)

Evaluate the Upper Confidence Bound acquisition function at a given set of points.

Arguments:
- `surrogate::AbstractSurrogate`: The surrogate model used by the acquisition function.
- `x::AbstractVector`: Vector of input points where the acquisition function is evaluated.

returns:
- `value::AbstractVector`: The evaluated acquisition values at the given points.
"""
function (UCB::UpperConfidenceBound)(surrogate::AbstractSurrogate, x::AbstractVector)
    μ = posterior_mean(surrogate, x)
    σ² = posterior_var(surrogate, x)

    σ²_safe = max.(σ², 0.0)  # Ensure non-negative variance

    return -μ .+ UCB.β .* sqrt.(σ²_safe)
end

"""
    update(acq::UpperConfidenceBound, ys::AbstractVector, surrogate::AbstractSurrogate)

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
