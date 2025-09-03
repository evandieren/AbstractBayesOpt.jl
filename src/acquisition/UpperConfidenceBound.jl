"""
    UpperConfidenceBound(β::Float64)

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

function (UCB::UpperConfidenceBound)(surrogate::AbstractSurrogate, x, x_buf=nothing)

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
    return -μ + UCB.β*sqrt.(σ²)
end


"""
    update!(acqf::UpperConfidenceBound, ys::AbstractVector, surrogate::AbstractSurrogate)

Update the Upper Confidence Bound acquisition function with new array of observations.

Arguments:
- `acqf::UpperConfidenceBound`: Current Upper Confidence Bound acquisition function
- `ys::AbstractVector`: Array of updated observations
- `surrogate::AbstractSurrogate`: Surrogate model

returns:
- `UCB::UpperConfidenceBound`: Updated Upper Confidence Bound acquisition function
"""
function update!(acqf::UpperConfidenceBound,ys::AbstractVector, surrogate::AbstractSurrogate)
    acqf
end