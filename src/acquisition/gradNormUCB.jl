"""
    GradientNormUCB(β)

Acquisition function implementing the Squared 2-norm of the gradient with Upper Confidence Bound (UCB) exploration strategy.


Arguments:
- `β::Float64`: Exploration-exploitation balance parameter

returns:
- `gradUCB::GradientNormUCB`: GradientNormUCB acquisition function instance

References:
    Derived by Van Dieren, E. but open to previous references if existing.
    Originally proposed by [Makrygiorgos et al., 2023](https://www.sciencedirect.com/science/article/pii/S2405896323020487) but adapted to the squared 2-norm of the gradient.
"""
struct GradientNormUCB <: AbstractAcquisition
    β::Float64  # exploration-exploitation balance parameter
end

Base.copy(gradUCB::GradientNormUCB) = GradientNormUCB(gradUCB.β)

function (gradUCB::GradientNormUCB)(surrogate::AbstractSurrogate, x, x_buf=nothing)

    # no buf needed here because we prep everything

    m = posterior_grad_mean(surrogate, x)[2:end]      # Vector{Float64}
    Σ = posterior_grad_cov(surrogate, x)[2:end,2:end]       # Matrix{Float64}

    μ_sqnorm = dot(m,m) + tr(Σ)  # mean of the squared norm of the gradient
    var_sqnorm = 4 * dot(m, Σ * m) + 2 * sum(Σ.^2)  # variance of the squared norm of the gradient

    return -μ_sqnorm + gradUCB.β * sqrt(max(var_sqnorm, 1e-12))
end

"""
    update!(acqf::GradientNormUCB, ys::AbstractVector, surrogate::AbstractSurrogate)

Update the GradientNormUCB acquisition function with new array of observations.

Arguments:
- `acqf::GradientNormUCB`: Current GradientNormUCB acquisition function
- `ys::AbstractVector`: Array of updated observations
- `surrogate::AbstractSurrogate`: Surrogate model

returns:
- `gradUCB::GradientNormUCB`: Updated GradientNormUCB acquisition function
"""
function update(acq::GradientNormUCB, ys::AbstractVector, surrogate::AbstractSurrogate)
    return acq
end