"""
    GradientNormUCB{Y}(β::Y) <: AbstractAcquisition

Acquisition function implementing the Squared 2-norm of the gradient with Upper Confidence Bound (UCB) exploration strategy.

Attributes:
- `β::Y`: Exploration-exploitation balance parameter

References:
    Derived by Van Dieren, E. but open to previous references if existing.
    Originally proposed by [Makrygiorgos et al., 2023](https://www.sciencedirect.com/science/article/pii/S2405896323020487) but adapted to the squared 2-norm of the gradient.
"""
struct GradientNormUCB{Y} <: AbstractAcquisition
    β::Y # exploration-exploitation balance parameter
end


"""
    Base.copy(gradUCB::GradientNormUCB)

Creates a copy of the GradientNormUCB instance.

returns:
- `new_gradUCB::GradientNormUCB`: A new instance of GradientNormUCB
"""
Base.copy(gradUCB::GradientNormUCB) = GradientNormUCB(gradUCB.β)


"""
    (gradUCB::GradientNormUCB)(surrogate::AbstractSurrogate, x::AbstractVector)

Evaluate the GradientNormUCB acquisition function at a given set of points.

Arguments:
- `surrogate::AbstractSurrogate`: The surrogate model used by the acquisition function. Must be a GradientGP.
- `x::AbstractVector`: Vector of input points where the acquisition function is evaluated.

returns:
- `value::AbstractVector`: The evaluated acquisition values at the given points.
"""
function (gradUCB::GradientNormUCB)(surrogate::AbstractSurrogate, x::AbstractVector)
    return _single_input_gradUCB.(Ref(gradUCB), Ref(surrogate), x)
end

function _single_input_gradUCB(gradUCB::GradientNormUCB, surrogate::GradientGP, x)
    
    m = posterior_grad_mean(surrogate, [x])[2:end]      # Vector{Float64}
    Σ = posterior_grad_cov(surrogate, [x])[2:end, 2:end]       # Matrix{Float64}

    μ_sqnorm = dot(m, m) + tr(Σ)  # mean of the squared norm of the gradient
    var_sqnorm = 4 * dot(m, Σ * m) + 2 * sum(Σ .^ 2)  # variance of the squared norm of the gradient
    
    return -μ_sqnorm + gradUCB.β * sqrt(max(var_sqnorm, 1e-12))
end

"""
    update(acqf::GradientNormUCB, ys::AbstractVector, surrogate::AbstractSurrogate)

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
