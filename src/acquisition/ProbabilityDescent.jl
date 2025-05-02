struct ProbabilityDescent <: AbstractAcquisition
    # (Optionally) step size τ, but can be omitted if absorbed
end

function (pd::ProbabilityDescent)(surrogate::AbstractSurrogate, x)
    # Obtain posterior gradient mean and covariance at x
    m = posterior_grad_mean(surrogate, x)      # Vector{Float64}
    Σ = posterior_grad_covar(surrogate, x)       # Matrix{Float64}

    # Compute scalar quantities
    num = m'*m
    denom2 = m'*Σ*m
    if denom2 <= 0
        # If no uncertainty, descent is certain if gradient non-zero
        return num > 0 ? 1.0 : 0.0
    end
    z = - num / sqrt(denom2)
    return cdf(Normal(), z)
end

function update!(acqf::ProbabilityDescent,ys::AbstractVector, surrogate::AbstractSurrogate)
    # No internal state to update for PD
    return acqf
end