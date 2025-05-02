#using LinearAlgebra

# --- UCB on the squared norm of the gradient ---
struct GradientNormUCB <: AbstractAcquisition
    β::Float64  # exploration-exploitation balance parameter
end

function (gradUCB::GradientNormUCB)(surrogate::AbstractSurrogate, x)
    m = posterior_grad_mean(surrogate, x)[2:end]      # Vector{Float64}
    Σ = posterior_grad_cov(surrogate, x)[2:end,2:end]       # Matrix{Float64}

    μ_sqnorm = m'*m # not taking f(x)
    σ_sqnorm = 2 * m'*Σ*m + tr(Σ * Σ)    # Approximate variance of ||∇μ(x)||²

    return -μ_sqnorm + gradUCB.β * sqrt(max(σ_sqnorm, 1e-12))
end

function update!(acqf::GradientNormUCB, ys::AbstractVector, surrogate::AbstractSurrogate)
    return acqf
end