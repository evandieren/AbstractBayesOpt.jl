#using LinearAlgebra

# --- UCB on the squared norm of the gradient ---
struct GradientNormUCB <: AbstractAcquisition
    β::Float64  # exploration-exploitation balance parameter
end

function (gradUCB::GradientNormUCB)(surrogate::AbstractSurrogate, x, x_buf=nothing)

    # Allocate buffer if not provided
    if x_buf === nothing
        x_buf = reshape(x, 1, :)   # create 1×d buffer
    else
        x_buf[1, :] .= x           # reuse existing buffer
    end

    m = posterior_grad_mean(surrogate, x)[2:end]      # Vector{Float64}
    Σ = posterior_grad_cov(surrogate, x)[2:end,2:end]       # Matrix{Float64}

    μ_sqnorm = dot(m,m) + tr(Σ)  # mean of the squared norm of the gradient
    var_sqnorm = 4 * dot(m, Σ * m) + 2 * sum(Σ.^2)  # variance of the squared norm of the gradient

    return -μ_sqnorm + gradUCB.β * sqrt(max(var_sqnorm, 1e-12))
end

function update!(acqf::GradientNormUCB, ys::AbstractVector, surrogate::AbstractSurrogate)
    return acqf
end