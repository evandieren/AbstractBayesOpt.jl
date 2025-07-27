"""
    StandardGP.jl

Implementation of the Abstract structures for the standard GP.

Reason: This is a simple wrapper around AbstractGPs that implements the AbstractSurrogate abstract type
"""

struct StandardGP <: AbstractSurrogate
    gp::AbstractGPs.GP
    noise_var::Float64
    gpx
end

Base.copy(s::StandardGP) = StandardGP(s.gp, s.noise_var, copy(s.gpx))

function StandardGP(kernel::Kernel,noise_var::Float64;mean=ZeroMean())
    """
    Initialises the model
    """
    gp = AbstractGPs.GP(mean,kernel) # Creates GP(0,k) for the prior
    StandardGP(gp, noise_var, nothing)
end

function update!(model::StandardGP, xs::AbstractVector, ys::AbstractVector)
    gpx = model.gp(xs, model.noise_var...) # This is a FiniteGP with Σy with noise_var on its diagonal.
    updated_gpx = posterior(gpx,reduce(vcat,ys))
    return StandardGP(model.gp, model.noise_var, updated_gpx)
end

# Negative log marginal likelihood (no noise term)
function nlml(mod::StandardGP,params,kernel,x,y;mean=ZeroMean())
    log_ℓ, log_scale = params
    ℓ = exp(log_ℓ)
    scale = exp(log_scale)

    # Kernel with current parameters
    k = scale * (kernel ∘ ScaleTransform(1/ℓ))
    gp = StandardGP(k, mod.noise_var,mean=mean) # Use fixed noise here, or optimize σ² too

    # Evaluate GP at training points with noise, creates a FiniteGP
    gpx = gp.gp(x,mod.noise_var)

    -AbstractGPs.logpdf(gpx, y)
end

function standardize_y(mod::StandardGP,y_train::AbstractVector)
    y_flat = reduce(vcat, y_train)
    y_mean = mean(y_flat)
    std_mean = std(y_flat)
    y_standardized = [(y .- y_mean) ./ std_mean for y in y_train]
    # this re-creates a Vector{Vector{Float64}}, which is what we need
    return y_standardized, y_mean, std_mean
end

get_lengthscale(model::StandardGP) = 1 ./ model.gp.kernel.kernel.transform.s

get_scale(model::StandardGP) = model.gp.kernel.σ²

prep_input(model::StandardGP,x::AbstractVector) = x

posterior_mean(model::StandardGP,x) = Statistics.mean(model.gpx([x]))[1]

posterior_var(model::StandardGP,x) = Statistics.var(model.gpx([x]))[1]

function unstandardized_mean_and_var(gp::StandardGP, X, params::Tuple)
    μ, σ = params
    m, v = mean_and_var(gp.gpx(X))
    # Un-standardize mean and variance
    m_unstd = (m .* σ) .+ μ
    v_unstd = v .* (σ.^2)
    return m_unstd, v_unstd
end