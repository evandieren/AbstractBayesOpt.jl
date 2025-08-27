"""
    StandardGP.jl

Implementation of the Abstract structures for the standard GP.

Remark: this is a simple wrapper around AbstractGPs that implements the AbstractSurrogate abstract type
"""

struct StandardGP <: AbstractSurrogate
    gp::AbstractGPs.GP
    noise_var::Float64
    gpx::Union{Nothing,AbstractGPs.PosteriorGP} 
    # gpx is the posterior GP after conditioning on data, nothing if not conditioned yet
end

Base.copy(s::StandardGP) = StandardGP(s.gp, s.noise_var, copy(s.gpx))

"""
    StandardGP(kernel::Kernel, noise_var::Float64; mean=ZeroMean())

StandardGP constructor with the specified kernel and noise variance.

Arguments:
- `kernel::Kernel`: The kernel function to be used in the GP.
- `noise_var::Float64`: The noise variance of the observations.
- `mean`: (optional) The mean function of the GP, defaults to ZeroMean()

returns:
- `StandardGP`: An instance of the StandardGP model.
"""
function StandardGP(kernel::Kernel,noise_var::Float64;mean=ZeroMean())
    gp = AbstractGPs.GP(mean,kernel) # Creates GP(0,k) for the prior
    StandardGP(gp, noise_var, nothing)
end

"""
    update!(model::StandardGP, xs::AbstractVector, ys::AbstractVector)

Update the GP model with new data points (xs, ys).

Arguments:
- `model::StandardGP`: The current GP model.
- `xs::AbstractVector`: A vector of input points where the function has been evaluated.
- `ys::AbstractVector`: A vector of corresponding function values at the input points.

returns:
- `StandardGP`: A new StandardGP model updated with the provided data.
"""
function update!(model::StandardGP, xs::AbstractVector, ys::AbstractVector)
    gpx = model.gp(xs, model.noise_var...) # This is a FiniteGP with Σy with noise_var on its diagonal.
    updated_gpx = posterior(gpx,reduce(vcat,ys))
    return StandardGP(model.gp, model.noise_var, updated_gpx)
end


"""
    nlml(mod::StandardGP,params,kernel,x,y;mean=ZeroMean())

Compute the negative log marginal likelihood (NLML) of the GP model given hyperparameters.

Arguments:
- `mod::StandardGP`: The GP model.
- `params::Tuple`: A tuple containing the log lengthscale and log scale parameters.
- `kernel`: The kernel function used in the GP.
- `x`: The input data points.
- `y`: The observed function values.
- `mean`: (optional) The mean function of the GP, defaults to ZeroMean()

returns:
- nlml : The negative log marginal likelihood of the model.
"""
function nlml(mod::StandardGP,params::AbstractVector{T},kernel::Kernel,x::AbstractVector,y::AbstractVector;mean::AbstractGPs.MeanFunction=ZeroMean()) where T

    log_ℓ, log_scale = params
    ℓ = exp(log_ℓ)
    scale = exp(log_scale)

    # Kernel with current parameters
    k = scale * (kernel ∘ ScaleTransform(1/ℓ))
    #println("creation time of standardgp")
    gp = StandardGP(k, mod.noise_var,mean=mean) # Use fixed noise here, or optimize σ² too

    # Evaluate GP at training points with noise, creates a FiniteGP
    #println("finite gpx time")
    gpx = gp.gp(x,mod.noise_var)

    #println("logpdf")
    -AbstractGPs.logpdf(gpx, y)
end


function nlml_ls(mod::StandardGP,log_ℓ::T,log_scale,kernel::Kernel,x::AbstractVector,y::AbstractVector;mean::AbstractGPs.MeanFunction=ZeroMean()) where T

    ℓ = exp(log_ℓ)
    scale = exp(log_scale)

    # Kernel with current parameters
    k = scale * (kernel ∘ ScaleTransform(1/ℓ))
    #println("creation time of standardgp")
    gp = StandardGP(k, mod.noise_var,mean=mean) # Use fixed noise here, or optimize σ² too

    # Evaluate GP at training points with noise, creates a FiniteGP
    #println("finite gpx time")
    gpx = gp.gp(x,mod.noise_var)

    #println("logpdf")
    -AbstractGPs.logpdf(gpx, y)
end


"""
    standardize_y(mod::StandardGP,y_train::AbstractVector)

Standardize the output values of the training data.

Arguments:
- `mod::StandardGP`: The GP model.
- `y_train::AbstractVector`: A vector of observed function values.

returns:
- `y_standardized`: A vector of standardized function values.
- `y_mean`: The mean of the original function values.
- `std_mean`: The standard deviation of the original function values.
"""
function standardize_y(mod::StandardGP,y_train::AbstractVector)
    y_flat = reduce(vcat, y_train)
    y_mean = mean(y_flat)
    std_mean = std(y_flat)
    
    # Protect against very small standard deviations
    if std_mean < 1e-12
        @warn "Very small standard deviation detected: $std_mean. Using std = 1.0"
        std_mean = 1.0
    end
    
    y_standardized = [(y .- y_mean) ./ std_mean for y in y_train]
    # this re-creates a Vector{Vector{Float64}}, which is what we need
    return y_standardized, y_mean, std_mean
end

get_lengthscale(model::StandardGP) = 1 ./ model.gp.kernel.kernel.transform.s

get_scale(model::StandardGP) = model.gp.kernel.σ²

prep_input(model::StandardGP,x::AbstractVector) = x

posterior_mean(model::StandardGP,x) = Statistics.mean(model.gpx(x))[1]

posterior_var(model::StandardGP,x) = Statistics.var(model.gpx(x))[1]


"""
    unstandardized_mean_and_var(gp::StandardGP, X, params::Tuple)

Compute the unstandardized mean and variance of the GP predictions at new input points.

Arguments:
- `gp::StandardGP`: The GP model.
- `X`: A vector of new input points where predictions are to be made.
- `params::Tuple`: A tuple containing the mean and standard deviation used for standardization.

returns:
- `m_unstd`: The unstandardized mean predictions at the input points.
- `v_unstd`: The unstandardized variance predictions at the input points.
"""
function unstandardized_mean_and_var(gp::StandardGP, X, params::Tuple)
    μ, σ = params
    m, v = mean_and_var(gp.gpx(X))
    # Un-standardize mean and variance
    m_unstd = (m .* σ) .+ μ
    v_unstd = v .* (σ.^2)
    return m_unstd, v_unstd
end