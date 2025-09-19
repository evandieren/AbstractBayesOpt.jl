"""
Implementation of the Abstract structures for the standard GP.

Remark: this is a simple wrapper around AbstractGPs.jl that implements the AbstractSurrogate abstract type
"""
struct StandardGP <: AbstractSurrogate
    gp::AbstractGPs.GP
    noise_var::Float64
    gpx::Union{Nothing,AbstractGPs.PosteriorGP}
    # gpx is the posterior GP after conditioning on data, nothing if not conditioned yet
end

Base.copy(s::StandardGP) = StandardGP(s.gp, s.noise_var, copy(s.gpx))

"""
StandardGP constructor with the specified kernel and noise variance.

Arguments:
- `kernel::Kernel`: The kernel function to be used in the GP.
- `noise_var::Float64`: The noise variance of the observations.
- `mean`: (optional) The mean function of the GP, defaults to nothing (creates a ZeroMean() if not provided).

returns:
- `StandardGP`: An instance of the StandardGP model.
"""
function StandardGP(kernel::Kernel, noise_var::Float64; mean=nothing)
    if isnothing(mean)
        mean = ZeroMean()
    end

    # Extract inner kernel, scale, and lengthscale
    inner, scale, lengthscale = extract_scale_and_lengthscale(kernel)

    # Decide defaults
    if lengthscale === nothing
        inner = with_lengthscale(inner, 1.0)
    else
        inner = with_lengthscale(inner, lengthscale)
    end

    if scale == 1.0 && !isa(kernel, AbstractGPs.ScaledKernel)
        kernel = ScaledKernel(inner, 1.0)
    else
        kernel = ScaledKernel(inner, scale)
    end

    gp = AbstractGPs.GP(mean, kernel) # Creates GP(0,k) for the prior
    return StandardGP(gp, noise_var, nothing)
end

"""
Update the GP model with new data points (xs, ys).

Arguments:
- `model::StandardGP`: The current GP model.
- `xs::AbstractVector`: A vector of input points where the function has been evaluated.
- `ys::AbstractVector`: A vector of corresponding function values at the input points.

returns:
- `StandardGP`: A new StandardGP model updated with the provided data.
"""
function update(model::StandardGP, xs::AbstractVector, ys::AbstractVector)
    gpx = model.gp(xs, model.noise_var...) # This is a FiniteGP with Σy with noise_var on its diagonal.
    updated_gpx = posterior(gpx, reduce(vcat, ys))
    return StandardGP(model.gp, model.noise_var, updated_gpx)
end

"""
Compute the negative log marginal likelihood (NLML) of the GP model given hyperparameters.

Arguments:
- `model::StandardGP`: The GP model.
- `params::Tuple`: A tuple containing the log lengthscale and log scale parameters.
- `x`: The input data points.
- `y`: The observed function values.
- `mean`: (optional) The mean function of the GP, defaults to ZeroMean()

returns:
- nlml : The negative log marginal likelihood of the model.
"""
function nlml(
    model::StandardGP,
    params::Vector{T},
    xs::Vector{X},
    ys::Vector{Y};
    mean::AbstractGPs.MeanFunction=ZeroMean(),
) where {T, X, Y}
    log_ℓ, log_scale = params
    ℓ = exp(log_ℓ)
    scale = exp(log_scale)

    kernel_constructor = get_kernel_constructor(model)

    # Kernel with current parameters
    k = scale * with_lengthscale(kernel_constructor, ℓ)
    gp = StandardGP(k, model.noise_var; mean=mean) # Use fixed noise here, or optimize σ² too

    # Evaluate GP at training points with noise, creates a FiniteGP
    gpx = gp.gp(xs, model.noise_var)

    return -AbstractGPs.logpdf(gpx, ys)
end

"""
Compute the negative log marginal likelihood (NLML) of the GP model given log lengthscale and log scale parameters.

Arguments:
- `model::StandardGP`: The GP model.
- `log_ℓ::T`: The log lengthscale parameter.
- `log_scale::Float64`: The log scale parameter.
- `x`: The input data points.
- `y`: The observed function values.
- `mean`: (optional) The mean function of the GP, defaults to ZeroMean()

returns:
- nlml : The negative log marginal likelihood of the model.

Remark: This function is useful for optimizing only the lengthscale and scale parameters while keeping other parameters fixed.
"""
function nlml_ls(
    model::StandardGP,
    log_ℓ::T,
    log_scale::Float64,
    x::AbstractVector,
    y::AbstractVector;
    mean::AbstractGPs.MeanFunction=ZeroMean(),
) where {T}
    ℓ = exp(log_ℓ)
    scale = exp(log_scale)

    kernel_constructor = get_kernel_constructor(model)

    # Kernel with current parameters
    k = scale * (kernel_constructor ∘ ScaleTransform(1/ℓ))
    #println("creation time of standardgp")
    gp = StandardGP(k, model.noise_var; mean=mean) # Use fixed noise here, or optimize σ² too

    # Evaluate GP at training points with thought noise, creates a FiniteGP
    #println("finite gpx time")
    gpx = gp.gp(x, model.noise_var)

    #println("logpdf")
    return -AbstractGPs.logpdf(gpx, y)
end

"""
Gets the empirical mean and std of y_train (Vector of Vector of Float64)

Arguments:
- `model::StandardGP`: The GP model.
- `y_train::AbstractVector`: A vector of observed function values.

returns:
- `y_mean`: Empirical mean
- `y_std`: Empirical standard deviation
"""
function get_mean_std(model::StandardGP, y_train::Vector{Y}, choice::String) where {Y}
    y_flat = reduce(vcat, y_train)

    y_mean = mean(y_flat)
    y_std = std(y_flat)


    # Taking into account the choice of the user
    if choice == "scale_only"
        y_mean = 0.0
    elseif choice == "mean_only"
        y_std = 1.0 
    end

    # This is of type Y
    return y_mean, y_std
end

"""
Standardize the output values of the training data

Arguments:
- `model::StandardGP`: The GP model.
- `ys::AbstractVector`: A vector of observed function values.
- `μ`: Empirical mean
- `σ`: Empirical standard deviation

returns:
- `y_std`: A vector of standardized function values.
"""
function std_y(model::StandardGP, ys::Vector{Y}, μ, σ) where {Y}
    y_std = [(y .- μ) ./ σ for y in ys]
    return y_std
end

"""
Update the kernel scale of the GP model.

Arguments:
- `model::StandardGP`: The GP model.
- `σ`: Empirical standard deviation

returns:
- `model::StandardGP`: The updated GP model with the new kernel scale.
"""
function rescale_model(model::StandardGP, σ)
    ℓ::Float64 = get_lengthscale(model)[1]
    old_scale::Float64 = get_scale(model)[1]
    kernel_constructor = get_kernel_constructor(model)

    new_scale = old_scale / (σ^2)

    new_kernel = new_scale * (with_lengthscale(kernel_constructor, ℓ))

    # If the GP mean is not a ZeroMean, we need to rescale it too for consistency
    if !(model.gp.mean isa ZeroMean)
        new_mean_val = model.gp.mean.c / σ # works for ConstMean
        model = StandardGP(
            new_kernel, model.noise_var / (σ^2); mean=ConstMean(new_mean_val)
        )
        return model
    end

    return StandardGP(new_kernel, model.noise_var / (σ^2); mean=model.gp.mean)
end

get_lengthscale(model::StandardGP) = 1 ./ model.gp.kernel.kernel.transform.s

get_scale(model::StandardGP) = model.gp.kernel.σ²

get_kernel_constructor(model::StandardGP) = model.gp.kernel.kernel.kernel

prep_input(model::StandardGP, x::AbstractVector) = x

prep_output(model::StandardGP, y::AbstractVector) = y

# These functions are used when we need to query one point
posterior_mean(model::StandardGP, x::AbstractVector) = mean(model.gpx([x]))[1] # we do the function values
posterior_var(model::StandardGP, x::AbstractVector) = var(model.gpx([x]))[1] # we do the function values

# These functions are used in a buffer way within the optimisation of the acquisition function
function posterior_mean(model::StandardGP, x_buf::Vector{Vector{Float64}})
    Statistics.mean(model.gpx(x_buf))[1]
end
function posterior_var(model::StandardGP, x_buf::Vector{Vector{Float64}})
    Statistics.var(model.gpx(x_buf))[1]
end

"""
Compute the unstandardized mean and variance of the GP predictions at new input points.

Arguments:
- `model::StandardGP`: The GP model.
- `X`: A vector of new input points where predictions are to be made.
- `params::Tuple`: A tuple containing the mean and standard deviation used for standardization.

returns:
- `m_unstd`: The unstandardized mean predictions at the input points.
- `v_unstd`: The unstandardized variance predictions at the input points.
"""
function unstandardized_mean_and_var(model::StandardGP, X, params::Tuple)
    μ, σ = params
    m, v = mean_and_var(model.gpx(X))
    # Un-standardize mean and variance
    m_unstd = (m .* σ) .+ μ
    v_unstd = v .* (σ^2)
    return m_unstd, v_unstd
end
