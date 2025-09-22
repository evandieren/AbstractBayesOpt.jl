"""
Implementation of the Abstract structures for the gradient-enhanced GP.


This relies on MOGP from AbstractGPs.jl and KernelFunctions.jl.
"""
struct GradientGP <: AbstractSurrogate
    gp::AbstractGPs.GP
    noise_var::Float64
    p::Int
    gpx::Union{Nothing, AbstractGPs.PosteriorGP}
    # gpx is the posterior GP after conditioning on data, nothing if not conditioned yet
end

Base.copy(s::GradientGP) = GradientGP(s.gp, s.noise_var, s.p, copy(s.gpx))

# Need to approximate around d ≈ 0 because of differentiation issues.
# We will use the squared euclidean distance because this is fine to differentiate when d ≈ 0.

"""
Approximate Matern 5/2 kernel using a second-order Taylor expansion around d=0.

Arguments:
- `metric`: The distance metric to be used, defaults to squared Euclidean distance.

returns:
- `ApproxMatern52Kernel`: An instance of the approximate Matern 5/2 kernel.
"""
struct ApproxMatern52Kernel{M} <: KernelFunctions.SimpleKernel
    metric::M
end
ApproxMatern52Kernel(; metric = Distances.SqEuclidean()) = ApproxMatern52Kernel(metric)
KernelFunctions.metric(k::ApproxMatern52Kernel) = k.metric

function KernelFunctions.kappa(k::ApproxMatern52Kernel, d²::Real)
    if d² < 1e-10 # we do Taylor of order 2 around d = 0.
        return 1.0 - (5.0 / 6.0) * d²
    else
        d = sqrt(d²)
        return (1 + sqrt(5) * d + 5 * d² / 3) * exp(-sqrt(5) * d)
    end
end
function Base.show(io::IO, k::ApproxMatern52Kernel)
    return print(
        io,
        "Matern 5/2 Kernel, quadratic approximation around d=0 (metric = ",
        k.metric,
        ")"
    )
end

"""
Custom mean function for the GradientGP model. Returns a constant per-output
mean across MO inputs (function value + gradients). The first element corresponds
to the function value, the following ones to the gradient outputs.

Use gradConstMean([μ_f; zeros(d)]) to set a constant prior mean μ_f for the function
value and zero for the gradients.
"""

struct gradConstMean{V}
    c::V
    function f_mean(vec_const, (x, px)::Tuple{X, Int}) where {X}
        return vec_const[px]
    end

    function gradConstMean(c)
        return CustomMean(x -> f_mean(c, x))
    end
end

function Base.show(io::IO, m::gradConstMean)
    return print(io, "gradConstMean(c=$(m.c))")
end

"""
Custom kernel function for the GradientGP model that handles both function values and gradients.

Arguments:
- `base_kernel::KernelFunctions.Kernel`: The base kernel function to be used.

returns:
- `gradKernel`: An instance of the custom gradient kernel function.
"""
mutable struct gradKernel{K} <: MOKernel
    base_kernel::K
end

function (κ::gradKernel)(
        (x, px)::Tuple{X, Int}, (y, py)::Tuple{Y, Int}) where {X, Y}
    """
    ```math
    k((\vec{x},p),(\vec{x}',p'))
    ```
    where if ``p = p' = 1`` returns ``k(\vec{x},\vec{x}')``,
          if ``p = 1, p' \neq 1`` returns ``(\nabla_{\vec{x}'} k(\vec{x},\vec{x}'))_{p'}``,
          if ``p \neq 1, p' = 1`` returns ``(\nabla_{\vec{x}} k(\vec{x}),\vec{x}')_{p}``,
          and if ``p \neq 1, p' \neq 1``, returns ``(\nabla_x \nabla_{x'} k(\vec{x},\vec{x}')_{(p,p')}``

    Some snippets kindly provided by Niklas Schmitz, MatMat group, EPFL.
    """
    (px > length(x) + 1 || py > length(y) + 1 || px < 1 || py < 1) &&
        error("`px` and `py` must be within the range of the number of outputs")

    onehot(n, i) = 1:n .== i # collect(1:n) .== i

    val = px == 1 && py == 1 # we are looking at f(x), f(y)

    ∇_val_1 = (px != 1 && py == 1) # we are looking at ∇f(x)-f(y)
    ∇_val_2 = (px == 1 && py != 1) # we are looking at f(x)-∇f(y)

    if val # we are just computing the usual matrix K
        κ.base_kernel(x, y)
    elseif ∇_val_1
        return ForwardDiff.derivative(
            h -> κ.base_kernel(x .+ h .* (1:length(x) .== (px - 1)), y), 0.0
        )
    elseif ∇_val_2 # we are looking at f(x)-∇f(y)
        return ForwardDiff.derivative(
            h -> κ.base_kernel(x, y .+ h .* (1:length(y) .== (py - 1))), 0.0
        )
    else # we are looking at ∇f(x)-∇f(y), this avoids computing the entire hessian each time.
        return ForwardDiff.derivative(
            h1 -> ForwardDiff.derivative(
                h2 -> κ.base_kernel(
                    x .+ h1 .* (1:length(x) .== (px - 1)),
                    y .+ h2 .* (1:length(y) .== (py - 1))
                ),
                0.0
            ),
            0.0
        )
    end
end

"""
Constructor for the GradientGP model.

Arguments:
- `kernel::Kernel`: The base kernel function to be used in the GP.
- `p::Int`: The number of outputs (1 for function value + d for gradients).
- `noise_var::Float64`: The noise variance of the observations.
- `mean`: (optional) The mean function of the GP, defaults to gradMean with 0 constant

returns:
- `GradientGP`: An instance of the GradientGP model.
"""
function GradientGP(
        kernel::Kernel, p::Int, noise_var::Float64; mean = gradConstMean(zeros(p)))

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

    kernel = gradKernel(kernel)

    gp = AbstractGPs.GP(mean, kernel) # Creates GP(0,k) for the prior
    return GradientGP(gp, noise_var, p, nothing)
end

"""
Update the GP model with new data points (xs, ys).

Arguments:
- `model::GradientGP`: The current GP model.
- `xs::AbstractVector`: A vector of input points where the function has been evaluated.
- `ys::AbstractVector`: A vector of corresponding function values and gradients at the input points

returns:
- `GradientGP`: A new GradientGP model updated with the provided data.
"""
function update(model::GradientGP, xs::AbstractVector, ys::AbstractVector)
    # we could do something better for this, such as inserting the batch of new
    # points in xs and ys which are already MOInputIsotopicByOutputs elements.
    x_tilde, y_tilde = prepare_isotopic_multi_output_data(xs, ColVecs(reduce(hcat, ys)))

    gpx = model.gp(x_tilde, model.noise_var...)
    updated_gpx = posterior(gpx, y_tilde)

    return GradientGP(model.gp, model.noise_var, model.p, updated_gpx)
end

"""
Compute the negative log marginal likelihood (NLML) of the GP model given hyperparameters.

Arguments:
- `model::GradientGP`: The GP model.
- `params::Tuple`: A tuple containing the log lengthscale and log scale parameters.
- `xs`: The input data points.
- `ys`: The observed function values and gradients.

returns:
- nlml : The negative log marginal likelihood of the model.
"""
function nlml(
        model::GradientGP,
        params,
        xs::AbstractVector,
        ys::AbstractVector
)
    log_ℓ, log_scale = params
    ℓ = exp(log_ℓ)
    scale = exp(log_scale)

    # Kernel with current parameters
    kernel_constructor::Kernel = get_kernel_constructor(model)
    k = scale * with_lengthscale(kernel_constructor, ℓ)

    # GP with current parameters
    gp = GradientGP(k, model.p, model.noise_var; mean = model.gp.mean)
    gpx = gp.gp(xs, model.noise_var)

    return -AbstractGPs.logpdf(gpx, ys)  # Negative log marginal likelihood
end

"""
Compute the negative log marginal likelihood (NLML) of the gradient GP model for a fixed scale and varying lengthscale.

Arguments:
- `model::GradientGP`: The GP model.
- `log_ℓ::T`: The logarithm of the lengthscale parameter.
- `log_scale::Float64`: The logarithm of the scale parameter.
- `x::AbstractVector`: The input data points.
- `y::AbstractVector`: The observed function values and gradients.

returns:
- nlml : The negative log marginal likelihood of the model.

Remark: This function is a helper function for the hyperparameter_optiomize function when we want to optimize only the lengthscale.
"""
function nlml_ls(
        model::GradientGP,
        log_ℓ,
        log_scale,
        xs::AbstractVector,
        ys::AbstractVector
)
    ℓ = exp(log_ℓ)
    scale = exp(log_scale)

    # Kernel with current parameters
    kernel_constructor::Kernel = get_kernel_constructor(model)

    k = scale * with_lengthscale(kernel_constructor, ℓ)

    gp = GradientGP(k, model.p, model.noise_var; mean = model.gp.mean)

    # Evaluate GP at training points with noise, creates a FiniteGP
    #println("finite gpx time")
    gpx = gp.gp(xs, model.noise_var)

    #println("logpdf")
    return -AbstractGPs.logpdf(gpx, ys)
end

"""
Gets the empirical mean and std of y_train (Vector of Vector of Float64)

Arguments:
- `model::GradientGP`: The GP model.
- `y_train::AbstractVector`: A vector of observed function values.

returns:
- `y_mean`: Empirical mean
- `y_std`: Empirical standard deviation
"""
function get_mean_std(model::GradientGP, y_train::AbstractVector, choice::String)
    y_mat = reduce(hcat, y_train)

    μ = vec(mean(y_mat; dims = 2))
    μ[2:end] .= 0.0  # Only standardize function values, not gradients
    σ = vec(std(y_mat; dims = 2))
    σ[2:end] .= σ[1]  # Use same scaling for gradients

    if choice == "scale_only"
        μ .= 0.0
    elseif choice == "mean_only"
        σ .= ones(length(σ))
    end
    # This will be of the same type as elements of y_train
    return μ, σ
end

"""
Standardize the output values of the training data

Arguments:
- `model::GradientGP`: The GP model.
- `ys::AbstractVector`: A vector of observed function values.
- `μ::AbstractVector`: Empirical mean
- `σ::AbstractVector`: Empirical standard deviation

returns:
- `y_std`: A vector of standardized function values.
"""
function std_y(model::GradientGP, ys::AbstractVector, μ::AbstractVector, σ::AbstractVector)
    y_std = [(y .- μ) ./ σ[1] for y in ys]
    return y_std
end

"""
Update the kernel scale of the GP model.

Arguments:
- `model::GradientGP`: The GP model.
- `σ::AbstractVector`: Empirical standard deviation

returns:
- `model::GradientGP`: The updated GP model with the new kernel scale.
"""
function rescale_model(model::GradientGP, σ::AbstractVector)
    ℓ = get_lengthscale(model)[1]
    old_scale = get_scale(model)[1]
    kernel_constructor = get_kernel_constructor(model)

    new_scale = old_scale / (σ[1]^2)

    new_kernel = new_scale * with_lengthscale(kernel_constructor, ℓ)

    if isa(model.gp.mean, gradConstMean)
        old_c = model.gp.mean.c
        new_c = old_c ./ σ[1]
        return GradientGP(
            new_kernel, model.p, model.noise_var / (σ[1]^2); mean = gradConstMean(new_c)
        )
    end

    return GradientGP(new_kernel, model.p, model.noise_var / (σ[1]^2); mean = model.gp.mean)
end


function _update_model_parameters(model::GradientGP, kernel::Kernel)
    return GradientGP(kernel, model.p, model.noise_var; mean = model.gp.mean)
end

get_lengthscale(model::GradientGP) = 1 ./ model.gp.kernel.base_kernel.kernel.transform.s

get_scale(model::GradientGP) = model.gp.kernel.base_kernel.σ²

get_kernel_constructor(model::GradientGP) = model.gp.kernel.base_kernel.kernel.kernel

prep_input(model::GradientGP, xs) = _prep_input(xs, model.p)

function _prep_input(x::AbstractVector{X}, p::Int) where {X}
    return KernelFunctions.MOInputIsotopicByOutputs(x, p)
end

function _prep_input(x::AbstractVector{<:Tuple{X, Int}}, p::Int) where {X}
    return x
end

function _prep_input(x::Tuple{X, Int}, p::Int) where {X}
    return [x]
end

function _prep_input(x::X, p::Int) where {X <: Real}
    return _prep_input([x], p)
end

function prep_output(model::GradientGP, y::Vector{Y}) where {Y}
    # Need to align them properly for MOGP
    return vec(permutedims(reduce(hcat, y)))
end

function posterior_grad_mean(model::GradientGP, x)
    mean(model.gpx(_prep_input(x, model.p)))
end

function posterior_grad_var(model::GradientGP, x)
    var(model.gpx(_prep_input(x, model.p)))
end

function posterior_grad_cov(model::GradientGP, x)
    cov(model.gpx(_prep_input(x, model.p)))
end

function posterior_mean(model::GradientGP, x)
    mean(model.gpx(_prep_input(x, 1)))
end

function posterior_var(model::GradientGP, x)
    var(model.gpx(_prep_input(x, 1)))
end

"""
Compute the unstandardized mean and variance of the GP predictions at new input points.

Arguments:
- `gp::GradientGP`: The GP model.
- `X`: A vector of new input points where predictions are to be made.
- `params::Tuple`: A tuple containing the mean and standard deviation used for standardization.

returns:
- `m_unstd`: The unstandardized mean predictions at the input points. (matrix of size (num_points, p) where p is the number of outputs)
- `v_unstd`: The unstandardized variance predictions at the input points. (matrix of size (num_points, p) where p is the number of outputs)
"""
function unstandardized_mean_and_var(gp::GradientGP, X, params::Tuple)
    μ, σ = params[1], params[2][1]
    m, v = mean_and_var(gp.gpx(X))
    # Un-standardize mean and variance
    m = reshape(m, :, gp.p)
    m_unstd = (m .* σ) .+ μ'

    v = reshape(v, :, gp.p)
    v_unstd = v .* (σ .^ 2)
    return m_unstd, v_unstd
end

_get_minimum(gp::GradientGP, ys::Vector{Y}) where {Y} = minimum(hcat(ys...)[1, :])[1]
