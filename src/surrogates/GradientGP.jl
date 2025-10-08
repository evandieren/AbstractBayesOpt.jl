"""
    GradientGP{T}(gp::AbstractGps.GP, noise_var::T, p::Int, gpx::Union{Nothing,AbstractGPs.PosteriorGP}) <: AbstractSurrogate

Implementation of the Abstract structures for the gradient-enhanced GP.

This relies on MOGP from AbstractGPs.jl and KernelFunctions.jl.

Attributes:
- `gp::AbstractGPs.GP`: The underlying Gaussian Process model.
- `noise_var::T`: The noise variance of the observations.
- `p::Int`: The number of outputs (1 for function value + d for gradients).
- `gpx::Union{Nothing,AbstractGPs.PosteriorGP}`: The posterior GP after conditioning on data, `nothing` if not conditioned yet.
"""
struct GradientGP{T,GPT<:AbstractGPs.GP,TPostGP<:Union{Nothing,AbstractGPs.PosteriorGP}} <: AbstractSurrogate
    gp::GPT
    noise_var::T
    p::Int
    gpx::TPostGP
    # gpx is the posterior GP after conditioning on data, nothing if not conditioned yet
end

"""
    Base.copy(s::GradientGP)

Creates a copy of the GradientGP instance.

returns:
- `new_s::GradientGP`: A new instance of GradientGP with copied parameters.
"""
Base.copy(s::GradientGP) = GradientGP(s.gp, s.noise_var, s.p, copy(s.gpx))

# Need to approximate around d ≈ 0 because of differentiation issues.
# We will use the squared euclidean distance because this is fine to differentiate when d ≈ 0.

"""
    ApproxMatern52Kernel{M}(metric::M) <: KernelFunctions.SimpleKernel

Approximate Matern 5/2 kernel using a second-order Taylor expansion around d=0.

Attributes:
- `metric`: The distance metric to be used, defaults to squared Euclidean distance.
"""
struct ApproxMatern52Kernel{M} <: KernelFunctions.SimpleKernel
    metric::M
end

"""
    ApproxMatern52Kernel(; metric=Distances.SqEuclidean())

Constructor for the ApproxMatern52Kernel with an optional metric argument.

Arguments:
- `metric`: The distance metric to be used, defaults to squared Euclidean distance.

returns:
- `ApproxMatern52Kernel`: An instance of the approximate Matern 5/2 kernel.
"""
ApproxMatern52Kernel(; metric=Distances.SqEuclidean()) = ApproxMatern52Kernel(metric)

"""
    KernelFunctions.metric(k::ApproxMatern52Kernel)

Get the metric used in the ApproxMatern52Kernel.

Arguments:
- `k::ApproxMatern52Kernel`: The kernel instance.

returns:
- `metric::M`: The metric used in the kernel.
"""
KernelFunctions.metric(k::ApproxMatern52Kernel) = k.metric

"""
    KernelFunctions.kappa(k::ApproxMatern52Kernel, d²::Real)

Compute the kernel value for a given squared distance using the approximate Matern 5/2 kernel.

Arguments:
- `k::ApproxMatern52Kernel`: The kernel instance.
- `d²::Real`: The squared distance between two points.

returns:
- `value::Float64`: The computed kernel value.
"""
function KernelFunctions.kappa(k::ApproxMatern52Kernel, d²::Real)
    if d² < 1e-10 # we do Taylor of order 2 around d = 0.
        return 1.0 - (5.0 / 6.0) * d²
    else
        d = sqrt(d²)
        return (1 + sqrt(5) * d + 5 * d² / 3) * exp(-sqrt(5) * d)
    end
end

"""
    Base.show(io::IO, k::ApproxMatern52Kernel)

Pretty print for the ApproxMatern52Kernel.

Arguments:
- `io::IO`: The IO stream to print to.
- `k::ApproxMatern52Kernel`: The kernel instance.

returns:
- `nothing`: Prints the kernel information to the IO stream.
"""
function Base.show(io::IO, k::ApproxMatern52Kernel)
    return print(
        io,
        "Matern 5/2 Kernel, quadratic approximation around d=0 (metric = ",
        k.metric,
        ")",
    )
end

"""
    gradConstMean{V}(c::V)

Custom mean function for the GradientGP model. Returns a constant per-output
mean across MO inputs (function value + gradients). The first element corresponds
to the function value, the following ones to the gradient outputs.

Use `gradConstMean([μ; zeros(d)])` to set a constant prior mean `μ` for the function
value and zero for the gradients.

Attributes:
- `c::V`: A vector of constants for each output (function value + gradients).
"""
struct gradConstMean{V}
    c::V
    function f_mean(vec_const, (x, px)::Tuple{X,Int}) where {X}
        return vec_const[px]
    end

    # Constructor
    function gradConstMean(c)
        return CustomMean(x -> f_mean(c, x))
    end
end

"""
    Base.show(io::IO, m::gradConstMean)

Pretty print for the gradConstMean.

Arguments:
- `io::IO`: The IO stream to print to.
- `m::gradConstMean`: The mean function instance.

returns:
- `nothing`: Prints the mean function information to the IO stream.
"""
function Base.show(io::IO, m::gradConstMean)
    return print(io, "gradConstMean(c=$(m.c))")
end

"""
    gradKernel{K}(base_kernel::K) <: MOKernel

Custom kernel function for the GradientGP model that handles both function values and gradients.

Arguments:
- `base_kernel::KernelFunctions.Kernel`: The base kernel function to be used.

returns:
- `gradKernel`: An instance of the custom gradient kernel function.
"""
mutable struct gradKernel{K} <: MOKernel
    base_kernel::K
end

raw"""
    (κ::gradKernel)((x, px)::Tuple{X,Int}, (y, py)::Tuple{Y,Int}) where {X,Y}

Compute the kernel value for given inputs and output indices using the gradKernel.

```math
k((\mathbf{x}, p), (\mathbf{x}', p')) =
\begin{cases}
  k(\mathbf{x}, \mathbf{x}') & p = 1,\; p' = 1 \\
  (\nabla_{\mathbf{x}'} k(\mathbf{x}, \mathbf{x}'))_{p'} & p = 1,\; p' \neq 1 \\
  (\nabla_{\mathbf{x}} k(\mathbf{x}, \mathbf{x}'))_{p}   & p \neq 1,\; p' = 1 \\
  (\nabla_{\mathbf{x}} \nabla_{\mathbf{x}'} k(\mathbf{x}, \mathbf{x}'))_{(p,p')} & p \neq 1,\; p' \neq 1
\end{cases}
```

Arguments:
- `κ::gradKernel`: The kernel instance.
- `(x, px)::Tuple{X,Int}`: A tuple containing the input point and its output index.
- `(y, py)::Tuple{Y,Int}`: A tuple containing the input point and its output index.

returns:
- `value::Float64`: The computed kernel value.

Some snippets kindly provided by [N. Schmitz](https://github.com/niklasschmitz), MatMat group, EPFL.
"""
function (κ::gradKernel)((x, px)::Tuple{X,Int}, (y, py)::Tuple{Y,Int}) where {X,Y}
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
                    y .+ h2 .* (1:length(y) .== (py - 1)),
                ),
                0.0,
            ),
            0.0,
        )
    end
end

"""
    GradientGP(kernel::Kernel, p::Int, noise_var::T; mean=gradConstMean(zeros(p))) where {T}

Constructor for the GradientGP model.

Arguments:
- `kernel::Kernel`: The base kernel function to be used in the GP.
- `p::Int`: The number of outputs (1 for function value + d for gradients).
- `noise_var::T`: The noise variance of the observations.
- `mean`: (optional) The mean function of the GP, defaults to gradMean with 0 constant

returns:
- `GradientGP`: An instance of the GradientGP model.
"""
function GradientGP(
    kernel::Kernel, p::Int, noise_var::T; mean=gradConstMean(zeros(p))
) where {T}
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
    update(model::GradientGP, xs::AbstractVector, ys::AbstractVector)

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
    nlml(model::GradientGP, params, xs::AbstractVector, ys::AbstractVector)

Compute the negative log marginal likelihood (NLML) of the GP model given hyperparameters.

Arguments:
- `model::GradientGP`: The GP model.
- `params`: Parameters containing the log lengthscale and log scale.
- `xs::AbstractVector`: The input data points.
- `ys::AbstractVector`: The observed function values and gradients.

returns:
- nlml::Float64: The negative log marginal likelihood of the model.
"""
function nlml(model::GradientGP, params, xs::AbstractVector, ys::AbstractVector)
    log_ℓ, log_scale = params
    ℓ = exp(log_ℓ)
    scale = exp(log_scale)

    # Kernel with current parameters
    kernel_constructor::Kernel = get_kernel_constructor(model)
    k = scale * with_lengthscale(kernel_constructor, ℓ)

    # GP with current parameters
    gp = GradientGP(k, model.p, model.noise_var; mean=model.gp.mean)
    gpx = gp.gp(xs, model.noise_var)

    return -AbstractGPs.logpdf(gpx, ys)  # Negative log marginal likelihood
end

"""
    nlml_ls(model::GradientGP, log_ℓ, log_scale, xs::AbstractVector, ys::AbstractVector)

Compute the negative log marginal likelihood (NLML) of the gradient GP model for a fixed scale and varying lengthscale.

#TODO: Think this could be merged with the general nlml function.

Arguments:
- `model::GradientGP`: The GP model.
- `log_ℓ`: The logarithm of the lengthscale parameter.
- `log_scale::Float64`: The logarithm of the scale parameter.
- `x::AbstractVector`: The input data points.
- `y::AbstractVector`: The observed function values and gradients.

returns:
- nlml : The negative log marginal likelihood of the model.

Remark: This function is a helper function for the hyperparameter_optiomize function when we want to optimize only the lengthscale.
"""
function nlml_ls(
    model::GradientGP, log_ℓ, log_scale, xs::AbstractVector, ys::AbstractVector
)
    ℓ = exp(log_ℓ)
    scale = exp(log_scale)

    # Kernel with current parameters
    kernel_constructor::Kernel = get_kernel_constructor(model)

    k = scale * with_lengthscale(kernel_constructor, ℓ)

    gp = GradientGP(k, model.p, model.noise_var; mean=model.gp.mean)

    # Evaluate GP at training points with noise, creates a FiniteGP
    #println("finite gpx time")
    gpx = gp.gp(xs, model.noise_var)

    #println("logpdf")
    return -AbstractGPs.logpdf(gpx, ys)
end

"""
    get_mean_std(model::GradientGP, y_train::AbstractVector, choice::String)

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

    μ = vec(mean(y_mat; dims=2))
    μ[2:end] .= 0.0  # Only standardize function values, not gradients
    σ = vec(std(y_mat; dims=2))
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
    std_y(model::GradientGP, ys::AbstractVector, μ::AbstractVector, σ::AbstractVector)

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
    rescale_model(model::GradientGP, σ::AbstractVector)

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
            new_kernel, model.p, model.noise_var / (σ[1]^2); mean=gradConstMean(new_c)
        )
    end

    return GradientGP(new_kernel, model.p, model.noise_var / (σ[1]^2); mean=model.gp.mean)
end

"""
    _update_model_parameters(model::GradientGP, kernel::Kernel)

Update the GP model with a new kernel (no posterior is computed here.).

Arguments:
- `model::GradientGP`: The GP model.
- `kernel::Kernel`: The new kernel to be used in the GP.

returns:
- `model::GradientGP`: The updated GP model with the new kernel
"""
function _update_model_parameters(model::GradientGP, kernel::Kernel)
    return GradientGP(kernel, model.p, model.noise_var; mean=model.gp.mean)
end

"""
    get_lengthscale(model::GradientGP)

Get the lengthscale of the GP model.

Arguments:
- `model::GradientGP`: The GP model.

returns:
- `lengthscale::Vector`: The lengthscale of the GP model.
"""
get_lengthscale(model::GradientGP) = 1 ./ model.gp.kernel.base_kernel.kernel.transform.s

"""
    get_scale(model::GradientGP)

Get the scale of the GP model.

Arguments:
- `model::GradientGP`: The GP model.

returns:
- `scale::Vector`: The scale of the GP model.
"""
get_scale(model::GradientGP) = model.gp.kernel.base_kernel.σ²

"""
    get_kernel_constructor(model::GradientGP)

Get the kernel constructor of the GP model.

Arguments:
- `model::GradientGP`: The GP model.

returns:
- `kernel_constructor::Kernel`: The kernel constructor of the GP model.
"""
get_kernel_constructor(model::GradientGP) = model.gp.kernel.base_kernel.kernel.kernel

"""
    prep_input(model::GradientGP, xs)

Prepare the input data for the GP model.

Arguments:
- `model::GradientGP`: The GP model.
- `xs`: The input data to be prepared.

returns:
- `prepared_xs`: The prepared input data.
"""
prep_input(model::GradientGP, xs) = _prep_input(xs, model.p)

function _prep_input(x::AbstractVector{X}, p::Int) where {X}
    return KernelFunctions.MOInputIsotopicByOutputs(x, p)
end

function _prep_input(x::AbstractVector{<:Tuple{X,Int}}, p::Int) where {X}
    return x
end

function _prep_input(x::Tuple{X,Int}, p::Int) where {X}
    return [x]
end

function _prep_input(x::X, p::Int) where {X<:Real}
    return _prep_input([x], p)
end

"""
    prep_output(model::GradientGP, y::Vector{Y}) where {Y}

Prepare the output data for the GP model.

Arguments:
- `model::GradientGP`: The GP model.
- `y::Vector{Y}`: The output data to be prepared.

returns:
- `prepared_y`: The prepared output data.
"""
function prep_output(model::GradientGP, y::Vector{Y}) where {Y}
    # Need to align them properly for MOGP
    return vec(permutedims(reduce(hcat, y)))
end

"""
    posterior_grad_mean(model::GradientGP, x)

Compute the mean predictions of the GP model at new input points, including gradients.

Arguments:
- `model::GradientGP`: The GP model.
- `x`: A vector of new input points where predictions are to be made.

returns:
- `mean::Vector`: The mean predictions
"""
function posterior_grad_mean(model::GradientGP, x)
    # Be careful with the output order, it is (f(x1),f(x2),...,∂₁f(x1),∂₁f(x2),...)
    mean(model.gpx(_prep_input(x, model.p)))
end

"""
    posterior_grad_var(model::GradientGP, x)

Compute the variance predictions of the GP model at new input points, including gradients.

Arguments:
- `model::GradientGP`: The GP model.
- `x`: A vector of new input points where predictions are to be made.

returns:
- `var::Vector`: The variance predictions
"""
function posterior_grad_var(model::GradientGP, x)
    var(model.gpx(_prep_input(x, model.p)))
end

"""
    posterior_grad_cov(model::GradientGP, x)

Compute the covariance matrix of the GP model at new input points, including gradients.

Arguments:
- `model::GradientGP`: The GP model.
- `x`: A vector of new input points where predictions are to be made.

returns:
- `cov::Matrix`: The covariance matrix of the predictions
"""
function posterior_grad_cov(model::GradientGP, x)
    cov(model.gpx(_prep_input(x, model.p)))
end

"""
    posterior_mean(model::GradientGP, x)

Compute the function mean predictions of the GP model at new input points.

Arguments:
- `model::GradientGP`: The GP model.
- `x`: A vector of new input points where predictions are to be made.

returns:
- `mean::Vector`: The mean predictions (function value only)
"""
function posterior_mean(model::GradientGP, x)
    mean(model.gpx(_prep_input(x, 1)))
end

"""
    posterior_var(model::GradientGP, x)

Compute the function variance predictions of the GP model at new input points.

Arguments:
- `model::GradientGP`: The GP model.
- `x`: A vector of new input points where predictions are to be made.

returns:
- `var::Vector`: The variance predictions (function value only)
"""
function posterior_var(model::GradientGP, x)
    var(model.gpx(_prep_input(x, 1)))
end

"""
    unstandardized_mean_and_var(gp::GradientGP, X, params::Tuple)

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
    m = posterior_grad_mean(gp, X)
    v = posterior_grad_var(gp, X)
    # Un-standardize mean and variance
    m = reshape(m, :, gp.p)
    m_unstd = (m .* σ) .+ μ'

    v = reshape(v, :, gp.p)
    v_unstd = v .* (σ .^ 2)
    return m_unstd, v_unstd
end

"""
    _get_minimum(gp::GradientGP, ys::Vector{Y}) where {Y}

Helper to get the minimum function value from the outputs.

Arguments:
- `gp::GradientGP`: The GP model.
- `ys::Vector{Y}`: A vector of observed function values.

returns:
- `min_y::Y`: The minimum function value from the outputs.
"""
_get_minimum(gp::GradientGP, ys::Vector{Y}) where {Y} = minimum(hcat(ys...)[1, :])[1]
