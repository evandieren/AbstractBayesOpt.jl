"""
    abstract type AbstractLinearOperator

Abstract type for linear operators applied to a latent GP.
All concrete operators must implement the callable interface `(op::MyOperator)(f, x)`
which applies the operator to a scalar-valued function `f` at point `x`.
"""
abstract type AbstractLinearOperator end

"""
    struct IdentityOperator <: AbstractLinearOperator

Linear operator that returns the function value unchanged, i.e. `(If)(x) = f(x)`.
"""
struct IdentityOperator <: AbstractLinearOperator end

"""
    struct PartialDerivative <: AbstractLinearOperator

Linear operator that computes the partial derivative of a function with respect to
dimension `dim`, i.e. `(∂_i f)(x) = ∂f/∂x_i`.

Attributes:
- `dim::Int`: The dimension along which to differentiate.
"""
struct PartialDerivative <: AbstractLinearOperator
    dim::Int
end

"""
    struct LaplacianOperator <: AbstractLinearOperator

Linear operator that computes the Laplacian of a function, i.e. `(Δf)(x) = Σᵢ ∂²f/∂xᵢ²`.
"""
struct LaplacianOperator <: AbstractLinearOperator end

"""
    (::IdentityOperator)(f, x)

Apply the identity operator to `f` at `x`, returning `f(x)` unchanged.

Arguments:
- `f`: A scalar-valued function.
- `x`: The input point.

returns:
- `f(x)`: The function value at `x`.
"""
(::IdentityOperator)(f, x) = f(x)

"""
    _unitvec(x, i)

Construct a unit vector of the same type and length as `x` with a one in position `i`.

Arguments:
- `x`: A vector used to determine the length and element type.
- `i::Int`: The index of the non-zero entry.

returns:
- `e::Vector`: A unit vector with `e[i] = 1` and zeros elsewhere.
"""
function _unitvec(x, i)
    e = zeros(eltype(x), length(x))
    e[i] = one(eltype(x))
    return e
end

"""
    (op::PartialDerivative)(f, x)

Apply the partial derivative operator to `f` at `x` along dimension `op.dim`,
computed via forward-mode automatic differentiation.

Arguments:
- `op::PartialDerivative`: The operator, carrying the dimension `op.dim`.
- `f`: A scalar-valued function.
- `x`: The input point.

returns:
- `∂f/∂x_i`: The partial derivative of `f` at `x` along dimension `op.dim`.
"""
function (op::PartialDerivative)(f, x)
    e = _unitvec(x, op.dim)
    return ForwardDiff.derivative(h -> f(x .+ h .* e), zero(eltype(x)))
end

"""
    (::LaplacianOperator)(f, x)

Apply the Laplacian operator to `f` at `x`, computed as the trace of the Hessian
via forward-mode automatic differentiation.

Arguments:
- `f`: A scalar-valued function.
- `x`: The input point.

returns:
- `Δf(x)`: The Laplacian of `f` at `x`, i.e. `Σᵢ ∂²f/∂xᵢ²`.
"""
function (::LaplacianOperator)(f, x)
    return LinearAlgebra.tr(ForwardDiff.hessian(f, x))
end

"""
    struct LinearOperatorKernel{K, Ops} <: KernelFunctions.Kernel

Kernel for a GP observed through linear operators. For operators `Lᵢ` and `Lⱼ`,
the kernel between two tagged inputs `(x, i)` and `(y, j)` is defined as:

    k_ij(x, y) = Lᵢˣ Lⱼʸ k(x, y)

where `Lᵢˣ` denotes operator `i` applied in the `x` argument.

Attributes:
- `base_kernel::K`: The kernel of the latent GP.
- `ops::Ops`: The collection of linear operators.
"""
struct LinearOperatorKernel{K,Ops} <: KernelFunctions.Kernel
    base_kernel::K
    ops::Ops
end

"""
    struct LinearOperatorMean{M, Ops} <: AbstractGPs.MeanFunction

Mean function for a GP observed through linear operators. For operator `Lᵢ`,
the mean at tagged input `(x, i)` is defined as:

    m_i(x) = Lᵢ m(x)

where `m` is the base mean function of the latent GP.

Attributes:
- `base_mean::M`: The mean function of the latent GP.
- `ops::Ops`: The collection of linear operators.
"""
struct LinearOperatorMean{M,Ops} <: AbstractGPs.MeanFunction
    base_mean::M
    ops::Ops
end

"""
    AbstractGPs.mean_vector(m::LinearOperatorMean, xs)

Evaluate the linear operator mean function at a vector of tagged inputs.

Arguments:
- `m::LinearOperatorMean`: The mean function.
- `xs`: A vector of tagged inputs of the form `(x, i)`.

returns:
- `Vector`: The mean vector, where each entry is `Lᵢ m(x)` for the corresponding `(x, i)`.
"""
function AbstractGPs.mean_vector(m::LinearOperatorMean, xs)
    return [m.ops[i](z -> _scalar_mean(m.base_mean, z), x) for (x, i) in xs]
end

"""
    _scalar_mean(m, x)

Evaluate a mean function `m` at a single point `x`, returning a scalar.

Arguments:
- `m`: A `MeanFunction` compatible with `AbstractGPs.mean_vector`.
- `x`: A single input point.

returns:
- `Float64`: The scalar mean value at `x`.
"""
_scalar_mean(m, x) = only(AbstractGPs.mean_vector(m, [x]))

"""
    make_linear_operator_mean(base_mean, ops)

Construct a `CustomMean` that applies the appropriate linear operator to the base mean
function based on the output index of each tagged input.

Arguments:
- `base_mean`: The mean function of the latent GP.
- `ops`: The collection of linear operators.

returns:
- `AbstractGPs.CustomMean`: A mean function that evaluates `Lᵢ m(x)` for tagged input `(x, i)`.
"""
function make_linear_operator_mean(base_mean, ops)
    return AbstractGPs.CustomMean() do xi
        x, i = xi
        return ops[i](z -> _scalar_mean(base_mean, z), x)
    end
end

"""
    (κ::LinearOperatorKernel)((x, i), (y, j))

Evaluate the linear operator kernel at two tagged inputs `(x, i)` and `(y, j)`.
Applies operator `i` in the `x` argument and operator `j` in the `y` argument
to the base kernel:

    k_ij(x, y) = Lᵢˣ Lⱼʸ k(x, y)

Arguments:
- `κ::LinearOperatorKernel`: The kernel instance.
- `(x, i)`: A tagged input, where `i` is the operator index.
- `(y, j)`: A tagged input, where `j` is the operator index.

returns:
- `Float64`: The kernel value `Lᵢˣ Lⱼʸ k(x, y)`.
"""
function (κ::LinearOperatorKernel)((x, i), (y, j))
    opx = κ.ops[i]
    opy = κ.ops[j]
    return opx(a -> opy(b -> κ.base_kernel(a, b), y), x)
end

"""
    struct LinearOperatorGP{G, T, Ops} <: AbstractSurrogate

Gaussian Process surrogate model for observations of linear functionals of a latent GP.

Attributes:
- `gp::G`: The underlying Gaussian Process model.
- `noise_var::T`: The noise variance of the observations. Can be a `Real` (uniform noise),
    a `Tuple` of length p (one variance per operator type), or an `AbstractVector` of
    length n (one variance per observation, fully heteroskedastic).
- `ops::Ops`: The collection of linear operators applied to the latent GP.
- `p::Int`: The number of operators (output types).
- `gpx::Union{Nothing, AbstractGPs.PosteriorGP}`: The posterior GP after conditioning on
    data, `nothing` if not yet conditioned.
"""
struct LinearOperatorGP{G,T,Ops} <: AbstractSurrogate
    gp::G
    noise_var::T
    ops::Ops
    p::Int
    gpx::Union{Nothing,AbstractGPs.PosteriorGP}
end

"""
    LinearOperatorGP(base_kernel, ops, noise_var; mean=AbstractGPs.ZeroMean())

Constructor for the LinearOperatorGP model.

Arguments:
- `base_kernel`: The kernel of the latent GP.
- `ops`: A collection of linear operators to apply to the latent GP.
- `noise_var`: The noise variance. Accepts a `Real`, a `Tuple` of length p, or an
    `AbstractVector` of length n (see `_build_noise` for details).
- `mean`: (optional) The mean function of the latent GP, defaults to `ZeroMean()`.

returns:
- `LinearOperatorGP`: An instance of the LinearOperatorGP model with no posterior.
"""
function LinearOperatorGP(base_kernel::Kernel, ops, noise_var::T; mean=AbstractGPs.ZeroMean()) where {T}
    inner, scale, lengthscale = extract_scale_and_lengthscale(base_kernel)

    if lengthscale === nothing
        inner = with_lengthscale(inner, 1.0)
    else
        inner = with_lengthscale(inner, lengthscale)
    end

    if scale == 1.0 && !isa(base_kernel, AbstractGPs.ScaledKernel)
        base_kernel = ScaledKernel(inner, 1.0)
    else
        base_kernel = ScaledKernel(inner, scale)
    end

    p = length(ops)
    wrapped_mean = LinearOperatorMean(mean, ops)
    wrapped_kernel = LinearOperatorKernel(base_kernel, ops)
    gp = AbstractGPs.GP(wrapped_mean, wrapped_kernel)

    return LinearOperatorGP(gp, noise_var, ops, p, nothing)
end

"""
    Base.copy(model::LinearOperatorGP)

Create a shallow copy of the LinearOperatorGP instance.

Arguments:
- `model::LinearOperatorGP`: The model to copy.

returns:
- `LinearOperatorGP`: A new instance with the same fields.
"""
Base.copy(model::LinearOperatorGP) = LinearOperatorGP(
    model.gp,
    model.noise_var,
    model.ops,
    model.p,
    copy(model.gpx),
)

"""
    update(model::LinearOperatorGP, xs::AbstractVector{<:Tuple}, ys::AbstractVector{<:Real})

Update the LinearOperatorGP with heterogeneous (tagged) observations. Each input is a
`(x, i)` tuple explicitly specifying which operator was applied at that point.

Arguments:
- `model::LinearOperatorGP`: The current model.
- `xs::AbstractVector{<:Tuple}`: Tagged input points of the form `(x, i)`.
- `ys::AbstractVector{<:Real}`: Scalar observations corresponding to each tagged input.

returns:
- `LinearOperatorGP`: A new model conditioned on the provided data.

Throws:
- `ArgumentError`: If `length(xs) ≠ length(ys)`.
"""
function update(model::LinearOperatorGP, xs::AbstractVector{<:Tuple}, ys::AbstractVector{<:Real})
    length(xs) == length(ys) || throw(ArgumentError("xs and ys must have the same length"))
    noise = _build_noise(model.noise_var, xs)
    gpx = model.gp(xs, noise)
    updated_gpx = posterior(gpx, ys)
    return LinearOperatorGP(model.gp, model.noise_var, model.ops, model.p, updated_gpx)
end

"""
    update(model::LinearOperatorGP, xs::AbstractVector, ys::AbstractVector{<:AbstractVector})

Update the LinearOperatorGP with full observations, where each input point has a
full vector of outputs (one per operator).

Arguments:
- `model::LinearOperatorGP`: The current model.
- `xs::AbstractVector`: Input points, one per observation location.
- `ys::AbstractVector{<:AbstractVector}`: Output vectors, one per input point, each of
    length p (one entry per operator).

returns:
- `LinearOperatorGP`: A new model conditioned on the provided data.
"""
function update(model::LinearOperatorGP, xs::AbstractVector, ys::AbstractVector{<:AbstractVector})
    x_tilde, y_tilde = prepare_isotopic_multi_output_data(xs, ColVecs(reduce(hcat, ys)))
    noise = _build_noise(model.noise_var, x_tilde)
    gpx = model.gp(x_tilde, noise)
    updated_gpx = posterior(gpx, y_tilde)
    return LinearOperatorGP(model.gp, model.noise_var, model.ops, model.p, updated_gpx)
end

"""
    posterior_mean(model::LinearOperatorGP, x, i=1)

Compute the posterior mean of the GP at input `x` for operator index `i`.

Arguments:
- `model::LinearOperatorGP`: The conditioned model.
- `x`: The input point.
- `i::Int`: (optional) The operator index, defaults to 1.

returns:
- `AbstractVector`: The posterior mean at `x` under operator `i`.
"""
function posterior_mean(model::LinearOperatorGP, x::Union{Real,AbstractVector}, i::Int=1)
    x = x isa Real ? [x] : x
    return mean(model.gpx([(x, i)]))
end

"""
    posterior_var(model::LinearOperatorGP, x, i=1)

Compute the posterior variance of the GP at input `x` for operator index `i`.

Arguments:
- `model::LinearOperatorGP`: The conditioned model.
- `x`: The input point.
- `i::Int`: (optional) The operator index, defaults to 1.

returns:
- `AbstractVector`: The posterior variance at `x` under operator `i`.
"""
function posterior_var(model::LinearOperatorGP, x::Union{Real,AbstractVector}, i::Int=1)
    x = x isa Real ? [x] : x
    return var(model.gpx([(x, i)]))
end

"""
    prep_input(model::LinearOperatorGP, xs)

Prepare input data into the tagged format expected by the multi-output GP.

Arguments:
- `model::LinearOperatorGP`: The model (used to read `p`).
- `xs`: Raw input points.

returns:
- Prepared input in `MOInputIsotopicByOutputs` or tagged-tuple format.
"""
prep_input(model::LinearOperatorGP, xs) = _prep_input(xs, model.p)

"""
    prep_output(model::LinearOperatorGP, ys::AbstractVector)

Flatten a vector of per-point output vectors into the column-major ordering expected
by `MOInputIsotopicByOutputs`.

Arguments:
- `model::LinearOperatorGP`: The model.
- `ys::AbstractVector`: A vector of output vectors, one per input point.

returns:
- `AbstractVector`: The flattened output vector in isotopic column-major order.
"""
function prep_output(model::LinearOperatorGP, ys::AbstractVector)
    return vec(permutedims(reduce(hcat, ys)))
end

"""
    _build_noise(noise_var::Real, xs::AbstractVector)

Build a uniform diagonal noise matrix from a scalar variance.

Arguments:
- `noise_var::Real`: A single noise variance applied to all observations.
- `xs::AbstractVector`: The vector of input points (used only to determine length).
    In the full observation case, `xs` is the prepped input vector of length n*p,
    as returned by `_prep_input`, where n is the number of observation locations
    and p is the number of operators.

returns:
- `Diagonal`: A diagonal noise matrix of size (n*p)×(n*p) with constant diagonal `noise_var`.
"""
function _build_noise(noise_var::Real, xs::AbstractVector)
    return Diagonal(fill(noise_var, length(xs)))
end

"""
    _build_noise(noise_var::Tuple, xs::AbstractVector{<:Tuple})

Build a diagonal noise matrix from a per-operator noise tuple, for heterogeneous
(tagged) inputs where each element of `xs` is a `(x, i)` tuple with operator index `i`.
In this case `xs` is the raw unprepped input vector, as different operators may be
observed at different locations.

Arguments:
- `noise_var::Tuple`: A tuple of length p, one noise variance per operator type.
- `xs::AbstractVector{<:Tuple}`: Tagged input points of the form `(x, i)`.

returns:
- `Diagonal`: A diagonal noise matrix with each entry set to `noise_var[i]`
    for the corresponding operator index `i`.
"""
function _build_noise(noise_var::Tuple, xs::AbstractVector{<:Tuple})
    return Diagonal([noise_var[i] for (_, i) in xs])
end

"""
    _build_noise(noise_var::Tuple, xs::AbstractVector)

Build a diagonal noise matrix from a per-operator noise tuple, for the full observation
case where all p operators are observed at every input location. `xs` is the prepped
input vector of length n*p, as returned by `_prep_input` via `MOInputIsotopicByOutputs`,
ordered as all outputs for location 1, then location 2, ..., location n.

Arguments:
- `noise_var::Tuple`: A tuple of length p, one noise variance per operator type.
- `xs::AbstractVector`: Prepped isotopic input points of length n*p (not tagged).

returns:
- `Diagonal`: A diagonal noise matrix of size (n*p)×(n*p) where each block of n
    consecutive entries corresponds to one operator's noise variance.
"""
function _build_noise(noise_var::Tuple, xs::AbstractVector)
    n = length(xs)
    p = length(noise_var)
    return Diagonal(repeat(collect(noise_var), inner = div(n, p)))
end

"""
    _build_noise(noise_var::AbstractVector, xs::AbstractVector)

Build a diagonal noise matrix from a per-observation noise vector.

Arguments:
- `noise_var::AbstractVector`: A vector of length n, one noise variance per observation.
    In the full observation case, n = n_locations * p and `xs` is the prepped input
    vector as returned by `_prep_input`.
- `xs::AbstractVector`: The vector of input points.

returns:
- `Diagonal`: A diagonal noise matrix with `noise_var` on the diagonal.

Throws:
- `ArgumentError`: If `length(noise_var) ≠ length(xs)`.
"""
function _build_noise(noise_var::AbstractVector, xs::AbstractVector)
    length(noise_var) == length(xs) || throw(ArgumentError(
        "noise_var length ($(length(noise_var))) must match number of observations ($(length(xs)))"
    ))
    return Diagonal(noise_var)
end

"""
    get_mean_std(model::LinearOperatorGP, y_train::AbstractVector, choice::String)

Compute the empirical mean and standard deviation of the training outputs, used for
standardizing the GP before training. Only the first output (function values) is used
for centering; gradient outputs are scaled by the same factor.

Arguments:
- `model::LinearOperatorGP`: The GP model.
- `y_train::AbstractVector`: A vector of output vectors, one per training point.
- `choice::String`: Standardization mode. `"scale_only"` sets mean to zero,
    `"mean_only"` sets standard deviation to one, anything else does both.

returns:
- `μ::Vector`: The empirical mean vector (zero for all but the first output).
- `σ::Vector`: The empirical standard deviation vector (same value for all outputs).
"""
function get_mean_std(model::LinearOperatorGP, y_train::AbstractVector, choice::String)
    y_mat = reduce(hcat, y_train)

    μ = vec(mean(y_mat; dims=2))
    μ[2:end] .= 0.0

    σ = vec(std(y_mat; dims=2))
    σ[σ .== 0.0] .= 1.0
    σ[2:end] .= σ[1]

    if choice == "scale_only"
        μ .= 0.0
    elseif choice == "mean_only"
        σ .= 1.0
    end

    return μ, σ
end

"""
    std_y(model::LinearOperatorGP, ys::AbstractVector, μ::AbstractVector, σ::AbstractVector)

Standardize the training outputs using the provided mean and standard deviation.

Arguments:
- `model::LinearOperatorGP`: The GP model.
- `ys::AbstractVector`: A vector of output vectors, one per training point.
- `μ::AbstractVector`: The mean vector to subtract.
- `σ::AbstractVector`: The standard deviation vector to divide by.

returns:
- `Vector`: A vector of standardized output vectors.
"""
function std_y(model::LinearOperatorGP, ys::AbstractVector, μ::AbstractVector, σ::AbstractVector)
    return [(y .- μ) ./ σ[1] for y in ys]
end

"""
    _get_minimum(model::LinearOperatorGP, ys::Vector{Y}) where {Y}

Extract the minimum function value (first output) across all training points.

Arguments:
- `model::LinearOperatorGP`: The GP model.
- `ys::Vector{Y}`: A vector of output vectors, one per training point.

returns:
- The minimum value of the first output across all training points.
"""
_get_minimum(model::LinearOperatorGP, ys::Vector{Y}) where {Y} = minimum(hcat(ys...)[1, :])[1]

"""
    get_lengthscale(model::LinearOperatorGP)

Get the lengthscale of the base kernel of the latent GP.

Arguments:
- `model::LinearOperatorGP`: The GP model.

returns:
- `Vector`: The lengthscale of the base kernel.
"""
get_lengthscale(model::LinearOperatorGP) = 1 ./ model.gp.kernel.base_kernel.kernel.transform.s

"""
    get_scale(model::LinearOperatorGP)

Get the output scale of the base kernel of the latent GP.

Arguments:
- `model::LinearOperatorGP`: The GP model.

returns:
- `Vector`: The output scale of the base kernel.
"""
get_scale(model::LinearOperatorGP) = model.gp.kernel.base_kernel.σ²

"""
    get_kernel_constructor(model::LinearOperatorGP)
Get the kernel constructor of the GP model.

Arguments:
- 'model::LinearOperatorGP': The GP model.

returns:
- 'get_kernel_constructor::Kernel': The kernel constructor of the GP model.
"""
get_kernel_constructor(model::LinearOperatorGP) = model.gp.kernel.base_kernel.kernel.kernel

"""
    nlml(model::LinearOperatorGP, params, xs::AbstractVector, ys::AbstractVector{<:AbstractVector})

Compute the negative log marginal likelihood (NLML) of the GP model given hyperparameters,
for the full observation case where all operators are observed at every input point.

Arguments:
- `model::LinearOperatorGP`: The GP model.
- `params`: Parameters containing the log lengthscale and log scale.
- `xs::AbstractVector`: The input data points.
- `ys::AbstractVector{<:AbstractVector}`: The observed values, one vector of length p
    per input point (one entry per operator).

returns:
- `Float64`: The negative log marginal likelihood of the model.
"""
function nlml(model::LinearOperatorGP, params, xs::AbstractVector, ys::AbstractVector{<:AbstractVector})
    log_ℓ, log_scale = params
    ℓ = exp(log_ℓ)
    scale = exp(log_scale)

    kernel_constructor::Kernel = get_kernel_constructor(model)
    k = scale * with_lengthscale(kernel_constructor, ℓ)

    gp = LinearOperatorGP(k, model.ops, model.noise_var; mean=model.gp.mean.base_mean)
    x_tilde, y_tilde = prepare_isotopic_multi_output_data(xs, ColVecs(reduce(hcat, ys)))
    noise = _build_noise(model.noise_var, x_tilde)
    gpx = gp.gp(x_tilde, noise)

    return -AbstractGPs.logpdf(gpx, y_tilde)
end

"""
    nlml(model::LinearOperatorGP, params, xs::AbstractVector{<:Tuple}, ys::AbstractVector{<:Real})

Compute the negative log marginal likelihood (NLML) of the GP model given hyperparameters,
for the partial observation case where each input is a tagged tuple `(x, i)` specifying
which operator was applied, allowing different operators to be observed at different points.

Arguments:
- `model::LinearOperatorGP`: The GP model.
- `params`: Parameters containing the log lengthscale and log scale.
- `xs::AbstractVector{<:Tuple}`: Tagged input points of the form `(x, i)`.
- `ys::AbstractVector{<:Real}`: Scalar observations corresponding to each tagged input.

returns:
- `Float64`: The negative log marginal likelihood of the model.
"""
function nlml(model::LinearOperatorGP, params, xs::AbstractVector{<:Tuple}, ys::AbstractVector{<:Real})
    log_ℓ, log_scale = params
    ℓ = exp(log_ℓ)
    scale = exp(log_scale)

    # Kernel with current parameters
    kernel_constructor::Kernel = get_kernel_constructor(model)
    k = scale * with_lengthscale(kernel_constructor, ℓ)

    # GP with current parameters
    gp = LinearOperatorGP(k, model.ops, model.noise_var; mean=model.gp.mean.base_mean)
    gpx = gp.gp(xs, model.noise_var)

    return -AbstractGPs.logpdf(gpx, ys)  # Negative log marginal likelihood
end


