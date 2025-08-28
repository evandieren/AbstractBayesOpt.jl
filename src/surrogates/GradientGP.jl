"""
    GradientGP.jl

Implementation of the Abstract structures for the gradient GP.


This relies on MOGP from AbstractGPs.jl and KernelFunctions.jl.
As we are leveraging AutoDiff with Matern 5/2 kernel, we need to approximate it around d ≈ 0 because of differentiation issues.
"""

struct GradientGP <: AbstractSurrogate
    gp::AbstractGPs.GP
    noise_var::Float64
    p::Int
    gpx
end

Base.copy(s::GradientGP) = GradientGP(s.gp, s.noise_var ,s.p, copy(s.gpx))

# Need to approximate around d ≈ 0 because of differentiation issues.
# We will use the squared euclidean distance because this is fine to differentiate when d ≈ 0.
struct ApproxMatern52Kernel{M} <: KernelFunctions.SimpleKernel
    metric::M
end
ApproxMatern52Kernel(; metric=Distances.SqEuclidean()) = ApproxMatern52Kernel(metric)
KernelFunctions.metric(k::ApproxMatern52Kernel) = k.metric

function KernelFunctions.kappa(k::ApproxMatern52Kernel, d²::Real)
    if d² < 1e-10 # we do Taylor of order 2 around d = 0.
        return 1.0 - (5.0/6.0) * d²
    else
        d = sqrt(d²)
        return (1 + sqrt(5) * d + 5 * d² / 3) * exp(-sqrt(5) * d)
    end
end
function Base.show(io::IO, k::ApproxMatern52Kernel)
    return print(io, "Matern 5/2 Kernel, quadratic approximation around d=0 (metric = ", k.metric, ")")
end

struct gradMean
    c::AbstractVector
    function f_mean(vec_const, (x, px)::Tuple{Any,Int})
        return vec_const[px]
    end

    function gradMean(c)
        return CustomMean(x -> f_mean(c,x))
    end
end

mutable struct gradKernel{K} <: MOKernel 
    base_kernel::K
    #function gradKernel(Tk)
    #    return new(Tk)
    #end
end

function (κ::gradKernel)((x, px)::Tuple{Any,Int}, (y, py)::Tuple{Any,Int})
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
        κ.base_kernel(x,y)
    elseif ∇_val_1
        #vi = onehot(length(x), px-1) # as px is 1 for func obvs, and goes from 2 to d+1 for gradients, so we need to substract 1
        return ForwardDiff.derivative(h -> κ.base_kernel(x .+ h .* (1:length(x) .== (px-1)), y), 0.)
    elseif ∇_val_2 # we are looking at f(x)-∇f(y)
        #vj = onehot(length(y), py-1) # same for py.
        return ForwardDiff.derivative(h -> κ.base_kernel(x, y .+ h .* (1:length(y) .== (py-1))), 0.)
    else # we are looking at ∇f(x)-∇f(y), this avoids computing the entire hessian each time.
        #_vi = onehot(length(x), px-1); _vj = onehot(length(y), py-1)
        return ForwardDiff.derivative(h1 -> ForwardDiff.derivative(h2 -> κ.base_kernel(x .+ h1 .* (1:length(x) .== (px-1)), 
                        y .+ h2 .* (1:length(y) .== (py-1))), 0.), 0.)
    end
end

"""
    GradientGP(kernel::gradKernel,p::Int,noise_var::Float64;mean=ZeroMean())

Constructor for the GradientGP model.

Arguments:
- `kernel::gradKernel`: The gradient kernel function to be used in the GP.
- `p::Int`: The number of outputs (1 for function value + d for gradients).
- `noise_var::Float64`: The noise variance of the observations.
- `mean`: (optional) The mean function of the GP, defaults to ZeroMean()

returns:
- `GradientGP`: An instance of the GradientGP model.
"""
function GradientGP(kernel::gradKernel,p::Int,noise_var::Float64;mean=ZeroMean())
    gp = AbstractGPs.GP(mean,kernel) # Creates GP(0,k) for the prior
    GradientGP(gp,noise_var,p,nothing)
end

"""
    update!(model::GradientGP, xs::AbstractVector, ys::AbstractVector)

Update the GP model with new data points (xs, ys).

Arguments:
- `model::GradientGP`: The current GP model.
- `xs::AbstractVector`: A vector of input points where the function has been evaluated.
- `ys::AbstractVector`: A vector of corresponding function values and gradients at the input points

returns:
- `GradientGP`: A new GradientGP model updated with the provided data.
"""
function update!(model::GradientGP, xs::AbstractVector, ys::AbstractVector)

    x̃, ỹ = KernelFunctions.MOInputIsotopicByOutputs(xs, size(ys[1])[1]), vec(permutedims(reduce(hcat, ys)))
    # we could do something better for this, such as inserting the batch of new points in xs and ys which are already MOInputIsotopicByOutputs elements.

    gpx = model.gp(x̃, model.noise_var...)
    updated_gpx = posterior(gpx,ỹ)

    return GradientGP(model.gp, model.noise_var, model.p, updated_gpx)
end

# helper for the nlml computation to avoid recomputing C but it's needed at each step...
# function fastnlml_grad(gpx,Y)
#     m = mean(gpx)
#     T = promote_type(eltype(m), eltype(gpx.data.C),eltype(Y))
#     return -((size(Y, 1) * T(AbstractGPs.log2π) + logdet(gpx.data.C)) .+ AbstractGPs._sqmahal(m, gpx.data.C, Y)) ./ 2
# end

"""
    nlml(mod::GradientGP,params,kernel,x,y;mean=ZeroMean())

Compute the negative log marginal likelihood (NLML) of the GP model given hyperparameters.

Arguments:
- `mod::GradientGP`: The GP model.
- `params::Tuple`: A tuple containing the log lengthscale and log scale parameters.
- `kernel`: The kernel function used in the GP.
- `x`: The input data points.
- `y`: The observed function values and gradients.
- `mean`: (optional) The mean function of the GP, defaults to ZeroMean()

returns:
- nlml : The negative log marginal likelihood of the model.
"""
function nlml(mod::GradientGP,params,kernel::Kernel,x::AbstractVector,y::AbstractVector;mean=ZeroMean())
    log_ℓ, log_scale = params
    ℓ = exp(log_ℓ)
    scale = exp(log_scale)

    # Kernel with current parameters
    k = scale * (kernel ∘ ScaleTransform(1/ℓ))
    #println("creation time of gradgp")
    gp = GradientGP(gradKernel(k),mod.p, mod.noise_var,mean=mean)

    #println("finite gpx time")
    gpx = gp.gp(x,mod.noise_var)

    #println("logpdf")
    #@time fastnlml_grad(gpx,y)
    -AbstractGPs.logpdf(gpx, y)  # Negative log marginal likelihood
end


"""
    standardize_y(mod::GradientGP,y_train::AbstractVector)

Standardize the output values (y_train) for the GradientGP model.

Arguments:
- `mod::GradientGP`: The GP model.
- `y_train::AbstractVector`: A vector of observed function values and gradients.

returns:
- `ys_std`: A vector of standardized output values.
- `μ`: The mean of the original output values.
- `σ`: The standard deviation of the original output values.
"""
function standardize_y(mod::GradientGP,y_train::AbstractVector)
    y_mat = reduce(hcat, y_train)

    μ = vec(mean(y_mat; dims=2))
    μ[2:end] .= 0.0  # Only standardize function values, not gradients
    σ = vec(std(y_mat; dims=2))
    
    # Protect against very small standard deviations
    if σ[1] < 1e-12
        @warn "Very small standard deviation detected: $(σ[1]). Using std = 1.0"
        σ[1] = 1.0
    end
    
    σ[2:end] .= σ[1]  # Use same scaling for gradients
    
    ys_std = [(y .- μ) ./ σ for y in y_train]
    # this re-creates a Vector{Vector{Float64}}, which is what we need
    return ys_std, μ, σ
end

get_lengthscale(model::GradientGP) = 1 ./ model.gp.kernel.base_kernel.kernel.transform.s

get_scale(model::GradientGP) = model.gp.kernel.base_kernel.σ²

prep_input(model::GradientGP, x::AbstractVector) = KernelFunctions.MOInputIsotopicByOutputs(x, model.p)


posterior_mean(model::GradientGP,x_buf::Vector{Tuple{Vector{Float64}, Int}}) = mean(model.gpx(x_buf))[1]

posterior_var(model::GradientGP,x_buf::Vector{Tuple{Vector{Float64}, Int}}) = var(model.gpx(x_buf))[1]

# posterior_mean(model::GradientGP,x) = mean(model.gpx([(x,1)]))[1]

posterior_grad_mean(model::GradientGP,x) = mean(model.gpx(prep_input(model, x))) # the whole vector

# posterior_var(model::GradientGP,x) = var(model.gpx([(x,1)]))[1] # we do the function value only for now

posterior_grad_var(model::GradientGP,x) = var(model.gpx(prep_input(model, x)))

posterior_grad_cov(model::GradientGP,x) = cov(model.gpx(prep_input(model, x))) # the matrix itself


"""
    unstandardized_mean_and_var(gp::GradientGP, X, params::Tuple)

Compute the unstandardized mean and variance of the GP predictions at new input points.

Arguments:
- `gp::GradientGP`: The GP model.
- `X`: A vector of new input points where predictions are to be made.
- `params::Tuple`: A tuple containing the mean and standard deviation used for standardization.

returns:
- `m_unstd`: The unstandardized mean predictions at the input points.
- `v_unstd`: The unstandardized variance predictions at the input points.
"""
function unstandardized_mean_and_var(gp::GradientGP, X, params::Tuple)
    μ, σ = params[1][1], params[2][1]
    m, v = mean_and_var(gp.gpx(X))
    # Un-standardize mean and variance
    m_unstd = (m .* σ) .+ μ
    v_unstd = v .* (σ.^2)
    return m_unstd, v_unstd
end