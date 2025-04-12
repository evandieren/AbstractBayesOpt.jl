"""
    GradientGP.jl

Implementation of the Abstract structures for the gradient GP.

Reason: This is a simple wrapper around AbstractGPs that implements the AbstractSurrogate abstract type for the gradient GP
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

struct gradKernel <: MOKernel 
    base_kernel
    function gradKernel(Tk)
        return new(Tk)
    end
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

    Snippets given by Niklas Schmitz, EPFL.
    """
    (px > length(x) + 1 || py > length(y) + 1 || px < 1 || py < 1) &&
        error("`px` and `py` must be within the range of the number of outputs")
    
    onehot(n, i) = collect(1:n) .== i
    
    vi = onehot(length(x), px-1) # as px is 1 for func obvs, and goes from 2 to d+1 for gradients, so we need to substract 1
    vj = onehot(length(y), py-1) # same for py.

    val = px == 1 && py == 1 # we are looking at f(x), f(y)

    ∇_val_1 = (px != 1 && py == 1) # we are looking at ∇f(x)-f(y)
    ∇_val_2 = (px == 1 && py != 1) # we are looking at f(x)-∇f(y)

    if val # we are just computing the usual matrix K
        κ.base_kernel(x,y)
    elseif ∇_val_1
        return ForwardDiff.derivative(h -> κ.base_kernel(x + h * vi, y), 0.)
    elseif ∇_val_2 # we are looking at f(x)-∇f(y)
        return ForwardDiff.derivative(h -> κ.base_kernel(x, y + h * vj), 0.)
    else # we are looking at ∇f(x)-∇f(y), this avoids computing the entire hessian each time.
        return ForwardDiff.derivative(h1 -> ForwardDiff.derivative(h2 -> κ.base_kernel(x + h1 * vi, y + h2 * vj), 0.), 0.)
    end
end

function GradientGP(kernel::gradKernel,p::Int,noise_var::Float64;mean=ZeroMean())
    """
    Initialises the model for Gradient GPs (multi-output GP)
    """
    gp = AbstractGPs.GP(mean,kernel) # Creates GP(0,k) for the prior
    GradientGP(gp,noise_var,p,nothing)
end

function update!(model::GradientGP, xs::AbstractVector, ys::AbstractVector)

    x̃, ỹ = KernelFunctions.MOInputIsotopicByOutputs(xs, size(ys[1])[1]), vec(permutedims(reduce(hcat, ys)))
    # we could do something better for this, such as inserting the batch of new points in xs and ys which are already MOInputIsotopicByOutputs elements.

    gpx = model.gp(x̃, model.noise_var...)
    updated_gpx = posterior(gpx,ỹ)

    return GradientGP(model.gp, model.noise_var, model.p, updated_gpx)
end

prep_input(model::GradientGP, x::AbstractVector) = KernelFunctions.MOInputIsotopicByOutputs(x, model.p)

posterior_mean(model::GradientGP,x) = mean(model.gpx(prep_input(model, [x])))[1] # we do the function value only for now

posterior_var(model::GradientGP,x) = var(model.gpx(prep_input(model, [x])))[1] # we do the function value only for now