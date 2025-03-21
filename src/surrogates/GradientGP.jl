"""
    GradientGP.jl

Implementation of the Abstract structures for the gradient GP.

Reason: This is a simple wrapper around AbstractGPs that implements the AbstractSurrogate abstract type for the gradient GP
"""

struct GradientGP <: AbstractSurrogate
    gp::AbstractGPs.GP
    p::Int
    gpx
end

# We need to define a new type of kernel for multi-ouput GPs.
struct gradKernel <: MOKernel 
    base_kernel
    ∂ₓ_kernel
    function gradKernel(Tk,∂ₓk)
        return new(Tk,∂ₓk)
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
    """
    (px > length(x) + 1 || py > length(y) + 1 || px < 1 || py < 1) &&
        error("`px` and `py` must be within the range of the number of outputs")

    val = px == 1 && py == 1 # we are looking at f(x), f(y)

    ∇_val_1 = (px != 1 && py == 1) # we are looking at ∇f(x)-f(y)
    ∇_val_2 = (px == 1 && py != 1) # we are looking at f(x)-∇f(y)

    if val # we are just computing the usual matrix K
        return κ.base_kernel(x,y)
    elseif ∇_val_1
        return ForwardDiff.derivative(s-> κ.base_kernel([x[1:px-2]; s; x[px:end]], y),x[px-1]) # the px-1 is because the observations are labeled 1, so first element of gradient is px=2, so will be px-1
    elseif ∇_val_2 # we are looking at f(x)-∇f(y)
        return ForwardDiff.derivative(s-> κ.base_kernel(x, [y[1:py-2]; s; y[py:end]]),y[py-1])
    else # we are looking at ∇f(x)-∇f(y), this avoids computing the entire hessian each time.
    
        return ForwardDiff.derivative(s-> κ.∂ₓ_kernel(x, [y[1:py-2]; s; y[py:end]],px-1),y[py-1])
    end
end

function GradientGP(kernel::gradKernel,p::Int)
    """
    Initialises the model for Gradient GPs (multi-output GP)
    """
    gp = AbstractGPs.GP(kernel) # Creates GP(0,k) for the prior
    GradientGP(gp,p,nothing)
end

function update!(model::GradientGP, xs::AbstractVector, ys::AbstractVector, noise_var::Float64)

    x̃, ỹ = KernelFunctions.MOInputIsotopicByOutputs(xs, size(ys[1])[1]), vec(permutedims(reduce(hcat, ys)))
    # we could do something better for this, such as inserting the batch of new points in xs and ys which are already MOInputIsotopicByOutputs elements.

    gpx = model.gp(x̃, noise_var...)
    updated_gpx = posterior(gpx,ỹ)

    return GradientGP(model.gp,model.p, updated_gpx)
end

prep_input(model::GradientGP, x::AbstractVector) = KernelFunctions.MOInputIsotopicByOutputs(x, model.p)

posterior_mean(model::GradientGP,x) = mean(model.gpx(prep_input(model, [x])))[1] # we do the function value only for now

posterior_var(model::GradientGP,x) = var(model.gpx(prep_input(model, [x])))[1] # we do the function value only for now