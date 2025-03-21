"""
    StandardGP.jl

Implementation of the Abstract structures for the standard GP.

Reason: This is a simple wrapper around AbstractGPs that implements the AbstractSurrogate abstract type
"""

struct StandardGP <: AbstractSurrogate
    gp::AbstractGPs.GP
    gpx
end

function StandardGP(kernel::Kernel)
    """
    Initialises the model
    """
    gp = AbstractGPs.GP(kernel) # Creates GP(0,k) for the prior
    StandardGP(gp,nothing)
end

function update!(model::StandardGP, xs::AbstractVector, ys::AbstractVector, noise_var::Float64)
    gpx = model.gp(xs, noise_var...)
    updated_gpx = posterior(gpx,reduce(vcat,ys))
    return StandardGP(model.gp, updated_gpx)
end

prep_input(model::StandardGP,x::AbstractVector) = x

posterior_mean(model::StandardGP,x) = Statistics.mean(model.gpx([x]))[1]

posterior_var(model::StandardGP,x) = Statistics.var(model.gpx([x]))[1]