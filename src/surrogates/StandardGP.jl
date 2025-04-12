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

prep_input(model::StandardGP,x::AbstractVector) = x

posterior_mean(model::StandardGP,x) = Statistics.mean(model.gpx([x]))[1]

posterior_var(model::StandardGP,x) = Statistics.var(model.gpx([x]))[1]