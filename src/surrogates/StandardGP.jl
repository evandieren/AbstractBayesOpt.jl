"""
    StandardGP.jl

Implementation of the Abstract structures for the standard GP.

Reason: This is a simple wrapper around AbstractGPs that implements the AbstractSurrogate abstract type
"""

struct StandardGP <: AbstractSurrogate
    gp::AbstractGPs.GP
    gpx
end

function StandardGP(gp::AbstractGPs.GP)
    StandardGP(gp,nothing)
end

function update!(model::StandardGP, xs::AbstractVector, ys::AbstractVector, noise_var)
    gpx = model.gp(xs, noise_var...)
    updated_gpx = posterior(gpx,ys)
    return StandardGP(model.gp, updated_gpx)
end

function posterior_mean(model,x)
    Statistics.mean(model.gpx([x]))[1]
end

function posterior_var(model,x)
    Statistics.var(model.gpx([x]))[1]
end