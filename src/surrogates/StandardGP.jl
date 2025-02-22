"""
    StandardGP.jl

Implementation of the Abstract structures for the standard GP.

Reason: This is a simple wrapper around AbstractGPs that implements the AbstractSurrogate abstract type
"""

struct StandardGP <: AbstractSurrogate
    gp::AbstractGPs.GP
    gpx:AbstractGPs.FiniteGP
end

function update!(model::StandardGP, xs, ys, noise, ::StaticHyperparams)
    gpx = model.gp(ColVecs(xs), noise...)
    updated_gpx = posterior(gpx,ys)
    return StandardGP(model.gp, updated_gpx)
end

function posterior_mean(model,x)
    Statistics.mean(model.gpx(x))
end

function posterior_var(model,x)
    Statistics.var(model.gpx(x))
end