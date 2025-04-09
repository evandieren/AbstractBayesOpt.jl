struct ExpectedImprovement <: AbstractAcquisition
    ξ::Float64
    best_y::Float64
end

function (ei::ExpectedImprovement)(surrogate::AbstractSurrogate, x)
    μ = posterior_mean(surrogate, x)
    σ² = posterior_var(surrogate, x)
    Δ = (ei.best_y - ei.ξ) - μ # we are substracting ξ because we are minimising.

    max(σ²,0) == 0 && return max(Δ,0.0) # maybe remove the max(Δ,0.0)?

    σ = sqrt(σ²)

    return Δ*normcdf(Δ/σ,1) + σ*normpdf(Δ/σ,1)
end

function update!(acqf::ExpectedImprovement,ys::AbstractVector, surrogate::AbstractSurrogate)
    if isa(ys[1],Float64) # we are in 1d
        ExpectedImprovement(acqf.ξ, minimum(reduce(vcat,ys)))
    else 
        ExpectedImprovement(acqf.ξ, minimum(hcat(ys...)[1,:]))
    end
end