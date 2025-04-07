struct ProbabilityImprovement <: AbstractAcquisition
    ξ::Float64
    best_y::Float64
end

function (pi::ProbabilityImprovement)(surrogate::AbstractSurrogate, x)
    μ = posterior_mean(surrogate, x)
    σ² = posterior_var(surrogate, x)
    Δ = (ei.best_y - ei.ξ) - μ # we are substracting ξ because we are minimising.
    
    max(σ²,0) == 0 && return max(Δ,0.0)

    σ = sqrt(σ²)

    z = (pi.best_y .- μ) ./ σ
    return normcdf(Δ/σ,1)
end

function update!(acqf::ProbabilityImprovement,ys::AbstractVector)
    if isa(ys[1],Float64) # we are in 1d
        ProbabilityImprovement(acqf.ξ, minimum(reduce(vcat,ys)))
    else 
        ProbabilityImprovement(acqf.ξ, minimum(hcat(ys...)[1,:]))
    end
end