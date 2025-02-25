struct ExpectedImprovement <: AbstractAcquisition
    ξ::Float64 # not implemented yet
    best_y::Float64
end

normpdf(μ, σ²) = 1 / √(2π * σ²) * exp(-μ^2 / (2 * σ²))
normcdf(μ, σ²) = 1 / 2 * (1 + erf(μ / √(2σ²)))

function (ei::ExpectedImprovement)(surrogate::StandardGP, x)
    μ = posterior_mean(surrogate, x)
    σ² = posterior_var(surrogate, x)
    Δ = (ei.best_y - ei.ξ) - μ # we are substracting ξ because we are minimising.

    max(σ²,0) == 0 && return max(Δ,0.0)

    σ = sqrt(max(σ²,0))

    return Δ*normcdf(Δ/σ,1) + σ*normpdf(Δ,1)
end

function update!(acqf::ExpectedImprovement,ys::AbstractVector)
    ExpectedImprovement(acqf.ξ, minimum(ys))
end