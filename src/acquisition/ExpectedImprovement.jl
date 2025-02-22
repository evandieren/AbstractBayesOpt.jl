struct ExpectedImprovement <: AbstractAcquisition
    ξ::Float64 # not implemented yet
    best_y::Float64
end

normpdf(μ, σ²) = 1 / √(2π * σ²) * exp(-μ^2 / (2 * σ²))
normcdf(μ, σ²) = 1 / 2 * (1 + erf(μ / √(2σ²)))

function (ei::ExpectedImprovement)(surrogate::StandardGP, x)
    μ = posterior_mean(surrogate, x)
    σ² = posterior_var(surrogate, x)

    max(σ²,0) == 0 && return max(μ - ei.best_y,0.0)



    σ = sqrt(max(σ²,0))
    γ = (μ - ei.best_y - ei.ξ) / σ

    return σ * (γ * normcdf(γ,1) + normpdf(γ,1))
end

function update!(acqf::ExpectedImprovement,ys::AbstractVector)
    ExpectedImprovement(acqf.ξ, minimum(ys))
end