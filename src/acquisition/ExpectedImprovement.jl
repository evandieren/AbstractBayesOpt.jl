struct ExpectedImprovement <: AbstractAcquisition
    ξ::Float64
    best_y::Float64
end

function (ei::ExpectedImprovement)(surrogate::AbstractSurrogate, x)
    μ = posterior_mean(surrogate, x)
    σ² = posterior_var(surrogate, x)
    σ = sqrt(σ²)
    γ = (μ - ei.best_y - ei.ξ) / σ
    return σ * (γ * normcdf(γ) + normpdf(γ))
end

function update!(acqf::ExpectedImprovement,ys::AbstractVector)
    ExpectedImprovement(acqf.ξ, min(ys))
end