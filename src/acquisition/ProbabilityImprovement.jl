struct ProbabilityImprovement <: AbstractAcquisition
    best_y::Float64
end

function (pi::ProbabilityImprovement)(surrogate::AbstractSurrogate, x)
    μ = posterior_mean(surrogate, x)
    σ² = posterior_var(surrogate, x)
    max(σ²,0) == 0 && return float(μ > pi.best_y)

    z = (μ .- pi.best_y) ./ σ
    return normcdf.(z)
end

function update!(acqf::ProbabilityImprovement,ys::AbstractVector)
    ProbabilityImprovement(maximum(ys))
end