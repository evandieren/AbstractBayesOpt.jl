struct ProbabilityImprovement <: AbstractAcquisition
    threshold::Float64
end

function acquire(pi::ProbabilityImprovement, model, candidates)
    μ, σ = posterior_stats(model, candidates)
    z = (μ .- pi.threshold) ./ σ
    return normcdf.(z)
end