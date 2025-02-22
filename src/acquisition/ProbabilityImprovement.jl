struct ProbabilityImprovement <: AbstractAcquisition
    best_y::Float64
end

function (ei::ExpectedImprovement)(surrogate::AbstractSurrogate, x)
    μ = posterior_mean(surrogate, x)
    σ = sqrt(posterior_var(surrogate, x))
    z = (μ .- pi.threshold) ./ σ
    return normcdf.(z)
end

function update!(acqf::ProbabilityImprovement,ys::AbstractVector)
    ProbabilityImprovement(min(ys))
end