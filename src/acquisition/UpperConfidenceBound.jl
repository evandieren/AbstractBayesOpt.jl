struct UpperConfidenceBound <: AbstractAcquisition
     β::Float64  # exploration-exploitation balance parameter
end

function (ucb::UpperConfidenceBound)(surrogate::AbstractSurrogate, x)
    μ = posterior_mean(surrogate, x)
    σ² = posterior_var(surrogate, x)
    return -μ + ucb.β*σ²
end

function update!(acqf::UpperConfidenceBound,ys::AbstractVector, surrogate::AbstractSurrogate)
    acqf
end