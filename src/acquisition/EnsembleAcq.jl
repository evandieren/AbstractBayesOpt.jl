struct EnsembleAcquisition <: AbstractAcquisition
    weights::Vector{Float64}
    acquisitions::Vector{AbstractAcquisition}

    function EnsembleAcquisition(weights::Vector{Float64},
        acqs::Vector{AbstractAcquisition})
        @assert length(weights) == length(acqs)  "weights and acquisitions must align"
        @assert all(w -> w >= 0, weights)        "weights must be non-negative"
        total = sum(weights)
        @assert total > 0                         "sum of weights must be positive"

        normalized = weights / total
        new(normalized, acqs) 
    end
end

Base.copy(EA::EnsembleAcquisition) = EnsembleAcquisition(copy(EA.weights), [Base.copy(acq) for acq in EA.acquisitions])

function (EA::EnsembleAcquisition)(surrogate::AbstractSurrogate, x, x_buf=nothing)
    sum(EA.weights[i] * EA.acquisitions[i](surrogate, x, x_buf) for i in eachindex(EA.weights))
end

function update!(acqf::EnsembleAcquisition, ys::AbstractVector, surrogate::AbstractSurrogate)
    new_acqs = [update!(acqf.acquisitions[i], ys, surrogate) for i in eachindex(acqf.acquisitions)]
    return EnsembleAcquisition(acqf.weights, new_acqs)
end