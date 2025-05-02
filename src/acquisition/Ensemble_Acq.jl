# --- Ensemble of multiple acquisitions ---
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
        new(normalized, acqs)      # <-- calls the *inner* constructor
    end
end

function (ea::EnsembleAcquisition)(surrogate::AbstractSurrogate, x)
    sum(ea.weights[i] * ea.acquisitions[i](surrogate, x) for i in eachindex(ea.weights))
end

function update!(ea::EnsembleAcquisition, ys::AbstractVector, surrogate::AbstractSurrogate)
    new_acqs = [update!(ea.acquisitions[i], ys, surrogate) for i in eachindex(ea.acquisitions)]
    return EnsembleAcquisition(ea.weights, new_acqs)
end