"""
    EnsembleAcquisition(weights::Vector{Float64}, acqs::Vector{AbstractAcquisition}) <: AbstractAcquisition

An ensemble acquisition function combines multiple acquisition functions, each weighted by a specified factor,

All weights must be non-negative.
"""
struct EnsembleAcquisition <: AbstractAcquisition
    weights::Vector{Float64}
    acquisitions::Vector{AbstractAcquisition}

    function EnsembleAcquisition(
        weights::Vector{Float64}, acqs::Vector{AbstractAcquisition}
    )
        @assert length(weights)==length(acqs) "weights and acquisitions must align"
        @assert all(w -> w >= 0, weights) "weights must be non-negative"
        total = sum(weights)
        @assert total>0 "sum of weights must be positive"

        normalized = weights / total
        new(normalized, acqs)
    end
end

"""
    Base.copy(EA::EnsembleAcquisition)

Creates a copy of the EnsembleAcquisition instance.

returns:
- `new_EA::EnsembleAcquisition`: A new instance of EnsembleAcquisition with copied weights and acquisitions.
"""
function Base.copy(EA::EnsembleAcquisition)
    EnsembleAcquisition(copy(EA.weights), [Base.copy(acq) for acq in EA.acquisitions])
end

"""
    (EA::EnsembleAcquisition)(surrogate::AbstractSurrogate, x::AbstractVector)

Evaluate the ensemble acquisition function at a given set of points.

Arguments:
- `surrogate::AbstractSurrogate`: The surrogate model used by the acquisition functions.
- `x::AbstractVector`: Vector of input points where the acquisition function is evaluated.

returns:
- `value::AbstractVector`: The weighted sum of the individual acquisition function evaluations at points `x`.
"""
function (EA::EnsembleAcquisition)(surrogate::AbstractSurrogate, x::AbstractVector)
    sum([EA.weights[i] .* EA.acquisitions[i](surrogate, x) for i in eachindex(EA.weights)])
end

function update(acq::EnsembleAcquisition, ys::AbstractVector, surrogate::AbstractSurrogate)
    new_acqs = [
        update(acq.acquisitions[i], ys, surrogate) for i in eachindex(acq.acquisitions)
    ]
    return EnsembleAcquisition(acq.weights, new_acqs)
end
