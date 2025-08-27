struct UpperConfidenceBound <: AbstractAcquisition
     β::Float64  # exploration-exploitation balance parameter
end

function (ucb::UpperConfidenceBound)(surrogate::AbstractSurrogate, x, x_buf=nothing)

    # Allocate buffer if not provided
    if x_buf === nothing
        if surrogate isa GradientGP
            x_buf = [(copy(x), 1)]
        else
            x_buf = [copy(x)]
        end
    else
        # Reuse buffer
        if surrogate isa GradientGP
            x_buf[1][1] .= x  # copy x into the tuple buffer
        else
            x_buf[1] .= x  # copy into 1×d matrix
        end
    end

    μ = posterior_mean(surrogate, x_buf)
    σ² = posterior_var(surrogate, x_buf)
    return -μ + ucb.β*σ²
end

function update!(acqf::UpperConfidenceBound,ys::AbstractVector, surrogate::AbstractSurrogate)
    acqf
end