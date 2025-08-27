struct ExpectedImprovement <: AbstractAcquisition
    ξ::Float64
    best_y::Float64
end

function (ei::ExpectedImprovement)(surrogate::AbstractSurrogate, x, x_buf=nothing)

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
    Δ = (ei.best_y - ei.ξ) - μ # we are substracting ξ because we are minimising.

    if σ² <= 1e-12
        return max(Δ, 0.0)
    end

    σ = sqrt(σ²)

    z = Δ/σ

    return Δ * cdf(Normal(0, 1), z) + σ * pdf(Normal(0, 1), z)
end

function update!(acqf::ExpectedImprovement,ys::AbstractVector, surrogate::AbstractSurrogate)
    if (length(ys[1])==1) # we are in 1d
        ExpectedImprovement(acqf.ξ, minimum(reduce(vcat,ys)))
    else 
        ExpectedImprovement(acqf.ξ, minimum(hcat(ys...)[1,:]))
    end
end