struct ProbabilityImprovement <: AbstractAcquisition
    ξ::Float64
    best_y::Float64
end

function (pi::ProbabilityImprovement)(surrogate::AbstractSurrogate, x, x_buf=nothing)

    # Allocate buffer if not provided
    if x_buf === nothing
        x_buf = reshape(x, 1, :)   # create 1×d buffer
    else
        x_buf[1, :] .= x           # reuse existing buffer
    end

    μ = posterior_mean(surrogate, x)
    σ² = posterior_var(surrogate, x)
    Δ = (pi.best_y - pi.ξ) - μ # we are substracting ξ because we are minimising.
    
    max(σ²,0) == 0 && return max(Δ,0.0)

    σ = sqrt(σ²)

    z = (pi.best_y .- μ) ./ σ
    return normcdf(Δ/σ,1)
end

function update!(acqf::ProbabilityImprovement,ys::AbstractVector, surrogate::AbstractSurrogate)
    if isa(ys[1],Float64) # we are in 1d
        ProbabilityImprovement(acqf.ξ, minimum(reduce(vcat,ys)))
    else 
        ProbabilityImprovement(acqf.ξ, minimum(hcat(ys...)[1,:]))
    end
end