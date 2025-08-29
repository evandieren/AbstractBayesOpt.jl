"""
    ProbabilityImprovement(ξ::Float64, best_y::Float64)

Probability of Improvement acquisition function.

Arguments:
- `ξ::Float64`: Exploration parameter
- `best_y::Float64`: Best observed objective value

returns:
- `PI::ProbabilityImprovement`: Probability of Improvement acquisition function instance

References:
[Kushner, 1964](https://asmedigitalcollection.asme.org/fluidsengineering/article/86/1/97/392213/A-New-Method-of-Locating-the-Maximum-Point-of-an)
"""
struct ProbabilityImprovement <: AbstractAcquisition
    ξ::Float64
    best_y::Float64
end

function (PI::ProbabilityImprovement)(surrogate::AbstractSurrogate, x, x_buf=nothing)

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
    Δ = (PI.best_y -PI.ξ) - μ # we are substracting ξ because we are minimising.
    
    max(σ²,0) == 0 && return max(Δ,0.0)

    σ = sqrt(σ²)

    z = (PI.best_y .- μ) ./ σ
    return normcdf(Δ/σ,1)
end


"""
    update!(acqf::ProbabilityImprovement, ys::AbstractVector, surrogate::AbstractSurrogate)

Update the Probability of Improvement acquisition function with new array of observations.

Arguments:
- `acqf::ProbabilityImprovement`: Current Probability of Improvement acquisition function
- `ys::AbstractVector`: Array of updated observations
- `surrogate::AbstractSurrogate`: Surrogate model

returns:
- `PI::ProbabilityImprovement`: Updated Probability of Improvement acquisition function
"""
function update!(acqf::ProbabilityImprovement,ys::AbstractVector, surrogate::AbstractSurrogate)
    if isa(ys[1],Float64) # we are in 1d
        ProbabilityImprovement(acqf.ξ, minimum(reduce(vcat,ys)))
    else 
        ProbabilityImprovement(acqf.ξ, minimum(hcat(ys...)[1,:]))
    end
end