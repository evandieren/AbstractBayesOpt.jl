struct ExpectedImprovement <: AbstractAcquisition
    ξ::Float64
end

function ExpectedImprovement(gp_x,options)
    μ, σ = posterior_stats(model, candidates)
    improvement = μ .- best_observed(model) .- ei.ξ
    Φ = normcdf.(improvement ./ σ)
    φ = normpdf.(improvement ./ σ)
    return σ .* (improvement ./ σ .* Φ .+ φ)
end

#function acquire(ei::ExpectedImprovement, model, candidates)
    
#end