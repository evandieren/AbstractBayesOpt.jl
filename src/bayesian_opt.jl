module bayesian_opt

"""
This module contains the structures and functions for the Bayesian Optimization framework.

BOProblem (struct) : structure containing the required information to run the Bayesian Optimization algorithm.
BOProblem (function) : Initiate a BOProblem structure
"""

struct BOProblem{S<:AbstractSurrogate,A::AbstractAcquisition}
    f
    domain
    xs::AbstractMatrix
    ys::AbstractVector
    prior::S
    gp::S
    acqf::A
    noise
    hyperparams_optim
end

function BOProblem(f, domain, prior::AbstractSurrogate, acqf_updated::AbstractAcquisition, noise, hyperparams_optim)
    # Infer input types.
    domain_type = typeof(domain[:lb])
    domain_eltype = eltype(domain[:lb])
    dim = size(domain[:lb])[1]

    # Dry run to determine output type. In the future should check
    # for type stability in f.
    # output_type = Base.return_types(f, (domain_type,))[1]
    output_type = typeof(f(Zeros{domain_eltype}(dim)))

    xs = ElasticArray{domain_eltype}(undef, dim, 0)
    ys = ElasticArray{output_type}(undef, 0)
    # Initialize the posterior with prior
    BOProblem(f, domain, xs, ys, prior, prior, acqf_updated, noise,
              hyperparams_optim)
end

function update(p::BOProblem, x, y)

    # Add the obserbed data
    append!(p.xs, x)
    append!(p.ys, [y])
    # Could create some issues if we have the same point twice.

    # Update the GP (finite GP)
    gp_updated = update_model(p, p.xs, p.ys, p.noise, p.hyperparams_optim)

    BOProblem(p.f, p.domain, p.xs, p.ys, p.prior, gp_updated,
              p.acqf, p.noise, p.hyperparams_optim)
    # Is this really necessary? Why not returning p directly with the updated xs,ys and gp?
end

function update_model(p::BOProblem{<:AbstractSurrogate,<:AbstractAcquisition}, xs, ys, noise, ::StaticHyperparams)
    gpx = p.prior.gp(ColVecs(xs), noise...) # This fits the AbstractGPs.GP defined in GPSurrogates amd returns a FiniteGP
    posterior(gpx, ys) # This calls AbstractGPs.posterior
end

function bayesian_optimize!(p::BOProblem, acqf_options)
    """
    This function implements the EGO framework: 
        While some criterion is not met, (1) optimize the acquisition function to obtain 
        the new best candidate, (2) query the target function f, (3) update the GP and the overall optimization state. 

    """

    while !stop_criteria(p)
        _, x_cand, _ = optimize_function(x -> p.acqf(x, p.gp),
                                                p.domain,
                                                BBoxOptimizer();
                                                optim_options=acqf_optim_options)
        y_cand = p.f(x_cand)
        p = update(p, x_cand, y_cand)
    end
    return p
end


end