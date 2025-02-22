module bayesian_opt

"""
This module contains the structures and functions for the Bayesian Optimization framework.

BOProblem (struct) : structure containing the required information to run the Bayesian Optimization algorithm.
BOProblem (function) : Initiate a BOProblem structure
update (for BOProblem) : updates the BOProblem once you have new values x,y
"""

struct BOProblem{T<:AbstractSurrogate,F<:Function,A<:AbstractAcquisition}
    f::F
    domain::AbstractDomain
    xs::AbstractMatrix
    ys::AbstractVector
    gp::T
    acqf::A
    max_iter::Int
    iter::Int
    noise
    hyperparams_optim
end

function BOProblem(f, domain, prior::AbstractSurrogate, acqf::AbstractAcquisition, max_iter, noise, hyperparams_optim)
    # Infer input types.
    #domain_type = typeof(domain[:lb])
    domain_eltype = eltype(domain[:lb])
    dim = size(domain[:lb])[1]

    # Dry run to determine output type. In the future should check
    # for type stability in f.
    # output_type = Base.return_types(f, (domain_type,))[1]
    output_type = typeof(f(Zeros{domain_eltype}(dim)))

    xs = ElasticArray{domain_eltype}(undef, dim, 0)
    ys = ElasticArray{output_type}(undef, 0)
    # Initialize the posterior with prior
    BOProblem(f, domain, xs, ys, prior, acqf,max_iter, 0, noise, hyperparams_optim)
end

function update!(p::BOProblem, x, y, i::Int)

    # Add the obserbed data
    append!(p.xs, x)
    append!(p.ys, [y])
    # Could create some issues if we have the same point twice.

    # Update the surrogate
    gp_udated = update!(p.gp,p.xs, p.ys, p.noise, p.hyperparams_optim)
    acqf_updated = update!(p.acqf,p.ys)

    #Is this really necessary? Why not returning p directly with the updated xs,ys and gp?
    # Do we need to create a new object everytime?
    return BOProblem(p.f, p.domain, p.xs, p.ys, gp_udated, acqf_updated,p.max_iter,i,p.noise, p.hyperparams_optim)
end

function stop_criteria(p::BOProblem)
    return p.i > p.max_iter
end


# Looping routine

function bayesian_optimize!(p::BOProblem)
    """
    This function implements the EGO framework: 
        While some criterion is not met, (1) optimize the acquisition function to obtain 
        the new best candidate, (2) query the target function f, (3) update the GP and the overall optimization state. 

    """
    i = 0
    while !stop_criteria(p)
        x_cand = optimize_acquisition!(p.acqf,p.gp,p.domain)
        y_cand = p.f(x_cand)
        i +=1 
        p = update(p, x_cand, y_cand, i)
    end
    return p
end


end