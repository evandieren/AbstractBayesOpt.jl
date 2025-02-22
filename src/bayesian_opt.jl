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
    noise::Float64
    flag::Bool
end

function print_info(p::BOProblem)
    println("== Printing information about the BOProblem ==")
    println("Target function: ",p.f)
    println("Domain: ",p.domain)
    println("xs: ",p.xs)
    println("ys: ",p.ys)
    println("Surrogate: ",p.gp.gp)
    println("ACQF: ",p.acqf)
    println("max_iter: ",p.max_iter)
    println("noise: ",p.noise)
end

function BOProblem(f::Function, domain::AbstractDomain, prior::AbstractSurrogate, acqf::AbstractAcquisition, max_iter::Int, noise::Float64)
    # Infer input types.
    #domain_type = typeof(domain[:lb])
    domain_eltype = eltype(domain.lower)
    dim = size(domain.lower)[1]

    # Dry run to determine output type. In the future should check
    # for type stability in f.
    # output_type = Base.return_types(f, (domain_type,))[1]
    output_type = typeof(f(Zeros{domain_eltype}(dim)))

    xs = ElasticArray{domain_eltype}(undef, dim, 0)
    ys = ElasticArray{output_type}(undef, 0)
    # Initialize the posterior with prior
    BOProblem(f, domain, xs, ys, prior, acqf, max_iter, 0, noise, false)
end

function update!(p::BOProblem, x::AbstractVector, y::Float64, i::Int)

    # Add the obserbed data
    append!(p.xs, x)
    append!(p.ys, [y])
    # Could create some issues if we have the same point twice.

    # Update the surrogate
    gp_udated = nothing
    try
        gp_udated = update!(p.gp,vec(p.xs), p.ys, p.noise)
    catch
        println("We reached ill-conditioning, returning NON-UPDATED GP, but updated xs, ys. Killing BO loop.")
        return BOProblem(p.f, p.domain, p.xs, p.ys, p.gp, p.acqf,p.max_iter,i,p.noise, true)
    end
    acqf_updated = update!(p.acqf,p.ys)

    #Is this really necessary? Why not returning p directly with the updated xs,ys and gp?
    # Do we need to create a new object everytime?
    return BOProblem(p.f, p.domain, p.xs, p.ys, gp_udated, acqf_updated,p.max_iter,i,p.noise, p.flag)
end

function stop_criteria(p::BOProblem)
    return p.iter > p.max_iter
end


# Looping routine

function optimize(p::BOProblem)
    """
    This function implements the EGO framework: 
        While some criterion is not met, (1) optimize the acquisition function to obtain 
        the new best candidate, (2) query the target function f, (3) update the GP and the overall optimization state. 

    """
    i = 0
    while !stop_criteria(p) & !p.flag 
        try
            println("Iteration #",i+1,", current min val: ",minimum(p.ys))
        catch
            println("Iteration #",i+1," current min val: NA")
        end
        x_cand = optimize_acquisition!(p.acqf,p.gp,p.domain)
        println("New candidate found: ",x_cand)
        y_cand = p.f(x_cand)
        println("New value found: ",y_cand)
        i +=1
        p = update!(p, x_cand, y_cand, i)
    end
    return p
end