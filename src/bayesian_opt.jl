"""
This module contains the structures and functions for the Bayesian Optimization framework.

BOProblem (struct) : structure containing the required information to run the Bayesian Optimization algorithm.
BOProblem (function) : Initiate a BOProblem structure
update (for BOProblem) : updates the BOProblem once you have new values x,y
"""

mutable struct BOProblem{T<:AbstractSurrogate,F<:Function,A<:AbstractAcquisition}
    f::F
    domain::AbstractDomain
    xs::AbstractVector
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

function BOProblem(f::Function, domain::AbstractDomain, prior::AbstractSurrogate,x_train::AbstractVector, y_train::AbstractVector, acqf::AbstractAcquisition, max_iter::Int, noise::Float64)
    """
    Initialize the Bayesian Optimization problem.
    """
    
    xs = x_train
    ys = y_train
    # Initialize the posterior with prior
    BOProblem(f, domain, xs, ys, prior, acqf, max_iter, 0, noise, false)
end

function update!(p::BOProblem, x::AbstractVector, y::Float64, i::Int)

    # Add the obserbed data
    push!(p.xs, x)
    push!(p.ys, y)
    # Could create some issues if we have the same point twice.


    # Update the surrogate
    gp_updated = nothing
    old_xs = nothing
    old_ys = nothing
    try
        # test for ill-conditioning
        gp_updated = update!(p.gp,p.xs, p.ys, p.noise)
    catch
        println("We reached ill-conditioning, returning NON-UPDATED GP. Killing BO loop.")


        # Issue: the gp_update in the try is updating the p.gp.gpx as it tries to create the posterior.
        # if it fails, it keeps the added x and y and overwrites the old structure, which I want to keep if it fails...
        # so now its a bit bruteforce but I try to recreate the previous GP structure. Maybe copying it every type would help.
        
        println("Final # points for posterior: ",length(old_gp.gpx.data.x))
        p.xs = p.xs[1:(length(p.xs)-1)]
        p.ys = p.ys[1:(length(p.ys)-1)]
        p.gp = update!(p.gp, old_xs, old_ys, p.noise)
        p.flag = true
        return p
    end

    acqf_updated = update!(p.acqf,p.ys)

    #Is this really necessary? Why not returning p directly with the updated xs,ys and gp?
    # Do we need to create a new object everytime?
    # return BOProblem(p.f, p.domain, p.xs, p.ys, gp_updated, acqf_updated,p.max_iter,i,p.noise, p.flag)

    p.gp = gp_updated
    p.acqf = acqf_updated
    p.iter = i + 1
    return p
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
        println("New point acquired: ",x_cand)
        y_cand = p.f(x_cand) + sqrt(p.noise)*randn()
        println("New value probed: ",y_cand)
        i +=1
        p = update!(p, x_cand, y_cand, i)
    end
    return p
end