"""
This module contains the structures and functions for the Bayesian Optimization framework.

BOProblem (struct) : structure containing the required information to run the Bayesian Optimization algorithm.
BOProblem (function) : Initiate a BOProblem structure
update (for BOProblem) : updates the BOProblem once you have new values x,y
"""

mutable struct BOProblem{T<:AbstractSurrogate,A<:AbstractAcquisition}
    f
    domain::AbstractDomain
    xs::AbstractVector
    ys::AbstractVector
    gp::T
    kernel_constructor
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

function BOProblem(f, domain::AbstractDomain, prior::AbstractSurrogate, kernel_constructor, x_train::AbstractVector, y_train::AbstractVector, acqf::AbstractAcquisition, max_iter::Int, noise::Float64)
    """
    Initialize the Bayesian Optimization problem.
    """
    
    xs = x_train
    ys = y_train
    # Initialize the posterior with prior
    BOProblem(f, domain, xs, ys, prior,kernel_constructor, acqf, max_iter, 0, noise, false)
end

function update!(p::BOProblem, x::AbstractVector, y::AbstractVector, i::Int)
    prev_gp = copy(p.gp)
    
    # Update the surrogate
    # Add the obserbed data
    push!(p.xs, x)
    push!(p.ys, y)

    # Could create some issues if we have the same point twice.
    try
        # test for ill-conditioning
        p.gp = update!(p.gp,p.xs, p.ys)
    catch
        println("We reached ill-conditioning, returning NON-UPDATED GP. Killing BO loop.")
        # Issue: the gp_update in the try is updating the p.gp.gpx as it tries to create the posterior.
        # if it fails, it keeps the added x and y and overwrites the old structure, which I want to keep if it fails...
        # so now its a bit bruteforce but I try to recreate the previous GP structure. Maybe copying it every time would help.
        p.gp = prev_gp
        println(length(prev_gp.gpx.data.x))
        p.xs = p.xs[1:(length(p.xs)-1)]
        p.ys = p.ys[1:(length(p.ys)-1)]
        println("Final # points for posterior: ",length(p.xs))
        p.flag = true
        return p
    end

    acqf_updated = update!(p.acqf,p.ys,p.gp)

    p.acqf = acqf_updated
    p.iter = i + 1
    return p
end

function stop_criteria(p::BOProblem)
    return p.iter > p.max_iter
end

function optimize_hyperparameters(gp_model, X_train, y_train, kernel_constructor,old_params,classic_bo; mean=ZeroMean(),num_restarts=5)

    best_nlml = Inf
    best_result = nothing

    lower_bounds = log.([0.1, 0.1])
    upper_bounds = log.([1e2, 1e2])

    x_train_prepped = prep_input(gp_model, X_train)
    y_train_prepped = nothing
    if classic_bo
        y_train_prepped = reduce(vcat,y_train)
    else
        y_train_prepped = vec(permutedims(reduce(hcat, y_train)))
    end

    obj = p -> nlml(gp_model, p, kernel_constructor, x_train_prepped, y_train_prepped, gp_model.noise_var, mean=mean)

    opts = Optim.Options(iterations = 100, g_tol=1e-5,f_abstol=2.2e-9)

    random_inits = [rand.(Uniform.(lower_bounds, upper_bounds)) for _ in 1:(num_restarts - 1)]
    init_guesses = [collect(old_params), random_inits...]
    for i in 1:num_restarts
        try
            result = Optim.optimize(obj, lower_bounds, upper_bounds, init_guesses[i], Fminbox(LBFGS()),opts)
                
            if Optim.converged(result)
                current_nlml = Optim.minimum(result)

                if current_nlml < best_nlml
                    best_nlml = current_nlml
                    best_result = Optim.minimizer(result)
                                
                end
            end
        catch e 
            @warn "Optimization failed at restart $i with error: $e"
            continue
        end
    end

    if best_result === nothing
        println("All restarts failed to converge.")
    else
        println("Best NLML after $(num_restarts) restarts: ", best_nlml)
    end

    ℓ, scale = exp.(best_result)
    k_opt = scale * (kernel_constructor ∘ ScaleTransform(ℓ))

    if classic_bo
        return StandardGP(k_opt, gp_model.noise_var, mean=mean)
    else
        return GradientGP(gradKernel(k_opt),gp_model.p, gp_model.noise_var, mean=mean)
    end
end

function rescale_output(y_scaled::AbstractVector,params::Tuple)
    μ, σ = params
    return [(y .* σ) .+ μ for y in y_scaled]
end

# Looping routine
function optimize(p::BOProblem;fn=nothing,standardize=true)
    """
    This function implements the EGO framework: 
        While some criterion is not met, 
        (1) optimize the acquisition function to obtain the new best candidate, 
        (2) query the target function f, 
        (3) update the GP and the overall optimization state. 
    """
    acqf_list = []
    n_train = length(p.xs)
    
    classic_bo = (length(p.ys[1])==1)
    
    μ=nothing
    σ=nothing
    if standardize
        p.ys, μ, σ = standardize_y(p.gp,p.ys)
        println("Standardization applied: μ=$μ, σ=$σ")
        p.gp = update!(p.gp, p.xs, p.ys)
        p.acqf = update!(p.acqf,p.ys, p.gp)
        println("New value of p.acqf best_y after standard $(p.acqf.best_y)")
    end
    
    f̃(x) = (p.f(x).-μ)./σ

    original_mean = p.gp.gp.mean
    println(original_mean)
    i = 0
    while !stop_criteria(p) & !p.flag

        if i % 20 == 0  # Optimize GP hyperparameters
            println("Re-optimizing GP hyperparameters at iteration $i...")
            println("Former parameters: ℓ=$(get_lengthscale(p.gp)), variance =$(get_scale(p.gp))")
            old_params = log.([get_lengthscale(p.gp)[1],get_scale(p.gp)[1]])
            println("Hyperparameter time taken:")
            @time out = optimize_hyperparameters(p.gp, p.xs, p.ys,p.kernel_constructor,old_params,classic_bo,mean=original_mean)
            if !isnothing(out)
                p.gp = out
                p.gp = update!(p.gp, p.xs, p.ys)
            end
            println("New parameters: ℓ=$(get_lengthscale(p.gp)), variance =$(get_scale(p.gp))")
        end

        
        if classic_bo
            println("Iteration #",i+1,", current min val: ",minimum(p.ys))
        else
            println("Iteration #",i+1,", current min val: ",minimum(hcat(p.ys...)[1,:]))
        end

        println("Time acq update:")
        @time x_cand = optimize_acquisition!(p.acqf,p.gp,p.domain)
        
        if !isnothing(fn)
            plot_state(p,n_train,x_cand,"./examples/plots/$(fn)_iter_$(i).png")
        end

        println("New point acquired: $(x_cand) with acq func $(p.acqf(p.gp, x_cand))")
        push!(acqf_list,p.acqf(p.gp, x_cand))
        y_cand = f̃(x_cand)
        y_cand = y_cand .+ sqrt(p.noise)*randn(length(y_cand))
        

        println("New value probed: ",y_cand)
        i +=1
        println("Time update GP")
        @time p = update!(p, x_cand, y_cand, i)
    end

    return p, acqf_list, (μ,σ)
end