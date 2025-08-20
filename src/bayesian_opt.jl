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
    ys_non_std::AbstractVector
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

function BOProblem(f, 
                   domain::AbstractDomain, 
                   prior::AbstractSurrogate, 
                   kernel_constructor, 
                   x_train::AbstractVector, 
                   y_train::AbstractVector, 
                   acqf::AbstractAcquisition, 
                   max_iter::Int, 
                   noise::Float64)
    """
    Initialize the Bayesian Optimization problem.
    """
    
    xs = x_train
    ys = y_train
    # Initialize the posterior with prior
    BOProblem(f, domain, xs, ys, copy(ys), prior,kernel_constructor, acqf, max_iter, 0, noise, false)
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
        p.ys_non_std = p.ys_non_std[1:(length(p.ys_non_std)-1)]
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

function optimize_hyperparameters(gp_model, X_train, y_train, kernel_constructor,old_params,classic_bo;length_scale_only=false, mean=ZeroMean(),num_restarts=1)

    # old_params is always a 2-element vector: [log(lengthscale), log(scale)]
    if length(old_params) != 2
        error("old_params must be a 2-element vector: [log(lengthscale), log(scale)]")
    end

    best_nlml = Inf
    best_result = nothing
    
    # Store original scale for length_scale_only case
    original_scale = exp(old_params[2])

    length_scale_only ? lower_bounds = log.([1e-3]) : lower_bounds = log.([1e-3, 1e-6])
    length_scale_only ? upper_bounds = log.([1e1]) : upper_bounds = log.([1e1, 1e2])

    
    x_train_prepped = prep_input(gp_model, X_train)
    y_train_prepped = nothing
    if classic_bo
        y_train_prepped = reduce(vcat,y_train)
    else
        y_train_prepped = vec(permutedims(reduce(hcat, y_train)))
    end

    obj = nothing
    if length_scale_only
        # Only optimize lengthscale (scalar p), keep scale fixed at original log value (second parameter)
        obj = p -> nlml(gp_model, [p; old_params[2]], kernel_constructor, x_train_prepped, y_train_prepped, mean=mean)
    else
        # Optimize both lengthscale and scale (vector p)
        obj = p -> nlml(gp_model, p, kernel_constructor, x_train_prepped, y_train_prepped, mean=mean)
    end

    if !classic_bo
        grad_cache = build_grad_cache(gp_model, kernel_constructor, x_train_prepped, y_train_prepped)
        obj = p -> fast_nlml!(grad_cache, p)
    end

    grad_obj! = nothing
    if !classic_bo
        grad_obj! = (G, p) -> ReverseDiff.gradient!(G, obj, p)
    end
    opts = Optim.Options(g_tol=1e-5,f_abstol=1e-6,x_abstol=1e-4,outer_iterations=100)

    random_inits = [rand.(Uniform.(lower_bounds, upper_bounds)) for _ in 1:(num_restarts - 1)]
    # Fix initial guess generation to be consistent with parameter dimensions
    if length_scale_only
        init_guesses = [[old_params[1]], random_inits...]
    else
        init_guesses = [collect(old_params), random_inits...]
    end

    inner_optimizer = LBFGS(;linesearch = Optim.LineSearches.HagerZhang(linesearchmax=20))

    for i in 1:num_restarts
        try
            if classic_bo
                result = Optim.optimize(obj, lower_bounds, upper_bounds, init_guesses[i], Fminbox(inner_optimizer),opts,autodiff = :forward)
            else
                result = Optim.optimize(obj, grad_obj!,
                                        lower_bounds, upper_bounds, init_guesses[i], 
                                        Fminbox(inner_optimizer), opts)
                #result = Optim.optimize(obj, lower_bounds, upper_bounds, init_guesses[i], Fminbox(inner_optimizer),opts) # work-around for now.
            end
            println("Restart $i starting from $(init_guesses[i]): Optimized parameters: ", result.minimizer, " with minimum value: ", result.minimum)
            println("Optimization result: ", result)
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
        return gp_model
    else
        println("Best LML after $(num_restarts) restarts: ", -best_nlml)
        if length_scale_only
            println("Optimized lengthscale (log): $(best_result[1]), kept scale (log): $(old_params[2])")
        else
            println("Optimized parameters (log): lengthscale=$(best_result[1]), scale=$(best_result[2])")
        end
    end

    ℓ = nothing; scale = nothing
    if length_scale_only
        ℓ = exp(best_result[1])
        scale = original_scale  # Use original scale instead of hardcoded 1.0
    else
        ℓ, scale = exp.(best_result)
    end

    k_opt = scale * (kernel_constructor ∘ ScaleTransform(1/ℓ))

    if classic_bo
        return StandardGP(k_opt, gp_model.noise_var, mean=mean)
    else
        return GradientGP(gradKernel(k_opt),gp_model.p, gp_model.noise_var, mean=mean)
    end
end

function rescale_output(ys::AbstractVector,params::Tuple)
    μ, σ = params

    if isnothing(μ) || isnothing(σ)
        return [y for y in ys]
    else
        return [(y .* σ) .+ μ for y in ys]
    end
end

function standardize_problem(p::BOProblem)    
    μ=nothing; σ=nothing
    p.ys, μ, σ = standardize_y(p.gp,p.ys_non_std)
    println("Standardization applied: μ=$μ, σ=$σ")
    p.gp = update!(p.gp, p.xs, p.ys)
    p.acqf = update!(p.acqf,p.ys, p.gp)
    return p, (μ, σ)
end

"""
    optimize(p::BOProblem;fn=nothing, standardize=true, hyper_params=true)
This function implements the EGO framework:
    While some criterion is not met,
    (1) optimize the acquisition function to obtain the new best candidate,
    (2) query the target function f,            
    (3) update the GP and the overall optimization state.

Arguments:
- `p::BOProblem`: The Bayesian Optimization problem to solve.
- `fn::String`: Optional filename for saving plots.
- `standardize::Bool`: Whether to standardize the problem.
- `hyper_params::String`: Specifies how to handle hyperparameters.
    - If "all", re-optimize hyperparameters every 20 iterations.
    - If 'length_scale_only', only optimize the lengthscale.
    - If nothing, do not re-optimize hyperparameters.

returns:
- `p::BOProblem`: The updated Bayesian Optimization problem after optimization.
- `acqf_list::Vector`: List of acquisition function values at each iteration.
- `standard_params::Tuple`: Tuple containing the mean and standard deviation used for standardization
"""
function optimize(p::BOProblem;
                  fn=nothing,
                  standardize=true,
                  hyper_params="all")

    @assert hyper_params in ["all", "length_scale_only", "none", nothing] "hyper_params must be one of: 'all', 'length_scale_only', 'none', or nothing."

    acqf_list = []
    n_train = length(p.xs)
    
    classic_bo = (length(p.ys[1])==1)
    
    μ=0.0; σ=1.0
    if standardize
        p, (μ, σ) = standardize_problem(p)
    else 
        p.gp = update!(p.gp, p.xs, p.ys) # because we might not to that before
    end

    original_mean = p.gp.gp.mean
    @assert isa(original_mean,ZeroMean) 
    i = 0
    while !stop_criteria(p) & !p.flag

        if !isnothing(hyper_params)&&(i%10==0) 
            println("Re-optimizing GP hyperparameters at iteration $i...")
            println("Former parameters: ℓ=$(get_lengthscale(p.gp)), variance =$(get_scale(p.gp))")
            old_params = log.([get_lengthscale(p.gp)[1],get_scale(p.gp)[1]])
            println("Hyperparameter time taken:")
            out = nothing
            if hyper_params == "length_scale_only"
                @time out = optimize_hyperparameters(p.gp, p.xs, p.ys,p.kernel_constructor,old_params,classic_bo, 
                                                    length_scale_only=true,mean=original_mean)
            elseif hyper_params == "all"
                @time out = optimize_hyperparameters(p.gp, p.xs, p.ys,p.kernel_constructor,old_params,classic_bo,
                                                length_scale_only = false, mean=original_mean)
            else
                out = nothing
            end

            println("MLE new parameters: ℓ=$(get_lengthscale(out)), variance =$(get_scale(out))")
            if !isnothing(out)
                p.gp = out
                p.gp = update!(p.gp, p.xs, p.ys)
            end

            println("New parameters: ℓ=$(get_lengthscale(p.gp)), variance =$(get_scale(p.gp))")
        end

        
        if classic_bo
            println("Iteration #",i+1,", current min val: ",minimum(p.ys_non_std))
        else
            println("Iteration #",i+1,", current min val: ",minimum(hcat(p.ys_non_std...)[1,:]))
        end

        println("Time acq update:")
        @time x_cand = optimize_acquisition!(p.acqf,p.gp,p.domain)
        
        if !isnothing(fn)
            plot_state(p,n_train,x_cand,"./examples/plots/$(fn)_iter_$(i).png")
        end

        println("New point acquired: $(x_cand) with acq func $(p.acqf(p.gp, x_cand))")
        push!(acqf_list,p.acqf(p.gp, x_cand))

        y_cand = p.f(x_cand) 
        y_cand = y_cand .+ sqrt(p.noise)*randn(length(y_cand))
        # y_cand here is NOT standardized
        push!(p.ys_non_std, y_cand)
        println("New value probed: ",y_cand)
        i +=1
        println("Time update GP")
        
        if standardize
            #push!(p.xs, x_cand)
            # @time p, (μ, σ) = standardize_problem(p)
            # Standardize new observation using existing parameters
            y_cand_std = (y_cand .- μ) ./ σ
            
            @time p = update!(p, x_cand, y_cand_std, i)
        else
            @time p = update!(p, x_cand, y_cand, i)
        end
    end

    return p, acqf_list, (μ,σ)
end