"""
This module contains the structures and functions for the Bayesian Optimization framework.


Parts of the code are inspired by:
- BayesianOptimization (python package) (optimization acq functions)
- GradientGPs.jl (internal package) of MatMat group at EPFL (BOStruct, and update routines)
"""


mutable struct BOStruct{F, M<:AbstractSurrogate, A<:AbstractAcquisition}

    # Core components of Bayesian Optimization problem
    func::F
    acq::A 
    model::M
    domain::AbstractDomain
    
    # Recording history of points and values
    xs::AbstractVector
    ys::AbstractVector
    ys_non_std::AbstractVector

    # Optimization parameters
    max_iter::Int
    iter::Int
    noise::Float64
    flag::Bool
end

"""
    BOStruct(func::Function,
              acq::AbstractAcquisition,
              model::AbstractSurrogate,
              domain::AbstractDomain, 
              x_train::AbstractVector, 
              y_train::AbstractVector, 
              max_iter::Int, 
              noise::Float64)

Initialize the Bayesian Optimization problem.

Arguments:
- `func::Function`: The target function to be optimized.
- `acq::AbstractAcquisition`: The acquisition function guiding the optimization.
- `model::AbstractSurrogate`: The surrogate model (e.g., Gaussian Process).
- `domain::AbstractDomain`: The domain over which to optimize.
- `x_train::AbstractVector`: Initial input training points.
- `y_train::AbstractVector`: Corresponding output training values.
- `max_iter::Int`: Maximum number of iterations for the optimization.
- `noise::Float64`: Noise level in the observations.

returns:
- `BOStruct`: An instance of the BOStruct containing all components for Bayesian Optimization.
"""
function BOStruct(func::Function,
                  acq::AbstractAcquisition,
                  model::AbstractSurrogate,
                  domain::AbstractDomain, 
                  x_train::AbstractVector, 
                  y_train::AbstractVector, 
                  max_iter::Int, 
                  noise::Float64)
    """
    Initialize the Bayesian Optimization problem.
    """
    BOStruct(func, copy(acq), model, domain, copy(x_train), copy(y_train), copy(y_train), max_iter, 0, noise, false)
end


"""
Update the Bayesian Optimization structure with a new data point (x, y).

Arguments:
- `BO::BOStruct`: The current Bayesian Optimization structure.
- `x::AbstractVector`: The new input point where the function has been evaluated.
- `y::AbstractVector`: The corresponding function value at the input point.
- `i::Int`: The current iteration number.

returns:
- `BO::BOStruct`: The updated Bayesian Optimization structure.

Remarks: 
    This function handles potential ill-conditioning issues when updating the GP, 
    by returning the previous state if an error occurs and setting a flag to stop the optimization loop.
"""
function update(BO::BOStruct, x::AbstractVector, y::AbstractVector, i::Int)


    #TODO make the copy only if we fail
    prev_gp = copy(BO.model)
    

    # Adding queried point to conditioning dataset
    push!(BO.xs, x)
    push!(BO.ys, y)

    # Could create some issues if we have the same point twice.
    try
        # test for ill-conditioning
        BO.model = update(BO.model, BO.xs, BO.ys)
    catch
        println("We reached ill-conditioning, returning NON-UPDATED GP. Killing BO loop.")
        # Issue: the gp_update in the try is updating the p.gp.gpx as it tries to create the posterior.
        # if it fails, it keeps the added x and y and overwrites the old structure, which I want to keep if it fails...
        # so now its a bit bruteforce but I try to recreate the previous GP structure. Maybe copying it every time would help.
        BO.model = prev_gp
        BO.xs = BO.xs[1:(length(BO.xs)-1)]
        BO.ys = BO.ys[1:(length(BO.ys)-1)]
        BO.ys_non_std = BO.ys_non_std[1:(length(BO.ys_non_std)-1)]
        
        
        #println("Final # points for posterior: ",length(BO.xs))
        BO.flag = true # we need to stop the BO loop
        return BO
    end

    # update the ACQ function
    acq_updated = update(BO.acq,BO.ys,BO.model)

    BO.acq = acq_updated
    BO.iter = i + 1

    return BO
end

function stop_criteria(p::BOStruct)
    return p.iter > p.max_iter
end


"""
Optimize the hyperparameters of the Gaussian Process model using Maximum Likelihood Estimation (MLE).

Arguments:
- `model::AbstractSurrogate`: Surrogate model.
- `x_train::AbstractVector`: A vector of input training points.
- `y_train::AbstractVector`: A vector of corresponding output training values.
- `old_params::Vector{Float64}`: A vector containing the current log lengthscale
    and log scale parameters.
- `classic_bo::Bool`: Indicates if using classic Bayesian Optimization (true) or gradient-enhanced (false).
- `length_scale_only::Bool`: If true, only optimize the lengthscale, keeping the scale fixed.
- `mean::AbstractGPs.MeanFunction`: The mean function of the GP, defaults to ZeroMean().
- `num_restarts::Int`: Number of random restarts for the optimization. If set to 1, uses the current parameters as the initial guess.


returns:
- `model::AbstractSurrogate`: The updated surrogate model with optimized hyperparameters.

"""
function optimize_hyperparameters(model::AbstractSurrogate,
                                  x_train::AbstractVector,
                                  y_train::AbstractVector,
                                  old_params::Vector{Float64},
                                  classic_bo::Bool;
                                  scale_std::Float64=1.0,
                                  length_scale_only::Bool=false,
                                  mean::AbstractGPs.MeanFunction=ZeroMean(),
                                  num_restarts::Int=1)

    # old_params is always a 2-element vector: [log(lengthscale), log(scale)]
    if length(old_params) != 2
        error("old_params must be a 2-element vector: [log(lengthscale), log(scale)]")
    end

    best_nlml = Inf
    best_result = nothing
    
    # Store original scale for length_scale_only case
    original_scale = exp(old_params[2])

    # Adjust scale bounds by 1/σ² to account for standardization
    # This ensures bounds in original space remain 1e-3 to 1e4 regardless of standardization
    # Define original space bounds
    length_scale_lower, length_scale_upper = 1e-3, 1e3
    scale_lower, scale_upper = 1e-3, 1e6

    # Adjust scale bounds for standardized space
    adjusted_scale_lower = scale_lower/(scale_std^2)
    adjusted_scale_upper = scale_upper/(scale_std^2)

    length_scale_only ? lower_bounds = log.([length_scale_lower]) : 
                        lower_bounds = log.([length_scale_lower, adjusted_scale_lower])

    length_scale_only ? upper_bounds = log.([length_scale_upper]) : 
                        upper_bounds = log.([length_scale_upper, adjusted_scale_upper])

    println("lower bounds (ℓ, scale): $(exp.(lower_bounds))")
    println("upper bounds (ℓ,scale): $(exp.(upper_bounds))")
    
    x_train_prepped = prep_input(model, x_train)
    y_train_prepped = nothing
    if classic_bo
        y_train_prepped = reduce(vcat,y_train)
    else
        y_train_prepped = vec(permutedims(reduce(hcat, y_train)))
    end

    obj = nothing
    if length_scale_only
        # Only optimize lengthscale, keep scale fixed at original log value (second parameter)
        obj = p -> nlml_ls(model, p[1], old_params[2], x_train_prepped, y_train_prepped, mean=mean)
    else
        # Optimize both lengthscale and scale (vector p)
        obj = p -> nlml(model,p, x_train_prepped, y_train_prepped, mean=mean)
    end

    opts = Optim.Options(g_tol=1e-6, f_abstol=2.2e-9)

    random_inits = [rand.(Uniform.(lower_bounds, upper_bounds)) for _ in 1:(num_restarts - 1)]
    if length_scale_only
        init_guesses = [[old_params[1]], random_inits...]
    else
        init_guesses = [collect(old_params), random_inits...]
    end

    inner_optimizer = LBFGS(;linesearch = Optim.LineSearches.HagerZhang(linesearchmax=20))

    for i in 1:num_restarts
        try
            result = Optim.optimize(obj,lower_bounds, upper_bounds, init_guesses[i], Fminbox(inner_optimizer),opts, autodiff = :forward)

            println("Optimization result: ", result)
            if Optim.converged(result)
                current_nlml = Optim.minimum(result)

                if current_nlml < best_nlml
                    best_nlml = current_nlml
                    best_result = Optim.minimizer(result)
                                
                end
            end
        catch e 
            @warn "Optimization failed at restart $i with error: $e, and parameters $(exp.(init_guesses[i]))"
            continue
        end
    end

    if best_result === nothing
        println("All restarts failed to converge.")
        return model
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
        scale = original_scale 
    else
        ℓ, scale = exp.(best_result)
    end
    
    kernel_constructor = get_kernel_constructor(model)

    k_opt = scale * (kernel_constructor ∘ ScaleTransform(1/ℓ))

    if classic_bo
        return StandardGP(k_opt, model.noise_var, mean=mean)
    else
        return GradientGP(gradKernel(k_opt),model.p, model.noise_var, mean=mean)
    end
end



"""
This function implements the EGO framework:
    While some criterion is not met,
        (1) optimize the acquisition function to obtain the new best candidate,
        (2) query the target function f,            
        (3) update the GP and the overall optimization state.
    returns best found solution.


Arguments:
- `BO::BOStruct`: The Bayesian Optimization structure.
- `standardize::String`: Specifies how to standardize the outputs.
    - If "mean_scale", standardize by removing mean and scaling by std.
    - If "scale_only", only scale the outputs without centering (in case we set a non-zero mean function with empirical mean).
    - If "mean_only", only remove the mean without scaling.
    - If nothing, do not standardize the outputs.
- `hyper_params::String`: Specifies how to handle hyperparameters.
    - If "all", re-optimize hyperparameters every 10 iterations.
    - If "length_scale_only", only optimize the lengthscale.
    - If nothing, do not re-optimize hyperparameters.
- `num_restarts_HP::Int`: Number of random restarts for hyperparameter optimization.

returns:
- `BO::BOStruct`: The updated Bayesian Optimization problem after optimization.
- `acqf_list::Vector`: List of acquisition function values at each iteration.
- `standard_params::Tuple`: Tuple containing the mean and standard deviation used for standardization
"""
function optimize(BO::BOStruct;
                  standardize::Union{String, Nothing}="mean_scale",
                  hyper_params::Union{String, Nothing}="all",
                  num_restarts_HP::Int=5)

    @assert hyper_params in ["all", "length_scale_only", nothing] "hyper_params must be one of: 'all', 'length_scale_only', or nothing."


    @assert standardize in ["mean_scale", "scale_only", "mean_only", nothing] "standardize must be one of: 'mean_scale', 'scale_only', 'mean_only', or nothing."

    acq_list = []
    n_train = length(BO.xs)
    
    classic_bo = (length(BO.ys[1])==1)
    
    μ=zeros(length(BO.ys[1]))
    σ=ones(length(BO.ys[1])) # default values if we do not standardize
    if isnothing(standardize)
        BO.model = update(BO.model, BO.xs, BO.ys) # because we might not to that before
    else 
        BO, (μ, σ) = standardize_problem(BO, choice=standardize)
    end

    original_mean = BO.model.gp.mean

    println(original_mean)
    i = 0
    while !stop_criteria(BO) & !BO.flag

        if !isnothing(hyper_params)&&(i%10==0) 
            println("Re-optimizing GP hyperparameters at iteration $i...")
            println("Former parameters: ℓ=$(get_lengthscale(BO.model)), variance =$(get_scale(BO.model))")
            old_params = log.([get_lengthscale(BO.model)[1],get_scale(BO.model)[1]])
            println("Hyperparameter time taken:")
            out = nothing
            if hyper_params == "length_scale_only"
                @time out = optimize_hyperparameters(BO.model, BO.xs, BO.ys,old_params,classic_bo, 
                                                    length_scale_only=true,mean=original_mean, scale_std=σ[1], num_restarts=num_restarts_HP)
            elseif hyper_params == "all"
                @time out = optimize_hyperparameters(BO.model, BO.xs, BO.ys,old_params,classic_bo,
                                                length_scale_only = false, mean=original_mean, scale_std=σ[1], num_restarts=num_restarts_HP)
            else
                out = nothing
            end

            println("MLE new parameters: ℓ=$(get_lengthscale(out)), variance =$(get_scale(out))")
            if !isnothing(out)
                BO.model = out
                BO.model = update(BO.model, BO.xs, BO.ys)
            end

            println("New parameters: ℓ=$(get_lengthscale(BO.model)), variance =$(get_scale(BO.model))")
        end

        
        if classic_bo
            println("Iteration #",i+1,", current min val: ",minimum(BO.ys_non_std))
        else
            println("Iteration #",i+1,", current min val: ",minimum(hcat(BO.ys_non_std...)[1,:]))
        end

        println("Time acq update:")
        @time x_cand = optimize_acquisition(BO.acq,BO.model,BO.domain)
        

        println("New point acquired: $(x_cand) with acq func $(BO.acq(BO.model, x_cand))")
        push!(acq_list,BO.acq(BO.model, x_cand))

        y_cand = BO.func(x_cand) 
        y_cand = y_cand .+ sqrt(BO.noise)/(σ[1])*randn(length(y_cand))
        # y_cand here is NOT standardized
        push!(BO.ys_non_std, y_cand)
        println("New value probed: ",y_cand)
        i +=1
        println("Time update GP")
       
        
        # here we have μ and σ according to the standardization choice
        # The vectors are given as (pseudo-code):
        # if scale_only, μ = 0, σ ≠ 1
        # if mean_scale, μ ≠ 0 and σ ≠ 1
        
        y_cand = (y_cand .- μ)./σ[1]
        @time BO = update(BO, x_cand, y_cand, i)
    end

    return BO, acq_list, (μ,σ)
end