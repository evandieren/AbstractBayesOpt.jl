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
    kernel_constructor::KernelFunctions.Kernel
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


function print_info(BO::BOStruct)
    println("== Printing information about the BOStruct ==")
    println("Target function: ",BO.func)
    println("Domain: ",BO.domain)
    println("xs: ",BO.xs)
    println("ys: ",BO.ys)
    println("Surrogate: ",BO.model)
    println("ACQ: ",BO.acq)
    println("max_iter: ",BO.max_iter)
    println("noise: ",BO.noise)
end

function BOStruct(func::Function,
                  acq::AbstractAcquisition,
                  model::AbstractSurrogate,
                  kernel_constructor::KernelFunctions.Kernel,
                  domain::AbstractDomain, 
                  x_train::AbstractVector, 
                  y_train::AbstractVector, 
                  max_iter::Int, 
                  noise::Float64)
    """
    Initialize the Bayesian Optimization problem.
    """
    
    #TODO get rid of kernel_constructor if we can get it from model
    
    BOStruct(func, copy(acq), model, kernel_constructor, domain, copy(x_train), copy(y_train), copy(y_train), max_iter, 0, noise, false)
end



"""
    update!(BO::BOStruct, x::AbstractVector, y::AbstractVector, i::Int)

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
function update!(BO::BOStruct, x::AbstractVector, y::AbstractVector, i::Int)


    #TODO make the copy only if we fail
    prev_gp = copy(BO.model)
    

    # Adding queried point to conditioning dataset
    push!(BO.xs, x)
    push!(BO.ys, y)

    # Could create some issues if we have the same point twice.
    try
        # test for ill-conditioning
        BO.model = update!(BO.model, BO.xs, BO.ys)
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
    acq_updated = update!(BO.acq,BO.ys,BO.model)

    BO.acq = acq_updated
    BO.iter = i + 1

    return BO
end

function stop_criteria(p::BOStruct)
    return p.iter > p.max_iter
end


"""
    optimize_hyperparameters(model::AbstractSurrogate,
                                  x_train::AbstractVector,
                                  y_train::AbstractVector,
                                  kernel_constructor::Kernel,
                                  old_params::Vector{Float64},
                                  classic_bo::Bool;
                                  length_scale_only::Bool=false,
                                  mean::AbstractGPs.MeanFunction=ZeroMean(),
                                  num_restarts::Int=1)

Optimize the hyperparameters of the Gaussian Process model using Maximum Likelihood Estimation (MLE).

Arguments:
- `model::AbstractSurrogate`: Surrogate model.
- `x_train::AbstractVector`: A vector of input training points.
- `y_train::AbstractVector`: A vector of corresponding output training values.
- `kernel_constructor::Kernel`: The kernel function used in the GP.
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
                                  kernel_constructor::Kernel,
                                  old_params::Vector{Float64},
                                  classic_bo::Bool;
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

    length_scale_only ? lower_bounds = log.([1e-3]) : lower_bounds = log.([1e-3, 1e-6])
    length_scale_only ? upper_bounds = log.([1e1]) : upper_bounds = log.([1e1, 1e2])

    
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
        obj = p -> nlml_ls(model, p[1], old_params[2], kernel_constructor, x_train_prepped, y_train_prepped, mean=mean)
    else
        # Optimize both lengthscale and scale (vector p)
        obj = p -> nlml(model,p, kernel_constructor, x_train_prepped, y_train_prepped, mean=mean)
    end

    #grad_obj! = nothing
    #if !classic_bo
    #    grad_obj! = (G, p) -> ReverseDiff.gradient!(G, obj, p)
    #end

    opts = Optim.Options(g_tol=1e-5,f_abstol=1e-6,x_abstol=1e-4,outer_iterations=100)

    random_inits = [rand.(Uniform.(lower_bounds, upper_bounds)) for _ in 1:(num_restarts - 1)]
    if length_scale_only
        init_guesses = [[old_params[1]], random_inits...]
    else
        init_guesses = [collect(old_params), random_inits...]
    end

    inner_optimizer = LBFGS(;linesearch = Optim.LineSearches.HagerZhang(linesearchmax=20))

    for i in 1:num_restarts
        try
            # if classic_bo
            #    result = Optim.optimize(obj, lower_bounds, upper_bounds, init_guesses[i], Fminbox(inner_optimizer),opts,autodiff = :forward)
            #else
            #    result = Optim.optimize(obj, grad_obj!,
            #                            lower_bounds, upper_bounds, init_guesses[i], 
            #                           Fminbox(inner_optimizer), opts)
            #end
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
            @warn "Optimization failed at restart $i with error: $e"
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

    k_opt = scale * (kernel_constructor ∘ ScaleTransform(1/ℓ))

    if classic_bo
        return StandardGP(k_opt, model.noise_var, mean=mean)
    else
        return GradientGP(gradKernel(k_opt),model.p, model.noise_var, mean=mean)
    end
end

"""
    rescale_output(ys::AbstractVector, params::Tuple)

Rescale the standardized output values back to the original scale.

Arguments:
- `ys::AbstractVector`: A vector of standardized function values.
- `params::Tuple`: A tuple containing the mean and standard deviation used for standardization.

returns:
- `ys_rescaled`: A vector of rescaled function values.
"""
function rescale_output(ys::AbstractVector, params::Tuple)
    μ, σ = params

    if isnothing(μ) || isnothing(σ)
        return [y for y in ys]
    else
        return [(y .* σ) .+ μ for y in ys]
    end
end

"""
    standardize_problem(p::BOStruct)

Standardize the output values of the BOStruct and update the GP and acquisition function accordingly.


Arguments:
- `p::BOStruct`: The Bayesian Optimization problem to standardize.

returns:
- `p::BOStruct`: The updated Bayesian Optimization problem with standardized outputs.
- `params::Tuple`: A tuple containing the mean and standard deviation used for standardization.
"""
function standardize_problem(BO::BOStruct)    
    μ=nothing; σ=nothing
    BO.ys, μ, σ = standardize_y(BO.model,BO.ys_non_std)
    println("Standardization applied: μ=$μ, σ=$σ")
    BO.model = update!(BO.model, BO.xs, BO.ys)
    BO.acq = update!(BO.acq,BO.ys, BO.model)
    return BO, (μ, σ)
end

"""
    optimize(BO::BOStruct;fn=nothing, standardize=true, hyper_params="all")

This function implements the EGO framework:
    While some criterion is not met,
        (1) optimize the acquisition function to obtain the new best candidate,
        (2) query the target function f,            
        (3) update the GP and the overall optimization state.
    returns best found solution.


Arguments:
- `BO::BOStruct`: The Bayesian Optimization structure.
- `fn::Union{Nothing, String}`: If provided, the function will save plots of the optimization state at each iteration to files named with this prefix.
- `standardize::Bool`: If true, standardize the output values before starting the optimization.
- `hyper_params::String`: Specifies how to handle hyperparameters.
    - If "all", re-optimize hyperparameters every 10 iterations.
    - If 'length_scale_only', only optimize the lengthscale.
    - If nothing, do not re-optimize hyperparameters.

returns:
- `BO::BOStruct`: The updated Bayesian Optimization problem after optimization.
- `acqf_list::Vector`: List of acquisition function values at each iteration.
- `standard_params::Tuple`: Tuple containing the mean and standard deviation used for standardization
"""
function optimize(BO::BOStruct;
                  fn=nothing,
                  standardize=true,
                  hyper_params="all")

    @assert hyper_params in ["all", "length_scale_only", "none", nothing] "hyper_params must be one of: 'all', 'length_scale_only', 'none', or nothing."

    acq_list = []
    n_train = length(BO.xs)
    
    classic_bo = (length(BO.ys[1])==1)
    
    μ=0.0; σ=1.0
    if standardize
        BO, (μ, σ) = standardize_problem(BO)
    else 
        BO.model = update!(BO.model, BO.xs, BO.ys) # because we might not to that before
    end

    original_mean = BO.model.gp.mean
    i = 0
    while !stop_criteria(BO) & !BO.flag

        if !isnothing(hyper_params)&&(i%10==0) 
            println("Re-optimizing GP hyperparameters at iteration $i...")
            println("Former parameters: ℓ=$(get_lengthscale(BO.model)), variance =$(get_scale(BO.model))")
            old_params = log.([get_lengthscale(BO.model)[1],get_scale(BO.model)[1]])
            println("Hyperparameter time taken:")
            out = nothing
            if hyper_params == "length_scale_only"
                @time out = optimize_hyperparameters(BO.model, BO.xs, BO.ys,BO.kernel_constructor,old_params,classic_bo, 
                                                    length_scale_only=true,mean=original_mean)
            elseif hyper_params == "all"
                @time out = optimize_hyperparameters(BO.model, BO.xs, BO.ys,BO.kernel_constructor,old_params,classic_bo,
                                                length_scale_only = false, mean=original_mean)
            else
                out = nothing
            end

            println("MLE new parameters: ℓ=$(get_lengthscale(out)), variance =$(get_scale(out))")
            if !isnothing(out)
                BO.model = out
                BO.model = update!(BO.model, BO.xs, BO.ys)
            end

            println("New parameters: ℓ=$(get_lengthscale(BO.model)), variance =$(get_scale(BO.model))")
        end

        
        if classic_bo
            println("Iteration #",i+1,", current min val: ",minimum(BO.ys_non_std))
        else
            println("Iteration #",i+1,", current min val: ",minimum(hcat(BO.ys_non_std...)[1,:]))
        end

        println("Time acq update:")
        @time x_cand = optimize_acquisition!(BO.acq,BO.model,BO.domain)
        
        if !isnothing(fn)
            plot_state(BO,n_train,x_cand,"./$(fn)_iter_$(i).png")
        end

        println("New point acquired: $(x_cand) with acq func $(BO.acq(BO.model, x_cand))")
        push!(acq_list,BO.acq(BO.model, x_cand))

        y_cand = BO.func(x_cand) 
        y_cand = y_cand .+ sqrt(BO.noise)*randn(length(y_cand))
        # y_cand here is NOT standardized
        push!(BO.ys_non_std, y_cand)
        println("New value probed: ",y_cand)
        i +=1
        println("Time update GP")
        
        if standardize
            y_cand_std = (y_cand .- μ) ./ σ
            
            @time BO = update!(BO, x_cand, y_cand_std, i)
        else
            @time BO = update!(BO, x_cand, y_cand, i)
        end
    end

    return BO, acq_list, (μ,σ)
end