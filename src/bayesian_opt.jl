"""
This module contains the structures and functions for the Bayesian Optimization framework.


Parts of the code are inspired by:
- BayesianOptimization (python package) (optimization acq functions)
- GradientGPs.jl (internal package) of MatMat group at EPFL (BOStruct, and update routines)
"""

mutable struct BOStruct{
    F, M <: AbstractSurrogate, A <: AbstractAcquisition, D <: AbstractDomain, X, Y, T}

    # Core components of Bayesian Optimization problem
    func::F
    acq::A
    model::M
    domain::D

    # Recording history of points and values
    xs::Vector{X}
    ys::Vector{Y}
    ys_non_std::Vector{Y}

    # Optimization parameters
    max_iter::Int
    iter::Int
    noise::T #TODO: should be Y I think
    flag::Bool
end

"""
Initialize the Bayesian Optimization problem.

Arguments:
- `func`: The target function to be optimized.
- `acq`: The acquisition function guiding the optimization.
- `model`: The surrogate model (e.g., Gaussian Process).
- `domain`: The domain over which to optimize.
- `x_train`: Initial input training points.
- `y_train`: Corresponding output training values.
- `max_iter`: Maximum number of iterations for the optimization.
- `noise`: Noise level in the observations.

returns:
- `BOStruct`: An instance of the BOStruct containing all components for Bayesian Optimization.
"""
function BOStruct(
        func,
        acq,
        model,
        domain,
        x_train,
        y_train,
        max_iter,
        noise
)
    return BOStruct(
        func,
        copy(acq),
        copy(model),
        domain,
        copy(x_train),
        copy(y_train),
        copy(y_train),
        max_iter,
        0,
        noise,
        false
    )
end

"""
Update the Bayesian Optimization structure with a new data point (x, y).

Arguments:
- `BO::BOStruct`: The current Bayesian Optimization structure.
- `x::X`: The new input point where the function has been evaluated.
- `y::Y`: The corresponding function value at the input point.
- `i::Int`: The current iteration number.

returns:
- `BO::BOStruct`: The updated Bayesian Optimization structure.

Remarks:
    This function handles potential ill-conditioning issues when updating the GP,
    by returning the previous state if an error occurs and setting a flag to stop the optimization loop.
"""
function update(BO::BOStruct, x::X, y::Y, i::Int) where {X, Y}

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
        @info "We reached ill-conditioning, returning NON-UPDATED GP. Killing BO loop."
        # Issue: the gp_update in the try is updating the p.gp.gpx as it tries to create the posterior.
        # if it fails, it keeps the added x and y and overwrites the old structure, which I want to keep if it fails...
        # so now its a bit bruteforce but I try to recreate the previous GP structure. Maybe copying it every time would help.
        BO.model = prev_gp
        BO.xs = BO.xs[1:(length(BO.xs) - 1)]
        BO.ys = BO.ys[1:(length(BO.ys) - 1)]
        BO.ys_non_std = BO.ys_non_std[1:(length(BO.ys_non_std) - 1)]

        #println("Final # points for posterior: ",length(BO.xs))
        BO.flag = true # we need to stop the BO loop
        return BO
    end

    # update the ACQ function
    acq_updated = update(BO.acq, BO.ys, BO.model)

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
- `x_train::Vector{X}`: A vector of input training points.
- `y_train::Vector{Y}`: A vector of corresponding output training values.
- `old_params::Vector{Float64}`: A vector containing the current log lengthscale
    and log scale parameters.
- `length_scale_only::Bool`: If true, only optimize the lengthscale, keeping the scale fixed.
- `mean::AbstractGPs.MeanFunction`: The mean function of the GP, defaults to ZeroMean().
- `num_restarts::Int`: Number of random restarts for the optimization. If set to 1, uses the current parameters as the initial guess.

returns:
- `model::AbstractSurrogate`: The updated surrogate model with optimized hyperparameters.

"""
function optimize_hyperparameters(
        model::AbstractSurrogate,
        x_train::Vector{X},
        y_train::Vector{Y},
        old_params::Vector{T};
        scale_std::Float64 = 1.0,
        length_scale_only::Bool = false,
        num_restarts::Int = 1,
        domain::Union{Nothing, AbstractDomain} = nothing
) where {X, Y, T}
    best_nlml = Inf
    best_result = nothing

    #TODO ask the user for the bounds of the hyperparameters, so this below should dissapear

    # Adjust scale bounds by 1/σ² to account for standardization
    # This ensures bounds in original space remain 1e-3 to 1e4 regardless of standardization
    # Define original space bounds for lengthscale
    length_scale_lower, length_scale_upper = 1e-3, 1e3

    if !isnothing(domain)

        # If we have a continuous domain and training data, compute data-informed bounds
        ℓL_vec, ℓU_vec = lengthscale_bounds(x_train, domain)
        # Current models use isotropic kernels (single ℓ). Collapse per-dim bounds conservatively.

        # Lower bound should not be larger than any per-dim lower bound → take minimum.
        # Upper bound should not be smaller than any per-dim upper bound → take maximum.
        length_scale_lower = max(min(ℓL_vec...), 1e-6)
        length_scale_upper = max(ℓU_vec...)
        # Ensure sensible ordering
        @assert length_scale_lower < length_scale_upper
    end

    scale_lower, scale_upper = 1e-3, 1e6

    # Adjust scale bounds for standardized space
    adjusted_scale_lower = scale_lower / (scale_std^2)
    adjusted_scale_upper = scale_upper / (scale_std^2)

    if length_scale_only
        lower_bounds = log.([length_scale_lower])
        upper_bounds = log.([length_scale_upper])
    else
        lower_bounds = log.([length_scale_lower, adjusted_scale_lower])
        upper_bounds = log.([length_scale_upper, adjusted_scale_upper])
    end
    
    @debug "lower bounds (ℓ, scale)" exp.(lower_bounds)
    @debug "upper bounds (ℓ,scale)" exp.(upper_bounds)

    x_train_prepped = prep_input(model, x_train)
    y_train_prepped = prep_output(model, y_train)


    obj = nothing
    if length_scale_only
        # Only optimize lengthscale, keep scale fixed at original log value (second parameter)
        obj = p -> nlml_ls(
            model, p[1], old_params[2], x_train_prepped, y_train_prepped
        )
    else
        # Optimize both lengthscale and scale (vector p)
        obj = p -> nlml(
            model, p, x_train_prepped, y_train_prepped
        )
    end

    opts = Optim.Options(; g_tol = 1e-6, f_abstol = 2.2e-9)

    random_inits = [rand.(Uniform.(lower_bounds, upper_bounds))
                    for _ in 1:(num_restarts - 1)]
    if length_scale_only
        init_guesses = [[old_params[1]], random_inits...]
    else
        init_guesses = [collect(old_params), random_inits...]
    end

    inner_optimizer = LBFGS(;
        linesearch = Optim.LineSearches.HagerZhang(; linesearchmax = 20))
    
    for i in 1:num_restarts
        try
            result = Optim.optimize(
                obj,
                lower_bounds,
                upper_bounds,
                init_guesses[i],
                Fminbox(inner_optimizer),
                opts;
                autodiff = :forward #AutoMooncake(),
            )

            @debug "Optimization result: " result
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
        @info "All restarts failed to converge."
        return model
    else
        @debug "Best LML after $(num_restarts) restarts: " -best_nlml
        if length_scale_only
            @debug "Optimized lengthscale (log): $(best_result[1]), kept scale (log): $(old_params[2])"
        else
            @debug "Optimized parameters (log): lengthscale=$(best_result[1]), scale=$(best_result[2])"
        end
    end

    ℓ = nothing
    ℓ = nothing
    scale = nothing
    if length_scale_only
        ℓ = exp(best_result[1])
        scale = original_scale
    else
        ℓ, scale = exp.(best_result)
    end

    kernel_constructor = get_kernel_constructor(model)
    k_opt = scale * with_lengthscale(kernel_constructor, ℓ)

    return _update_model_parameters(model, k_opt)
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
function optimize(
        BO::BOStruct;
        standardize::Union{String, Nothing} = "mean_scale",
        hyper_params::Union{String, Nothing} = "all",
        num_restarts_HP::Int = 1
)
    @assert hyper_params in ["all", "length_scale_only", nothing] "hyper_params must be one of: 'all', 'length_scale_only', or nothing."

    @assert standardize in ["mean_scale", "scale_only", "mean_only", nothing] "standardize must be one of: 'mean_scale', 'scale_only', 'mean_only', or nothing."

    d = length(BO.xs[1])

    μ = zero(BO.ys[1])
    σ = μ isa AbstractArray ? ones(eltype(μ), size(μ)) : one(μ)
    if isnothing(standardize)
        BO.model = update(BO.model, BO.xs, BO.ys) # because we might not to that before
    else
        BO, (μ, σ) = standardize_problem(BO, standardize)
    end

    acq_list = Vector{Float64}(undef, 0)

    i = 0
    while !stop_criteria(BO) & !BO.flag
        if !isnothing(hyper_params) && (i % 10 == 0)
            @info "Optimizing GP hyperparameters at iteration $i..."
            @debug "Former parameters: ℓ=$(get_lengthscale(BO.model)), variance =$(get_scale(BO.model))"
            old_params = log.([get_lengthscale(BO.model)[1], get_scale(BO.model)[1]])
            out = nothing
            if hyper_params == "length_scale_only"
                out = optimize_hyperparameters(
                    BO.model,
                    BO.xs,
                    BO.ys,
                    old_params;
                    length_scale_only = true,
                    scale_std = σ[1],
                    num_restarts = num_restarts_HP,
                    domain = BO.domain
                )
            elseif hyper_params == "all"
                out = optimize_hyperparameters(
                    BO.model,
                    BO.xs,
                    BO.ys,
                    old_params;
                    length_scale_only = false,
                    scale_std = σ[1],
                    num_restarts = num_restarts_HP,
                    domain = BO.domain
                )
            else
                out = nothing
            end

            @debug "MLE new parameters: ℓ=$(get_lengthscale(out)), variance =$(get_scale(out))"
            if !isnothing(out)
                BO.model = out
                BO.model = update(BO.model, BO.xs, BO.ys)
            end

            @info "New parameters: ℓ=$(get_lengthscale(BO.model)), variance =$(get_scale(BO.model))"
        end

        @info "Iteration #$(i+1), current min val: $(_get_minimum(BO.model, BO.ys_non_std))"
        x_cand = optimize_acquisition(BO.acq, BO.model, BO.domain)
        x_cand = d == 1 ? first(x_cand) : x_cand

        @info "Acquisition optimized, new candidate point: $(x_cand)"
        push!(acq_list, BO.acq(BO.model, [x_cand])[1])

        y_cand = BO.func(x_cand)
        y_cand = y_cand + _noise_like(y_cand, σ = sqrt(BO.noise) / σ[1]) # Add noise to the observation

        push!(BO.ys_non_std, y_cand)

        @debug "New point acquired: $(x_cand) with acq func $(BO.acq(BO.model, x_cand))" y_cand

        i += 1

        y_cand = (y_cand - μ) ./ σ
        BO = update(BO, x_cand, y_cand, i)
    end

    return BO, acq_list, (μ, σ)
end
