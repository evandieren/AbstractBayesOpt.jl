## Utility functions for Bayesian Optimization



## BOStruct and related functions
"""
    Prints information about the BOStruct

Arguments:
- `BO::BOStruct`: The BOStruct instance to print information about.

returns:
- Nothing, just prints to the console.
"""
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
    println("=============================================")
end


## Standardization functions


"""
    standardize_problem(BO::BOStruct; choice="mean_scale")

Standardize the output values of the BOStruct and update the GP and acquisition function accordingly.

Arguments:
- `BO::BOStruct`: The Bayesian Optimization problem to standardize.
- `choice::String`: Standardization mode:
    - "mean_scale": remove empirical mean and scale by empirical std.
    - "scale_only": only scale by empirical std (no centering). If the GP has a non-zero prior mean, it is rescaled accordingly for consistency.
    - "mean_only": only remove the empirical mean (no scaling).

returns:
- `BO::BOStruct`: The updated BOStruct with standardized outputs and updated model/acquisition.
- `params::Tuple`: A tuple containing the mean and standard deviation used for standardization (vectors matching the output dimension).
"""
function standardize_problem(BO::BOStruct; choice="mean_scale")
    @assert choice in ["mean_scale", "scale_only", "mean_only"] "choice must be one of: 'mean_scale', 'scale_only', 'mean_only'"

    ys_non_std = BO.ys_non_std
    p = length(BO.ys[1])

    # Attention: here it is the standard deviation, need to square for kernel scaling
    μ::Vector{Float64}, σ::Vector{Float64} = get_mean_std(BO.model,ys_non_std)

    # Taking into account the choice of the user
    if choice == "scale_only"
        μ = zeros(p)
    elseif choice == "mean_only"
        σ = ones(p)
    end

    println("Standardization choice: $choice")
    println("Standardization parameters: μ=$μ, σ=$σ")


    # Need to update original kernel scale if scale_only or mean_scale
    if choice in ["scale_only", "mean_scale"]
        BO.model = update_model_scale(BO.model, σ) # Update the kernel scale / σ²
    end 

    # Need to standardize the outputs too:
    BO.ys = std_y(BO.model, ys_non_std,μ, σ)
    BO.model = update(BO.model, BO.xs, BO.ys)
    BO.acq = update(BO.acq, BO.ys, BO.model)
    
    return BO, (μ, σ)
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