"""
Knowledge Gradient acquisition function. **UNDER DEVELOPMENT**

Arguments:
- `domain::AbstractDomain`: Domain over which to optimize the acquisition function
- `best_μ::AbstractVector`: Best predicted mean value from the surrogate model

returns:
- `KG::KnowledgeGradient`: Knowledge Gradient acquisition function instance

References:
[Frazier et al., 2009](https://pubsonline.informs.org/doi/10.1287/ijoc.1080.0314)
"""
struct KnowledgeGradient <: AbstractAcquisition
    domain::AbstractDomain
    best_μ::AbstractVector
end

function (KG::KnowledgeGradient)(surrogate::AbstractSurrogate, x::AbstractVector, J=1_000)
    Δ = zeros(J)
    μ = posterior_mean(surrogate, x)
    σ² = posterior_var(surrogate, x)

    y = μ .+ sqrt(σ²) .* randn(J)
    for j in 1:J
        new_model = update(
            surrogate, [surrogate.gpx.data.x; x], [surrogate.gpx.data.δ; y[j]]
        )

        μ_new = optimize_mean!(new_model, KG.domain; n_restarts=3)[2]

        Δ[j] = KG.best_μ[1] - μ_new[1]
    end
    return mean(Δ)
end

"""
Update the Knowledge Gradient acquisition function with new array of observations.

Arguments:
- `acqf::KnowledgeGradient`: Current Knowledge Gradient acquisition function
- `ys::AbstractVector`: Array of updated observations
- `surrogate::AbstractSurrogate`: Surrogate model

returns:
- `KG::KnowledgeGradient`: Updated Knowledge Gradient acquisition function
"""
function update(acq::KnowledgeGradient, ys::AbstractVector, surrogate::AbstractSurrogate)
    return KnowledgeGradient(domain, optimize_mean!(surrogate, acq.domain)[2])
end

# function ∇KG(surrogate::AbstractSurrogate, x, xs, ys, opt_params::Dict, J=1_000)
#     domain, box_optimizer, options, n_restarts = opt_params["domain"], opt_params["box_optimizer"], opt_params["options"], opt_params["n_restarts"]
#     μ = posterior_mean(surrogate, x)
#     σ² = posterior_var(surrogate, x)

#     y = μ + σ².*randn(J)
#     xs_n = [copy(xs);x]
#     ys_j = copy(ys)
#     for j = 1:J
#         new_model = update!(surrogate,xs_n,[ys_j;y[j]])
#         best_μ = Inf
#         best_x = nothing
#         for i in 1:n_restarts
#             # Generate a random starting point within the bounds
#             initial_x = [rand()*(u - l) + l for (l, u) in domain.bounds]
#             result = Optim.optimize(x -> posterior_mean(new_model, x),
#                                     domain.lower,domain.upper,
#                                     initial_x,
#                                     box_optimizer,
#                                     options
#                                     ; autodiff = :forward)
#             current_μ = Optim.minimum(result)
#             if current_μ < best_μ
#                 best_μ = current_acq
#                 best_x = Optim.minimizer(result)
#             end
#         end
#         # best_x is the x̂^* in the Frazier paper.

#         function grad_μ_x(x_star,x)
#             x_prepped = prep_input(model, xs_n)
#             kernel_mat = KernelFunctions.kernelmatrix(surrogate.gp.kernel, x_prepped)

#         end
#     end

# end
