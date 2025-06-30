normpdf(μ, σ²) = 1 / √(2π * σ²) * exp(-μ^2 / (2 * σ²))
normcdf(μ, σ²) = 1 / 2 * (1 + erf(μ / √(2σ²)))

using Optim
using Random
using Surrogates

inner_optimizer = LBFGS(;linesearch = Optim.LineSearches.HagerZhang(linesearchmax=20))
box_optimizer = Fminbox(inner_optimizer)

function optimize_acquisition!(acqf::AbstractAcquisition,
                               surrogate::AbstractSurrogate,
                               domain::ContinuousDomain;
                               n_grid = 10000,
                               n_local = 100)
    # We will use BFGS for now
    best_acq = -Inf
    best_x = nothing

    # Random search
    d = length(domain.bounds)
    grid_points = [domain.lower .+ rand(d) .* (domain.upper .- domain.lower) for _ in 1:n_grid]
    evaluated = [(x, acqf(surrogate, x)) for x in grid_points]
    sorted_points = sort(evaluated, by = x -> -x[2])  # higher EI is better
    top_points = first.(sorted_points[1:min(n_local, length(sorted_points))])
    
    # Loop over a number of random starting points
    for initial_x in top_points
        result = Optim.optimize(x -> -acqf(surrogate, x),
                                domain.lower,
                                domain.upper,
                                initial_x,
                                box_optimizer,
                                Optim.Options(g_tol = 1e-5, f_abstol = 2.2e-9, x_abstol = 1e-4))
        # Check if the current run is better (lower negative acqf)
        current_acq = -Optim.minimum(result)
        if current_acq > best_acq
            best_acq = current_acq
            best_x = Optim.minimizer(result)
        end
    end
    return best_x
end

function sample_gp_function(surrogate::AbstractSurrogate, domain::ContinuousDomain;n_points=250)
    d = length(domain.bounds)
    X = [domain.lower .+ rand(d) .* (domain.upper .- domain.lower) for _ in 1:n_points]
    X_mat = reduce(hcat, X)'  # shape (n_points, d)
    # Sample from the GP at these points
    y = rand(surrogate.gpx(X))  # 1 sample from posterior
    #println("here",y)
    
    itp = RadialBasis(X_mat, y, domain.lower, domain.upper)

    return x -> -itp(x)

end

function optimize_mean!(surrogate::AbstractSurrogate, domain::ContinuousDomain; n_restarts = 30)
    # This will minimize the posterior mean of the surrogate. Similar to optimize_acquisition!
    best_μ = Inf
    best_x = nothing
    for i in 1:n_restarts
        # Generate a random starting point within the bounds
        initial_x = [rand()*(u - l) + l for (l, u) in domain.bounds]
        result = Optim.optimize(x -> posterior_mean(surrogate, x),
                                domain.lower,
                                domain.upper,
                                initial_x,
                                box_optimizer,
                                Optim.Options(g_tol = 1e-5, f_abstol = 2.2e-9)
                                ; autodiff = :forward)
        current_μ = Optim.minimum(result)
        if current_μ < best_μ
            best_μ = current_μ
            best_x = Optim.minimizer(result)
        end
    end
    return best_x, best_μ
end

# Here we will have a look when we will do the gradient approximate function.
# #TODO Maybe switch to the Optim one later on
# # Gradient ascent for KG acquisition function
# function gradient_ascent(f, ∇f, x₀, lr, T, tol=1e-6)
#     x = copy(x0)
#     for t in 1:T
#         x_new = x + lr * ∇f(x)   # ascent step

#         if norm(x_new - x) < tol # indirectly linked to value of ∇f(x)
#             return x_new, f(x_new)
#         end

#         x = x_new
#     end

#     return x, f(x)
# end