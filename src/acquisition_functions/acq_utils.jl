"""
    _get_box_optimizer()

Helper function to create a box-constrained optimizer using L-BFGS.

returns:
- `box_optimizer::Fminbox`: Box-constrained optimizer instance
"""
function _get_box_optimizer()
    inner_optimizer = LBFGS(; linesearch=Optim.LineSearches.HagerZhang(; linesearchmax=20))
    box_optimizer = Fminbox(inner_optimizer)
    return box_optimizer
end

"""
    optimize_acquisition(acqf::AbstractAcquisition, surrogate::AbstractSurrogate, 
                         domain::AbstractDomain; n_grid::Int=10000, n_local::Int=100)

Optimize the acquisition function over the given domain.


Arguments:
- `acqf::AbstractAcquisition`: The acquisition function to optimize.
- `surrogate::AbstractSurrogate`: The surrogate model used by the acquisition function.
- `domain::AbstractDomain`: The domain over which to optimize the acquisition function.
- `n_grid::Int`: Number of random grid points to sample for initialisation (default: 10000).
- `n_local::Int`: Number of top points from the grid to use as starting points.

returns:
- `best_x::Vector{Float64}`: The point in the domain that maximizes
    the acquisition function.
"""
function optimize_acquisition(
    acqf::AbstractAcquisition,
    surrogate::AbstractSurrogate,
    domain::AbstractDomain;
    n_grid::Int=10000,
    n_local::Int=100,
)
    # We will use BFGS for now
    best_acq = -Inf
    best_x = nothing

    d = length(domain.bounds)
    grid_points = [
        domain.lower .+ rand(d) .* (domain.upper .- domain.lower) for _ in 1:n_grid
    ]

    # println("Grid points generated: ", grid_points[1:5])
    scores = acqf(surrogate, grid_points)
    indices_sorted = sortperm(scores; rev=true)
    top_points = grid_points[indices_sorted[1:min(n_local, length(indices_sorted))]]

    # Loop over a number of random starting points
    for initial_x in top_points
        result = Optim.optimize(
            x -> -acqf(surrogate, [x])[1],
            domain.lower,
            domain.upper,
            initial_x,
            _get_box_optimizer(),
            Optim.Options(; g_tol=1e-5, f_abstol=2.2e-9, x_abstol=1e-4),
        )
        # Check if the current run is better (lower negative acqf)

        current_acq = -Optim.minimum(result)
        if current_acq > best_acq
            best_acq = current_acq
            best_x = Optim.minimizer(result)
        end
    end
    return best_x
end