"""
This file contains various optimization techniques used throughout the project.
"""

using Optim

function optimize_acquisition!(acqf::AbstractAcquisition, surrogate::AbstractSurrogate, domain::ContinuousDomain; n_restarts = 10)
    # We will use BFGS for now
    inner_optimizer = LBFGS(;linesearch = Optim.LineSearches.HagerZhang(linesearchmax=20))
    box_optimizer = Fminbox(inner_optimizer)

    best_acq = Inf
    best_x = nothing

    # Loop over a number of random starting points
    for i in 1:n_restarts
        println("Restart n*",i)
        # Generate a random starting point within the bounds
        initial_x = [rand()*(u - l) + l for (l, u) in domain.bounds]
        result = Optim.optimize(x -> -acqf(surrogate, x),
                                domain.lower,
                                domain.upper,
                                initial_x,
                                box_optimizer,
                                Optim.Options(g_tol = 1e-5, f_tol = 2.2e-9)
                                ; autodiff = :forward
                                )
        # Check if the current run is better (lower negative acqf)
        current_acq = Optim.minimum(result)
        if current_acq < best_acq
            best_acq = current_acq
            best_x = Optim.minimizer(result)
        end
    end
    return best_x
end