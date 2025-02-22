"""
This file contains various optimization techniques used throughout the project.

Put it in some utils?
"""

using Optim

function optimize_acquisition!(acqf::AbstractAcquisition, surrogate::AbstractSurrogate, domain::ContinuousDomain)
    # We will use BFGS for now
    inner_optimizer = LBFGS()
    box_optimizer = Fminbox(inner_optimizer)

    # Initial guess: midpoint of domain
    initial_x = [0.5*(l+u) for (l,u) in domain.bounds]
    results = Optim.optimize(x -> -acqf(surrogate, x),
                     domain.lower,
                     domain.upper,
                     initial_x,
                     box_optimizer)
    return Optim.minimizer(results)
end