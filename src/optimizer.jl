"""
This file contains various optimization techniques used throughout the project.

Put it in some utils?
"""

using Optim

function optimize_acquisition!(acqf::AbstractAcquisition, surrogate::AbstractSurrogate, domain::ContinuousDomain)
    # We will use BFGS for now

    optim_bounds = Optim.Fminbox(LBFGS()).(domain.bounds)

    # Initial guess: midpoint of domain
    initial_x = [0.5*(l+u) for (l,u) in domain.bounds]

    optimize(x -> -acqf(surrogate, x), 
                     initial_x, 
                     LBFGS(),
                     autodiff = :forward,
                     lower = domain.lower,
                     upper = domain.upper)
end