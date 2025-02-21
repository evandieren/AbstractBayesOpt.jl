module GPSurrogates
"""
    GPSurrogates

Implementation of the Abstract structures for the GP. 


Reason: This is a simple wrapper around AbstractGPs but can be useful to differentiate when using gradients or not.
"""


using AbstractGPs
using KernelFunctions
using Abstract

struct StandardGP <: AbstractSurrogate
    gp:AbstractGPs.GP
    kernel::KernelFunctions.Kernel
end

function posterior_stats(model::StandardGP, X)
    μ_value, σ_value = mean_and_std(model.gp(X))
    return (μ_value, σ_value)
end

#struct GradientGP <: AbstractSurrogate
#    gp:AbstractGPs.GP
#    kernel::KernelFunctions.Kernel
#end


end # module GPSurrogates