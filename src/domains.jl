"""
Implements different types of domains

"""

"""
    ContinuousDomain

A concrete implementation of `AbstractDomain` for continuous domains.

Contains a constructor that takes in lower and upper bounds as vectors of floats.
"""
struct ContinuousDomain <: AbstractDomain
    lower::Vector{Float64}
    upper::Vector{Float64}
    bounds::Vector{Tuple{Float64,Float64}} # zipped lower and upper

    function ContinuousDomain(lower::Vector{Float64}, upper::Vector{Float64})

        #Sanity check
        length(lower) == length(upper) || throw(ArgumentError("Bounds mismatch"))
        all(lower .<= upper) || throw(ArgumentError("Invalid bounds"))
        # Creates the structure
        return new(lower, upper, collect(zip(lower, upper)))
    end
end
