"""
    ContinuousDomain(lower::Vector{Float64}, upper::Vector{Float64}, bounds::Vector{Tuple{Float64,Float64}}) <: AbstractDomain

A concrete implementation of `AbstractDomain` for continuous domains.

Attributes:
- `lower::Vector{Float64}`: The lower bounds of the domain.
- `upper::Vector{Float64}`: The upper bounds of the domain.
- `bounds::Vector{Tuple{Float64,Float64}}`: A vector of tuples representing the (lower, upper) bounds for each dimension.

Constructor:
- `ContinuousDomain(lower::Vector{Float64}, upper::Vector{Float64})`:
    Creates a `ContinuousDomain` instance given lower and upper bounds.
    Performs sanity checks to ensure the bounds are valid.
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
