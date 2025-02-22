"""
Implements different types of domains

"""

struct ContinuousDomain <: AbstractDomain
    lower::Vector{Float64}
    upper::Vector{Float64}
    bounds::Vector{Tuple{Float64,Float64}} # zipped lower and upper
    
    function ContinuousDomain(lower::Vector{Float64}, upper::Vector{Float64})

        #Sanity check
        length(lower) == length(upper)
        all(lower .<= upper)
        # Creates the structure
        new(lower, upper, collect(zip(lower, upper)))
    end
end