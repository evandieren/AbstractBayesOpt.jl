# struct ThompsonSampling <: AbstractAcquisition
#     f_sample::Function
#     domain::ContinuousDomain
# end

# function (ts::ThompsonSampling)(surrogate::AbstractSurrogate, x)
#     return ts.f_sample(x)
# end

# function update!(acqf::ThompsonSampling, ys::AbstractVector, surrogate::AbstractSurrogate)
#     # Sample a new function from the updated GP posterior
#     f_sample = sample_gp_function(surrogate, acqf.domain)  # assumes you’ve set domain inside surrogate

#     # Return a new acquisition function with updated sample
#     return ThompsonSampling(f_sample,acqf.domain)
# end
