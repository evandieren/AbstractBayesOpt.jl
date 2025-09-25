"""
    Base.copy(gp::AbstractGPs.PosteriorGP)

Create a copy of a PosteriorGP object.

Arguments:
- `gp::AbstractGPs.PosteriorGP`: The PosteriorGP object to be copied.

returns:
- `AbstractGPs.PosteriorGP`: A new PosteriorGP object that is a copy of the input.
"""
function Base.copy(gp::AbstractGPs.PosteriorGP)
    AbstractGPs.PosteriorGP(gp.prior, NamedTuple{keys(gp.data)}(map(copy, values(gp.data))))
end
Base.copy(s::Nothing) = nothing

"""
    extract_scale_and_lengthscale(kernel::Kernel)

Extract the scale and lengthscale from a given kernel.

Arguments:
- `kernel::Kernel`: The kernel from which to extract the scale and lengthscale.

returns:
- `(inner::Kernel, scale::Real, lengthscale::Union{Real, Nothing})`: A tuple containing the inner kernel, scale, and lengthscale.
"""
function extract_scale_and_lengthscale(kernel::Kernel)
    scale = 1.0
    lengthscale = nothing
    inner = kernel

    # Unwrap ScaledKernel to get scale
    if isa(inner, AbstractGPs.ScaledKernel)
        scale = inner.σ²[1]
        inner = inner.kernel
    end

    # Unwrap TransformKernel to get lengthscale
    if isa(inner, KernelFunctions.TransformedKernel) &&
        isa(inner.transform, KernelFunctions.ScaleTransform)
        lengthscale = 1 / inner.transform.s[1]
        inner = inner.kernel
    end

    return (inner, scale, lengthscale)
end
