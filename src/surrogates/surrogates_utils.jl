function Base.copy(gp::AbstractGPs.PosteriorGP)
    AbstractGPs.PosteriorGP(gp.prior, NamedTuple{keys(gp.data)}(map(copy, values(gp.data))))
end
Base.copy(s::Nothing) = nothing

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
