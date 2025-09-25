# Reference (Public API)

```@meta
CollapsedDocStrings = true
```

This section documents the exported functions, types, etc from
**AbstractBayesOpt.jl**.

---

## Bayesian Optimisation loop

```@docs
BOStruct
optimize
```

## Abstract Interface

```@docs
AbstractAcquisition
AbstractDomain
AbstractSurrogate
```

## Surrogates

```@docs
StandardGP
GradientGP
posterior_mean
posterior_var
nlml
```

### Kernels

```@docs
ApproxMatern52Kernel
```

### GradientGP-related functions

```@docs
gradConstMean
gradKernel
posterior_grad_mean
posterior_grad_var
posterior_grad_cov
```

## Acquisition Functions

```@docs
EnsembleAcquisition
ExpectedImprovement
GradientNormUCB
ProbabilityImprovement
UpperConfidenceBound
```

## Domains

### Continuous domain

```@docs
ContinuousDomain
```
