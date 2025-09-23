# Reference (Public API)

This section documents the exported functions, types, and macros from **AbstractBayesOpt.jl**.

---

## Abstract Interface

```@autodocs
Modules = [AbstractBayesOpt]
Pages = ["abstract.jl"]
Private = false
```
## Surrogates

### StandardGP
```@autodocs
Modules = [AbstractBayesOpt]
Pages = ["surrogates/StandardGP.jl"]
Private = false
```

### GradientGP
```@autodocs
Modules = [AbstractBayesOpt]
Pages = ["surrogates/GradientGP.jl"]
Private = false
```

## Acquisition Functions

### Upper Confidence Bound
```@autodocs
Modules = [AbstractBayesOpt]
Pages = ["acquisition/UpperConfidenceBound.jl"]
Private = false
```

### Expected Improvement
```@autodocs
Modules = [AbstractBayesOpt]
Pages = ["acquisition_functions/ExpectedImprovement.jl"]
Private = false
```

### Probability of Improvement
```@autodocs
Modules = [AbstractBayesOpt]
Pages = ["acquisition/ProbabilityImprovement.jl"]
Private = false
```

### gradient norm UCB
```@autodocs
Modules = [AbstractBayesOpt]
Pages = ["acquisition/gradNormUCB.jl"]
Private = false
```

### Ensemble of Acquisitions
```@autodocs
Modules = [AbstractBayesOpt]
Pages = ["acquisition/EnsembleAcq.jl"]
Private = false
```

## Domains

### Continuous domain
```@autodocs
Modules = [AbstractBayesOpt]
Pages = ["domains.jl"]
Private = false
```