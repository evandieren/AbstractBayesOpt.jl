# TODO list for the AbstractBayesOpt.jl package
This is the todo list for the AbstractBayesOpt.jl package

Should think about changing package name because conflicting with the Wrapper https://github.com/jbrea/BayesOpt.jl :)

## Features

### BO Loop
- Ensure we copy, and do not modify by reference the inputs. This is not the case for the model, which is a shame. 
    The BOStruct will have the updated model which is not ideal if we wish to run it several times.

### Surrogates
- Potentially look at Multi-Fidelity GPs (AR1 models?)
- Rank-1 updates of kernel matrix? This might creates numerical instability, but perhaps interesting for KG implementation. See [Chevalier and Ginsbourger, 2012](https://arxiv.org/abs/1203.6452)


### Optimization routines
- Work out AD rule to not rely on ReverseDiff for the MLE optimization for GradientGPs.
- Allow for custom choice of optimizer for the acq function maximisation, and/or hyper-parameter tuning if needed

### Acquisition functions
- Implement the KG. 
- Implement Thompson Sampling.
- Work on MO optimization for EnsembleAcq

### Plotting
- Provide more plotting routines, and enhance exisiting ones (might be out of date)

## Performance-related 
- Continue the profiling of the optimize function.
- Identified issues: posterior_mean and posterior_var suffer from GC

## Tests
- Do the tests, both mathematical and software related.

## Package-julia related tasks
- Use Documenter.jl and Literate.jl for the docs.
- Work on the coverage and travis CI.
- Check how to register the package.
- Check the dependenies of the module, and clean up not used packages and compat.