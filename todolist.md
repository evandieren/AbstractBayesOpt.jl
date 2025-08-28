# TODO list for the BayesOpt.jl package
This is the todo list for the BayesOpt.jl package

## Features

### Surrogates
- Potentially look at Multi-Fidelity GPs (AR1 models?)

### Optimization routines
- Work out AD rule to not rely on ReverseDiff for the MLE optimization for GradientGPs.

### Acquisition functions
- Implement the KG. 
- Implement Thompson Sampling.
- Work on MO optimization for EnsembleAcq

### Plotting
- Provide more plotting routines, and enhance exisiting ones (might be out of date)

## Performance-related 
- Continue the profiling of the optimize function.

## Tests
- Do the tests, both mathematical and software related.

## Package-julia related tasks
- Use Documenter.jl and Literate.jl for the docs.
- Work on the coverage and travis CI.
- Check how to register the package.
- Check the dependenies of the module, and clean up not used packages and compat.