# TODO list for the AbstractBayesOpt.jl package
This is the todo list for the AbstractBayesOpt.jl package


Things to do **before Oct 1st** are in **bold**

## Features

### BO Loop
- ~~**Get rid of the kernel_constructor in the call, it is a bit ugly and defeats the abstract purpose. Include this potentially in the `model`**~~
- Ensure we copy, and do not modify by reference the inputs. This is not the case for the model, which is a shame. 
    The BOStruct will have the updated model which is not ideal if we wish to run it several times.

### Surrogates
- Potentially look at Multi-Fidelity GPs (AR1 models?)
- Rank-1 updates of kernel matrix? This might creates numerical instability, but perhaps interesting for KG implementation. See [Chevalier and Ginsbourger, 2012](https://arxiv.org/abs/1203.6452)

### Optimization routines
- ~~Work out AD rule to not rely on ReverseDiff for the MLE optimization for GradientGPs.~~ -> Fixed
- We still have issues if the user decides to use Matern52, but the other ones are fine.
- Allow for custom choice of optimizer for the acq function maximisation, and/or hyper-parameter tuning if needed

### Acquisition functions
- Implement the KG. 
- Implement Thompson Sampling.
- Work on MO optimization for EnsembleAcq

### Plotting
- Provide more plotting routines, and enhance exisiting ones (might be out of date)

## Performance-related 
- Continue the profiling of the optimize function.
- **Identified issues: posterior_mean and posterior_var suffer from GC**

## Tests
- **Do the tests, both mathematical and software related.**

## Package-julia related tasks
- **Understand and define what needs to be exported or not in AbstractBayesOpt.jl**
- **Use Documenter.jl and Literate.jl for the docs for some examples and docs**
- **Work on the coverage and travis CI.**
- **Register the package**
- **Check the dependenies of the module, and clean up not used packages and compat.**
- Try to refactor the code using [BlueStyle](https://github.com/JuliaDiff/BlueStyle)