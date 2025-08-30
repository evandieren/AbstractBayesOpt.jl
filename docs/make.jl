using AbstractBayesOpt
using Documenter


makedocs(sitename="AbstractBayesOpt.jl",
         modules=[AbstractBayesOpt])


deploydocs(
    repo = "github.com:evandieren/AbstractBayesOpt.jl.git",
    devbranch = "main"
)