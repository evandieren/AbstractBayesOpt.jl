using AbstractBayesOpt
using Documenter


makedocs(sitename="AbstractBayesOpt.jl",
         modules=[AbstractBayesOpt],
         checkdocs=:none)


deploydocs(
    repo = "github.com:evandieren/AbstractBayesOpt.jl.git",
    devbranch = "main"
)