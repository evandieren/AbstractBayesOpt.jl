using AbstractBayesOpt
using Literate
using Documenter

makedocs(;
    sitename="AbstractBayesOpt.jl",
    modules=[AbstractBayesOpt],
    checkdocs=:none,
    pages=[
        "Home" => "index.md",
        "Tutorials" => ["StandardGP - 1D BO" => "tutorials/StandardGP_1D.md"],
    ],
)
deploydocs(; repo="github.com:evandieren/AbstractBayesOpt.jl.git", devbranch="main")
