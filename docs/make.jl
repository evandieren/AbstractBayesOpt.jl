using AbstractBayesOpt
using Literate
using Documenter

LITERATE_INPUT = joinpath(@__DIR__, "literate")
LITERATE_OUTPUT = joinpath(@__DIR__, "src")

for dir_path in filter(isdir, readdir(joinpath(@__DIR__, "literate"), join = true))
for dir_path in filter(isdir, readdir(joinpath(@__DIR__, "literate"), join = true))
    dirname = basename(dir_path)

    for (root, _, files) in walkdir(dir_path), file in files
    for (root, _, files) in walkdir(dir_path), file in files
        # ignore non julia files
        splitext(file)[2] == ".jl" || continue
        # full path to a literate script
        ipath = joinpath(root, file)
        # generated output path
        opath = splitdir(replace(ipath, LITERATE_INPUT => LITERATE_OUTPUT))[1]
        # generate the markdown file calling Literate
        Literate.markdown(ipath, opath)
    end
end

makedocs(;
    sitename = "AbstractBayesOpt.jl",
    modules = [AbstractBayesOpt],
    checkdocs = :none,
    pages = [
    sitename = "AbstractBayesOpt.jl",
    modules = [AbstractBayesOpt],
    checkdocs = :none,
    pages = [
        "Home" => "index.md",
        "Public API" => "reference.md",
        "Tutorials" => [
            "1D Bayesian Optimisation" => "tutorials/1D_BO.md",
            "2D Bayesian Optimisation" => "tutorials/2D_BO.md",
        ]
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ))
deploydocs(
    repo = "github.com:evandieren/AbstractBayesOpt.jl.git",
    devbranch = "main"
)
