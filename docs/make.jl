using Documenter
using OptimizationSNOPT

makedocs(
    sitename = "OptimizationSNOPT.jl",
    modules = [OptimizationSNOPT],
    checkdocs = :exports,
    format = Documenter.HTML(
        canonical = "https://EllissoideRotondo.github.io/OptimizationSNOPT.jl/dev/",
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Getting started" => "quickstart.md",
        "Configuration" => "configuration.md",
        "API reference" => "api.md",
    ],
)

deploydocs(repo = "github.com/EllissoideRotondo/OptimizationSNOPT.jl.git")
