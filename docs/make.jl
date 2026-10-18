using Documenter
using OptimizationSNOPT
using SNOPT

DocMeta.setdocmeta!(OptimizationSNOPT, :DocTestSetup, :(using OptimizationSNOPT); recursive = true)

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
        "Getting started" => "getting_started.md",
        "Installation" => "installation.md",
        "How-to guides" => [
            "Set solver options" => "howto/options.md",
            "Monitor and stop a solve" => "howto/monitoring.md",
            "Solve a problem several times" => "howto/cache.md",
        ],
        "Solver behaviour" => "concepts.md",
        "API reference" => "api.md",
    ],
    # The examples need a licensed SNOPT library, so they run wherever one is found.
    # Hosted CI has none and only renders the pages. SNOPT_DOCTESTS overrides the
    # default.
    doctest = haskey(ENV, "SNOPT_DOCTESTS") ? ENV["SNOPT_DOCTESTS"] == "true" :
              SNOPT.has_snopt(),
)

deploydocs(repo = "github.com/EllissoideRotondo/OptimizationSNOPT.jl.git")
