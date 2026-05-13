using Documenter
using ModelingToolkitTolerances

makedocs(
    sitename = "ModelingToolkitTolerances.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [ModelingToolkitTolerances],
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md"
    ]
)

deploydocs(
    repo = "github.com/bradcarman/MTKToleranceTools.jl.git"
)