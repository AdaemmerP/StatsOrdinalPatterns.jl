using StatsOrdinalPatterns
using QuartoDocBuilder

# QuartoDocBuilder writes to "docs/reference", relative to the working directory,
# so the script has to run from the repository root.
cd(dirname(@__DIR__))

# Regenerate the reference pages from the docstrings only. `quarto_build_site`
# would additionally rewrite `docs/_quarto.yml` with `force=true` and thereby
# discard the hand-curated navbar and sidebar.
quarto_rebuild_reference(StatsOrdinalPatterns)

# The docstrings use Documenter's cross reference syntax [`name`](@ref). Quarto does not
# resolve it, so these links would render as href="@ref" and lead nowhere. Point each one
# to the reference page of that name instead. All reference pages sit in one directory,
# so a plain relative link works. The docstrings themselves keep the Documenter syntax
# and therefore also work under Documenter.
const REFERENCE_DIR = joinpath("docs", "reference")
const REF_LINK = r"\[`([^`]+)`\]\(@ref\)"

# Many docstrings also mention exported names as plain code, e.g. `Persistence()` in a
# list of chart choices. These get linked as well, except on the page of the name
# itself. `Shannon` and `ShannonExtropy` come from ComplexityMeasures.jl and link to
# its documentation. Code blocks and headings are left untouched, and so are code spans
# that already are links.
const CM_DOCS = "https://juliadynamics.github.io/ComplexityMeasures.jl/stable/information_measures/"
const EXTERNAL = Dict(
  "Shannon" => CM_DOCS * "#ComplexityMeasures.Shannon",
  "ShannonExtropy" => CM_DOCS * "#ComplexityMeasures.ShannonExtropy",
)
const EXPORTED = Set(string.(names(StatsOrdinalPatterns)))
const CODE_SPAN = r"(?<!\[)`([A-Za-z_][A-Za-z0-9_!]*)(\([^`]*\))?`(?!\]\()"

function link_code_spans(text, self)
  incode = false
  lines = map(split(text, '\n')) do line
    stripped = strip(line)
    startswith(stripped, "```") && (incode = !incode; return line)
    (incode || startswith(stripped, "#")) && return line
    replace(line, CODE_SPAN => function (span)
      name = match(CODE_SPAN, span)[1]
      haskey(EXTERNAL, name) && return "[$span]($(EXTERNAL[name]))"
      linkable = name != self && name in EXPORTED &&
                 isfile(joinpath(REFERENCE_DIR, name * ".qmd"))
      return linkable ? "[$span]($name.qmd)" : span
    end)
  end
  return join(lines, '\n')
end

for file in filter(endswith(".qmd"), readdir(REFERENCE_DIR; join=true))
  text = read(file, String)
  for m in eachmatch(REF_LINK, text)
    isfile(joinpath(REFERENCE_DIR, m[1] * ".qmd")) ||
      @warn "No reference page for `$(m[1])`, linked from $(basename(file))"
  end
  text = replace(text, REF_LINK => s"[`\1`](\1.qmd)")
  write(file, link_code_spans(text, first(splitext(basename(file)))))
end
