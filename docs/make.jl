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

for file in filter(endswith(".qmd"), readdir(REFERENCE_DIR; join=true))
  text = read(file, String)
  occursin(REF_LINK, text) || continue
  for m in eachmatch(REF_LINK, text)
    isfile(joinpath(REFERENCE_DIR, m[1] * ".qmd")) ||
      @warn "No reference page for `$(m[1])`, linked from $(basename(file))"
  end
  write(file, replace(text, REF_LINK => s"[`\1`](\1.qmd)"))
end
