# Connection to ComplexityMeasures.jl
#
# The chart statistics of this package are functions of ordinal pattern probabilities.
# ComplexityMeasures.jl estimates these probabilities with its outcome space
# `OrdinalPatterns`, and both packages number the m! patterns in the same lexicographic
# (Lehmer) order. The methods below therefore compute the statistics of this package
# from a `Probabilities` object of ComplexityMeasures:
#
#   complexity(Persistence(), probabilities(OrdinalPatterns{3}(d), x))
#   complexity(DistanceToWhiteNoise(), probabilities(OrdinalPatterns{3}(d), x))
#
# All chart types of this package are complexity measures (`ComplexityEstimator`), not
# information measures. ComplexityMeasures requires an information measure to be a
# functional of any probability vector, computed by `information(measure, p)`. The
# statistics here instead need the probabilities of particular ordinal patterns, or,
# for Δ, the total number of patterns. Δ is a disequilibrium in the sense of
# `StatisticalComplexity` of ComplexityMeasures, which is a `ComplexityEstimator` for
# the same reason. `Shannon` and `ShannonExtropy` are information measures of
# ComplexityMeasures and need no methods here.
# Every method evaluates the same `chart_stat_op` or `stat_sop` as the functions of
# this package, so both routes give the same values (see test_complexitymeasures_api.jl).

using ComplexityMeasures: ComplexityMeasures, Probabilities, OrdinalPatterns,
  probabilities, outcomes

# The four statistics of Bandt (2019) and the distance to white noise
const BandtChart = Union{UpDownBalance,Persistence,RotationalAsymmetry,UpDownScaling}
const OrdinalChart = Union{BandtChart,DistanceToWhiteNoise}

# The SOP statistics of Weiß and Kim (2024)
const SOPChart = Union{TauHat,KappaHat,TauTilde,KappaTilde}

# Convert ordinal pattern probabilities of ComplexityMeasures into the vector of length
# m! that `chart_stat_op` expects, indexed by the Lehmer code, and return it together
# with the pattern length m. `probabilities` omits patterns that do not occur, so they
# are filled with zeros here; the Δ statistic depends on all m! entries.
function _lehmer_probabilities(p::Probabilities)
  os = outcomes(p)
  is_pattern(o) = o isa AbstractVector{<:Integer} && sort(collect(o)) == 1:length(o)
  (!isempty(os) && all(is_pattern, os) && allequal(length.(os))) || throw(ArgumentError(
    "the probabilities must come from the ordinal pattern outcome space, e.g. " *
    "probabilities(OrdinalPatterns{3}(1), x)."
  ))
  m = length(first(os))
  q = zeros(factorial(m))
  for (o, prob) in zip(os, p)
    q[perm_to_lehm_idx(collect(o))] += prob
  end
  return q, m
end

# Δ is defined for every pattern length
_check_pattern_length(::DistanceToWhiteNoise, m) = nothing

function _check_pattern_length(chart_choice::BandtChart, m)
  allowed = chart_choice isa UpDownBalance ? (2, 3) : (3,)
  m in allowed || throw(ArgumentError(
    "$(chart_choice) is defined for ordinal patterns of length " *
    join(allowed, " or ") * ", got length $m."
  ))
  return nothing
end

"""
    complexity(chart_choice, p::Probabilities)
    complexity(chart_choice, x::AbstractVector)

Compute an ordinal pattern statistic of this package through the interface of
ComplexityMeasures.jl. `chart_choice` is one of the four statistics of Bandt (2019),
[`UpDownBalance`](@ref)`()`, [`Persistence`](@ref)`()`,
[`RotationalAsymmetry`](@ref)`()` and [`UpDownScaling`](@ref)`()`, or the distance to
white noise [`DistanceToWhiteNoise`](@ref)`()`.

The first method takes ordinal pattern probabilities, for example
`probabilities(OrdinalPatterns{m}(d), x)` with pattern length `m` and delay `d`.
Patterns that do not occur count with probability zero. The pattern length must be 3
for the Bandt statistics, or 2 or 3 for `UpDownBalance()`; `DistanceToWhiteNoise()`
accepts any length. The second method takes a time series and uses patterns of length
3 with delay 1.

Returns the same value as `stat_op(x; chart_choice, m, d)[1]`; see [`stat_op`](@ref).
ComplexityMeasures.jl breaks ties between equal values at random, while
[`stat_op`](@ref) breaks them by position, so the two agree exactly for data without
ties.

# Examples
```julia
using ComplexityMeasures
x = randn(500)
complexity(Persistence(), x)                                        # delay 1
complexity(Persistence(), probabilities(OrdinalPatterns{3}(2), x))  # delay 2
complexity(DistanceToWhiteNoise(), probabilities(OrdinalPatterns{4}(1), x))
```
"""
function ComplexityMeasures.complexity(chart_choice::OrdinalChart, p::Probabilities)
  q, m = _lehmer_probabilities(p)
  _check_pattern_length(chart_choice, m)
  return chart_stat_op(q, chart_choice)
end

function ComplexityMeasures.complexity(chart_choice::OrdinalChart, x::AbstractVector{<:Real})
  return ComplexityMeasures.complexity(chart_choice, probabilities(OrdinalPatterns{3}(1), x))
end

"""
    complexity(chart_choice, X::AbstractMatrix)

Compute one of the spatial ordinal pattern (SOP) statistics of Weiß and Kim (2024)
through the interface of ComplexityMeasures.jl. `chart_choice` is one of
[`TauHat`](@ref)`()`, [`KappaHat`](@ref)`()`, [`TauTilde`](@ref)`()` or
[`KappaTilde`](@ref)`()`.

Uses the classical SOP classification at delays `(1, 1)` and returns the same value as
`stat_sop(X, 1, 1; chart_choice)[1]`; see [`stat_sop`](@ref), which also covers other
delays, the refined classifications and the entropy statistics.
"""
function ComplexityMeasures.complexity(chart_choice::SOPChart, X::AbstractMatrix{<:Real})
  return stat_sop(X isa Matrix ? X : Matrix(X), 1, 1; chart_choice=chart_choice)[1]
end
