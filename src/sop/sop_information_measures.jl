
"""
    TauHat()

Chart choice for spatial ordinal patterns (SOPs): the statistic
``\\hat{\\tau} = \\hat{p}_1 - 1/3``, based on the relative frequency of the first SOP
type.
"""
struct TauHat <: ComplexityEstimator end

"""
    KappaHat()

Chart choice for spatial ordinal patterns (SOPs): the statistic
``\\hat{\\kappa} = \\hat{p}_2 - \\hat{p}_3``, the difference of the relative frequencies
of the second and third SOP type.
"""
struct KappaHat <: ComplexityEstimator end

"""
    TauTilde()

Chart choice for spatial ordinal patterns (SOPs): the statistic
``\\tilde{\\tau} = \\tilde{p}_3 - 1/3``, based on the third type frequency of the
refined SOP classification.
"""
struct TauTilde <: ComplexityEstimator end

"""
    KappaTilde()

Chart choice for spatial ordinal patterns (SOPs): the statistic
``\\tilde{\\kappa} = \\tilde{p}_1 - \\tilde{p}_2``, the difference of the first and
second type frequencies of the refined SOP classification.
"""
struct KappaTilde <: ComplexityEstimator end

"""
    SOPClassification

Abstract supertype for the classification schemes of spatial ordinal patterns (SOPs) and
the type of every `refinement` argument in the package. The concrete schemes are
[`OrdinaryType`](@ref) for the classical classification and the subtypes of
[`RefinedType`](@ref) for the refined ones.
"""
abstract type SOPClassification end

"""
    OrdinaryType

Classical ("ordinary") classification of spatial ordinal patterns (SOPs) into three
types; see Weiß and Kim (2024). It is the default `refinement` of every SOP function.
"""
struct OrdinaryType <: SOPClassification end

"""
    RefinedType

Abstract supertype for the refined classification schemes of spatial ordinal patterns
(SOPs); see Weiß and Kim (2025). Concrete subtypes are [`RotationType`](@ref),
[`DirectionType`](@ref), and [`DiagonalType`](@ref).
"""
abstract type RefinedType <: SOPClassification end

"""
    RotationType()

Refined SOP classification that groups spatial ordinal patterns by rotational symmetry;
see Weiß and Kim (2025).
"""
struct RotationType <: RefinedType end

"""
    DirectionType()

Refined SOP classification that groups spatial ordinal patterns by direction; see Weiß
and Kim (2025).
"""
struct DirectionType <: RefinedType end

"""
    DiagonalType()

Refined SOP classification that groups spatial ordinal patterns by their diagonal
behavior; see Weiß and Kim (2025).
"""
struct DiagonalType <: RefinedType end


"""
    chart_stat_sop(p_vec, chart_choice)

Compute the chart statistic of a spatial ordinal pattern (SOP) distribution `p_vec` for
the given chart choice.

- `p_vec`: vector of (relative) SOP type frequencies.
- `chart_choice`: one of [`TauHat`](@ref)`()`, [`KappaHat`](@ref)`()`,
  [`TauTilde`](@ref)`()`, [`KappaTilde`](@ref)`()`, `Shannon()`, `ShannonExtropy()`,
  `DistanceToWhiteNoise()`.
  For `Shannon` and `ShannonExtropy`, the statistic is in the logarithm base of the
  chart, which must be larger than 1. Both default to base 2 in ComplexityMeasures.jl;
  use `Shannon(base=exp(1))` for the natural logarithm used in the papers.

Returns the value of the chart statistic.
"""
function chart_stat_sop(p_vec, ::TauHat)
  return p_vec[1] - 1.0 / 3.0
end

function chart_stat_sop(p_vec, ::KappaHat)
  return p_vec[2] - p_vec[3]
end

function chart_stat_sop(p_vec, ::TauTilde)
  return p_vec[3] - 1.0 / 3.0
end

function chart_stat_sop(p_vec, ::KappaTilde)
  return p_vec[1] - p_vec[2]
end


function chart_stat_sop(p_vec, chart_type::Shannon{T}) where {T}
  return chart_stat_op(p_vec, chart_type)
end

function chart_stat_sop(p_vec, chart_type::ShannonExtropy{T}) where {T}
  return chart_stat_op(p_vec, chart_type)
end

function chart_stat_sop(p_vec, chart_type::DistanceToWhiteNoise)
  return chart_stat_op(p_vec, chart_type)
end

# Rescaling (Theorem 2.1 / Corollaries 2.2, 3.1.1, 3.2.1, 3.3.1, Weiß and Kim 2025)
# `val` is in the logarithm base of the chart and is converted back to the natural
# logarithm first, so the rescaled statistic does not depend on the base (see `log_base`).
function rescale_sop(val, q, chart_choice::Shannon{T}) where {T}
  return (-2 / q) * (val * log_base(chart_choice) - log(q))
end

function rescale_sop(val, q, chart_choice::ShannonExtropy{T}) where {T}
  return (-2) * (1 - 1 / q) * (val * log_base(chart_choice) - (q - 1) * log(q / (q - 1)))
end

function rescale_sop(val, q, ::DistanceToWhiteNoise)
  return val
end




