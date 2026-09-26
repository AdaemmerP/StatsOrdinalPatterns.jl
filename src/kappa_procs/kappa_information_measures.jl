
"""
    KappaN()

Chart choice for the κ-chart based on Cohen's kappa for serial dependence in qualitative
(nominal) processes.
"""
struct KappaN <: ComplexityEstimator end

"""
    KappaO()

Chart choice for the κ-chart based on Cohen's kappa for serial dependence in qualitative
(ordinal) processes.
"""
struct KappaO <: ComplexityEstimator end

"""
    KappaN1()

Variant 1 of the nominal κ-chart ([`KappaN`](@ref)); the EWMA recursion smooths both the
marginal probabilities and the agreement statistic.
"""
struct KappaN1 <: ComplexityEstimator end

"""
    KappaN2()

Variant 2 of the nominal κ-chart ([`KappaN`](@ref)); the marginal probabilities are kept
fixed at their in-control values and only the agreement statistic is smoothed.
"""
struct KappaN2 <: ComplexityEstimator end

"""
    KappaO1()

Variant 1 of the ordinal κ-chart ([`KappaO`](@ref)); the EWMA recursion smooths both the
marginal probabilities and the agreement statistic.
"""
struct KappaO1 <: ComplexityEstimator end

"""
    KappaO2()

Variant 2 of the ordinal κ-chart ([`KappaO`](@ref)); the marginal probabilities are kept
fixed at their in-control values and only the agreement statistic is smoothed.
"""
struct KappaO2 <: ComplexityEstimator end

"""
    chart_stat_qual(q, Q, chart_choice)

Compute the κ (Cohen's kappa type) chart statistic for qualitative processes,
``(Q - \\sum_i q_i^2) / \\sum_i q_i (1 - q_i)``.

- `q`: vector of (smoothed) category probabilities.
- `Q`: (smoothed) probability of agreement between consecutive observations.
- `chart_choice`: one of [`KappaN`](@ref)`()`, [`KappaO`](@ref)`()`,
  [`KappaN1`](@ref)`()`, [`KappaN2`](@ref)`()`, [`KappaO1`](@ref)`()`,
  [`KappaO2`](@ref)`()`.
"""
function chart_stat_qual(q, Q, ::Union{KappaN,KappaO,KappaN1,KappaO1,KappaN2,KappaO2})

  # Sum for numerator and denominator part
  numerator_sum = 0.0
  denominator_sum = 0.0
  for i in axes(q, 1)
    numerator_sum += q[i]^2
    denominator_sum += q[i] * (1 - q[i])
  end

  return (Q - numerator_sum) / denominator_sum

end
