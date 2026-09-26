
"""
    DistanceToWhiteNoise()

Chart choice for the Δ-chart. The statistic measures the squared Euclidean distance of the
ordinal-pattern distribution to the uniform distribution obtained under white noise. See
Equation (3) in Weiß and Testik (2023).
"""
struct DistanceToWhiteNoise <: InformationMeasure end

"""
    UpDownBalance()

Chart choice for the β-chart (up-down balance) by Bandt (2019), Equation (3). For pattern
length `m = 3`, the statistic is ``\\hat{\\beta} = \\hat{p}_1 - \\hat{p}_6``, based on the
relative frequencies of the decreasing and increasing patterns.
"""
struct UpDownBalance <: ComplexityEstimator end

"""
    Persistence()

Chart choice for the τ-chart (persistence) by Bandt (2019), Equation (4). For pattern
length `m = 3`, the statistic is ``\\hat{\\tau} = \\hat{p}_1 + \\hat{p}_6 - 1/3``, based on
the relative frequencies of the two monotone patterns.
"""
struct Persistence <: ComplexityEstimator end

"""
    RotationalAsymmetry()

Chart choice for the γ-chart (rotational asymmetry) by Bandt (2019), Equation (5). For
pattern length `m = 3`, the statistic is
``\\hat{\\gamma} = \\hat{p}_3 + \\hat{p}_5 - \\hat{p}_2 - \\hat{p}_4``.
"""
struct RotationalAsymmetry <: ComplexityEstimator end

"""
    UpDownScaling()

Chart choice for the δ-chart (up-down scaling) by Bandt (2019), Equation (6). For pattern
length `m = 3`, the statistic is
``\\hat{\\delta} = \\hat{p}_2 + \\hat{p}_3 - \\hat{p}_4 - \\hat{p}_5``.
"""
struct UpDownScaling <: ComplexityEstimator end

"""
    chart_stat_op(p_vec, chart_choice)

Compute the (in-control) chart statistic of an ordinal-pattern distribution `p_vec` for the
given chart choice.

- `p_vec`: vector of (relative) ordinal-pattern frequencies of length `m!`, ordered by
  their Lehmer index (see [`perm_to_lehm_idx`](@ref)).
- `chart_choice`: one of `Shannon()`, `ShannonExtropy()`, `DistanceToWhiteNoise()`,
  `UpDownBalance()`, `Persistence()`, `RotationalAsymmetry()`, `UpDownScaling()`.

Returns the value of the chart statistic. The Shannon and Shannon-extropy statistics
follow Equation (3) in Weiß and Testik (2023); the β-, τ-, γ- and δ-statistics follow
Equations (3)–(6) in Bandt (2019) and require `m = 3` (`UpDownBalance` also supports
`m = 2`).

The Shannon and Shannon-extropy statistics are computed in the logarithm base of
`chart_choice`, which must be larger than 1. Note that `Shannon()` and `ShannonExtropy()`
default to base 2 in ComplexityMeasures.jl; use `Shannon(base=exp(1))` for the natural
logarithm used in the papers. See the section on the logarithm base in the docs.
"""
function chart_stat_op(p_vec, chart_choice::Shannon) # H-chart: Equation (3), page 342, Weiss and Testik (2023)
  value = 0.0
  for i in axes(p_vec, 1)
    p_vec[i] > 0 && (value -= p_vec[i] * log(p_vec[i])) # to avoid log(0)
  end
  return value / log_base(chart_choice)
end

# Hex-chart: Equation (3), page 342, Weiss and Testik (2023), Equation (15), page 6 in the paper
function chart_stat_op(p_vec, chart_choice::ShannonExtropy)
  value = 0.0
  for i in axes(p_vec, 1)
    p_vec[i] < 1 && (value -= (1 - p_vec[i]) * log(1 - p_vec[i])) # to avoid log of negative value
  end
  return value / log_base(chart_choice)
end

# Natural logarithm of the logarithm base of an entropy chart.
#
# The asymptotic theory of all tests is derived for the natural logarithm. An entropy in
# base b equals the entropy under the natural logarithm divided by log(b), so the
# statistics are computed with the natural logarithm and divided by `log_base` at the end,
# and theoretical quantities are converted the same way. Every Shannon or ShannonExtropy
# computation passes through this function, which makes it the single place where the
# base is validated. Bases in (0, 1) are rejected because log(b) < 0 flips the sign of the
# entropy and thereby the direction of every lower sided test, without raising an error.
function log_base(chart_choice::Union{Shannon,ShannonExtropy})
  base = chart_choice.base
  base > 1 || throw(ArgumentError(
    "the logarithm base of $(chart_choice) must be larger than 1, got base = $base. " *
    "Use for example $(nameof(typeof(chart_choice)))(base=exp(1)) for the natural " *
    "logarithm."
  ))
  return log(base)
end

# All other charts carry no logarithm base.
log_base(_) = 1.0

# Δ-chart: Equation (3), page 342, Weiss and Testik (2023)
function chart_stat_op(p_vec, ::DistanceToWhiteNoise)
  op_length = length(p_vec)
  value = 0.0
  for i in axes(p_vec, 1)
    value += (p_vec[i] - 1 / op_length)^2
  end
  return value
end

# β-chart: Bandt (2019), equation (3)
function chart_stat_op(p_vec, ::UpDownBalance)
  if length(p_vec) == 2
    # β-chart for op_length of 2
    return p_vec[2] - p_vec[1]
  else
    return p_vec[1] - p_vec[6]
  end
end

# τ-chart: Bandt (2019), equation (4)
function chart_stat_op(p_vec, ::Persistence)
  return p_vec[1] + p_vec[6] - (1 / 3)
end

# γ-chart: Bandt (2019), equation (5)
function chart_stat_op(p_vec, ::RotationalAsymmetry)
  return p_vec[3] + p_vec[5] - p_vec[2] - p_vec[4]
end

# δ-chart: Bandt (2019), equation (6)
function chart_stat_op(p_vec, ::UpDownScaling)
  return p_vec[2] + p_vec[3] - p_vec[5] - p_vec[4]
end
