# Asymptotic null distribution of the m=3 omnibus statistics (Δ, H, Hex): a weighted sum
# of independent chi-squared variables with the eigenvalues below as weights.
const _gc_op = _GChisqDist(
  [(2 + sqrt(2)) / 12, 2 / 15, 1 / 10, (2 - sqrt(2)) / 12],
  ones(4), zeros(4), 0.0, 0.0
)

# Quantile computation for the generalized chi-squared distribution.
#
# Inverting the generalized chi-squared cdf is numerical: it costs about 2 ms and
# allocates several MB per call. The result depends only on `alpha`, so the handful of
# levels that ever occur are cached. This matters for Monte Carlo studies, where
# `crit_val_op` would otherwise redo the same inversion in every replication. The lock
# keeps the cache correct when tests run on several threads; after the first call per
# level it only guards a dictionary lookup.
const _QUP3_CACHE = Dict{Float64,Float64}()
const _QUP3_LOCK = ReentrantLock()

function qup3_op_value(alpha)
  return lock(_QUP3_LOCK) do
    get!(() -> quantile(_gc_op, 1 - alpha), _QUP3_CACHE, alpha)
  end
end

"""
    crit_val_op(chart_choice, m, n_patterns; alpha=0.05)

Compute the asymptotic critical value of the ordinal-pattern test [`test_op`](@ref).

The critical value depends only on the chart, on the pattern length `m`, on the number of
patterns `n_patterns` and on the level `alpha`, but not on the data. In a Monte Carlo
study it can therefore be computed once per sample size and reused in every replication,
which is much faster than calling [`test_op`](@ref) in the loop: `test_op` additionally
evaluates the p-value, and for `Shannon()`, `ShannonExtropy()` and `DistanceToWhiteNoise()`
that means a numerical evaluation of a generalized chi-squared distribution per call.

- `chart_choice`: one of `Shannon()`, `ShannonExtropy()`, `DistanceToWhiteNoise()`,
  `UpDownBalance()`, `Persistence()`, `RotationalAsymmetry()`, `UpDownScaling()`.
  For `Shannon` and `ShannonExtropy`, the statistic is in the logarithm base of the
  chart, which must be larger than 1. Both default to base 2 in ComplexityMeasures.jl;
  use `Shannon(base=exp(1))` for the natural logarithm used in the papers. Statistic
  and critical value are both in that base, so the test decision and the p-value do
  not depend on it.
- `m::Int`: length of the ordinal patterns. Asymptotic theory is available for `m = 3`
  (and `m = 2` for `Shannon()`, `DistanceToWhiteNoise()` and `UpDownBalance()`).
- `n_patterns::Int`: number of ordinal patterns, `length(ts) - (m - 1) * d`.
- `alpha=0.05`: significance level.

The null hypothesis is rejected when the statistic falls below the critical value for
`Shannon()` and `ShannonExtropy()`, exceeds it for `DistanceToWhiteNoise()`, and exceeds
it in absolute value for the four statistics of Bandt (2019).

```julia
x = randn(500)
n = length(x) - 2
crit = crit_val_op(Persistence(), 3, n)
abs(chart_stat_op(stat_op(x; chart_choice=Persistence())[2], Persistence())) > crit
```
"""
function crit_val_op(chart_choice::Shannon, m, n_patterns; alpha=0.05)
  # The critical values of the entropy charts are derived for the natural logarithm and
  # divided by `log_base` so that they are on the same scale as the statistic.
  if m == 2
    # H-chart (m=2)
    @assert m in (2, 3) "Wrong m value for Shannon chart."
    qup2 = quantile(Chisq(1), 1 - alpha) / 6
    return (log(2) - qup2 / n_patterns) / log_base(chart_choice)
  elseif m == 3
    # H-chart (m=3)
    qup3 = qup3_op_value(alpha)
    return (log(6) - 3 * qup3 / n_patterns) / log_base(chart_choice)
  else
    throw(ArgumentError("Unsupported m value for Shannon chart: $m. Use 2 or 3."))
  end
end

# 2.) Method for ShannonExtropy
function crit_val_op(chart_choice::ShannonExtropy, m, n_patterns; alpha=0.05)
  @assert m == 3 "ShannonExtropy test only supports m = 3."

  # Hex-chart (m=3)
  qup3 = qup3_op_value(alpha)
  return (5 * log(6 / 5) - 3 * qup3 / 5 / n_patterns) / log_base(chart_choice)
end

# 3.) Method for DistanceToWhiteNoise 
function crit_val_op(::DistanceToWhiteNoise, m, n_patterns; alpha=0.05)
  @assert m in (2, 3) "Wrong m value for DistanceToWhiteNoise chart."

  if m == 2
    # Δ-chart (m=2)
    # For m = 2 the only estimated quantity is the relative frequency p̂ of up-steps, and
    # Δ = 2(p̂ - 1/2)². Consecutive comparisons overlap in one observation, so under H₀
    # Var(p̂) = 1/(12·n) (not 1/(4·n)), which gives 6·n·Δ ~ Chisq(1) and hence the factor
    # 6 below. The same factor appears in crit_val_op(::Shannon, 2, …) and in
    # crit_val_op_bp for m = 2.
    qup2 = quantile(Chisq(1), 1 - alpha)
    return qup2 / (6 * n_patterns)
  elseif m == 3
    # Δ-chart (m=3)
    qup3 = qup3_op_value(alpha)
    return qup3 / n_patterns
  end
end

# 4.) Method for UpDownBalance 
function crit_val_op(::UpDownBalance, m, n_patterns; alpha=0.05)
  @assert m in (2, 3) "Wrong m value for UpDownBalance chart."

  z2 = quantile(Normal(0, 1), 1 - alpha / 2)

  # β-chart (The calculation is the same for m=2 and m=3)
  return z2 * sqrt(1 / 3 / n_patterns)
end

# 5.) Method for Persistence
function crit_val_op(::Persistence, m, n_patterns; alpha=0.05)
  @assert m == 3 "Persistence chart only supports m = 3."

  z2 = quantile(Normal(0, 1), 1 - alpha / 2)

  # τ-chart (m=3)
  return z2 * sqrt(8 / 45 / n_patterns)
end

# 6.) Method for RotationalAsymmetry
function crit_val_op(::RotationalAsymmetry, m, n_patterns; alpha=0.05)
  @assert m == 3 "RotationalAsymmetry chart only supports m = 3."

  z2 = quantile(Normal(0, 1), 1 - alpha / 2)

  # γ-chart (m=3)
  return z2 * sqrt(2 / 5 / n_patterns)
end

# 7.) Method for UpDownScaling
function crit_val_op(::UpDownScaling, m, n_patterns; alpha=0.05)
  @assert m == 3 "UpDownScaling chart only supports m = 3."

  z2 = quantile(Normal(0, 1), 1 - alpha / 2)

  # δ-chart (m=3)
  return z2 * sqrt(2 / 3 / n_patterns)
end

# This helper function performs the calculations common to all chart types.
# It takes the specific chart type instance (e.g., ::Shannon) as an argument.
function _common_chart_calculations(ts, chart_type; m::Int=3, d::Int=1, alpha=0.05)

  # Number of patterns when d is integer      
  n_patterns = length(ts) - (m - 1) * d

  # Compute p vectors. We assume stat_op is dispatched on the chart_type.
  p_vec = stat_op(ts; chart_choice=chart_type, m=m, d=d)[2]

  # Compute test statistic and critical value. 
  # Assume chart_stat_op and crit_val_op are also dispatched on the type.
  test_stat = chart_stat_op(p_vec, chart_type)
  crit_val = crit_val_op(chart_type, m, n_patterns; alpha=alpha)

  return (test_stat, crit_val)
end


# --- 3. Result type for asymptotic test ---

"""
    OPTestResult

Result of the asymptotic ordinal-pattern test [`test_op`](@ref).

Fields:
- `chart`: the chart choice the test was computed for.
- `stat::Float64`: value of the test statistic.
- `asymp_crit::Float64`: asymptotic critical value.
- `asymp_pval::Float64`: asymptotic p-value.
- `asymp_reject::Bool`: whether the null hypothesis is rejected at the chosen level.
"""
struct OPTestResult{C}
  chart::C
  stat::Float64
  asymp_crit::Float64
  asymp_pval::Float64
  asymp_reject::Bool
end

function Base.show(io::IO, r::OPTestResult)
  println(io, "OPTestResult")
  println(io, "  Chart:            ", r.chart)
  println(io, "  Statistic:        ", round(r.stat,       digits=4))
  println(io, "  ─────────────────────────────")
  println(io, "  Asymptotic test")
  println(io, "    Critical value: ", round(r.asymp_crit, digits=4))
  println(io, "    p-value:        ", round(r.asymp_pval, digits=4))
  print(io,   "    Reject H₀:      ", r.asymp_reject)
end

# --- 4. Asymptotic p-value computation ---

# The generalized chi-squared null `_gc_op` is defined at the top of this file, next to
# the quantile it is also used for. Unlike the critical value, the p-value depends on the
# data and therefore cannot be cached.
function _asymp_pval(chart, stat::Float64, n_pat::Int, m::Int)::Float64
  if chart isa Union{Persistence, UpDownBalance, RotationalAsymmetry, UpDownScaling}
    se = chart isa Persistence         ? sqrt(8 / 45 / n_pat) :
         chart isa UpDownBalance       ? sqrt(1 / 3  / n_pat) :
         chart isa RotationalAsymmetry ? sqrt(2 / 5  / n_pat) : sqrt(2 / 3 / n_pat)
    return 2.0 * (1.0 - cdf(Normal(), abs(stat) / se))
  elseif chart isa DistanceToWhiteNoise
    # m = 2: 6·n·Δ ~ Chisq(1), see crit_val_op(::DistanceToWhiteNoise, 2, …).
    T = m == 2 ? 6 * n_pat * stat : n_pat * stat
    return m == 2 ? 1.0 - cdf(Chisq(1), T) : 1.0 - cdf(_gc_op, T)
  elseif chart isa Shannon
    # the null distribution is derived for the natural logarithm
    stat_ln = stat * log_base(chart)
    T = m == 2 ? 6 * n_pat * (log(2) - stat_ln) : n_pat * (log(6) - stat_ln) / 3
    return m == 2 ? 1.0 - cdf(Chisq(1), T) : 1.0 - cdf(_gc_op, T)
  else  # ShannonExtropy (m=3 only)
    stat_ln = stat * log_base(chart)
    return 1.0 - cdf(_gc_op, 5 * n_pat * (5 * log(6 / 5) - stat_ln) / 3)
  end
end

# --- 5. Implementation of test_op() ---

# Dispatch on chart type to determine rejection direction
reject(::Union{Shannon,ShannonExtropy}, test_stat, crit_val) = test_stat < crit_val
reject(::DistanceToWhiteNoise, test_stat, crit_val) = test_stat > crit_val
reject(::Union{UpDownBalance,Persistence,RotationalAsymmetry,UpDownScaling}, test_stat, crit_val) = abs(test_stat) > crit_val

"""
    test_op(ts; chart_choice, m=3, d=1, alpha=0.05)

Perform the asymptotic hypothesis test for serial dependence based on ordinal patterns
and return an [`OPTestResult`](@ref) containing the test statistic, the asymptotic
critical value, the p-value, and the reject decision.

- `ts`: the time series.
- `chart_choice`: one of `Shannon()`, `ShannonExtropy()`, `DistanceToWhiteNoise()`,
  `UpDownBalance()`, `Persistence()`, `RotationalAsymmetry()`, `UpDownScaling()`.
  Asymptotic theory is available for `m = 3` (and `m = 2` for `Shannon()`,
  `DistanceToWhiteNoise()`, and `UpDownBalance()`).
  For `Shannon` and `ShannonExtropy`, the statistic is in the logarithm base of the
  chart, which must be larger than 1. Both default to base 2 in ComplexityMeasures.jl;
  use `Shannon(base=exp(1))` for the natural logarithm used in the papers. Statistic
  and critical value are both in that base, so the test decision and the p-value do
  not depend on it.
- `m::Int=3`: length of the ordinal patterns.
- `d::Int=1`: delay between observations of a pattern.
- `alpha=0.05`: significance level.

For pattern lengths without asymptotic theory, use [`test_op_bootstrap`](@ref).
"""
function test_op(ts; chart_choice, m::Int=3, d::Int=1, alpha=0.05)
  n_pat = length(ts) - (m - 1) * d
  test_stat, crit_val = _common_chart_calculations(ts, chart_choice; m=m, d=d, alpha=alpha)
  p_val = _asymp_pval(chart_choice, test_stat, n_pat, m)
  return OPTestResult(chart_choice, test_stat, crit_val, p_val, reject(chart_choice, test_stat, crit_val))
end
