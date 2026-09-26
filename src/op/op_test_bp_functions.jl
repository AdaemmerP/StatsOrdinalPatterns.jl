# --- Result type ---

"""
    OPBPTestResult

Result of the asymptotic Box-Pierce type ordinal-pattern test [`test_op_bp`](@ref).

Fields:
- `chart`: the chart choice the test was computed for.
- `stat::Float64`: value of the test statistic.
- `asymp_crit::Float64`: asymptotic critical value.
- `asymp_pval::Float64`: asymptotic p-value, or `NaN` when the null distribution is not
  available in closed form (see [`test_op_bp`](@ref)).
- `asymp_reject::Bool`: whether the null hypothesis is rejected at the chosen level.
"""
struct OPBPTestResult{C}
  chart::C
  stat::Float64
  asymp_crit::Float64
  asymp_pval::Float64
  asymp_reject::Bool
end

function Base.show(io::IO, r::OPBPTestResult)
  println(io, "OPBPTestResult")
  println(io, "  Chart:            ", r.chart)
  println(io, "  Statistic:        ", round(r.stat,       digits=4))
  println(io, "  ─────────────────────────────")
  println(io, "  Asymptotic test")
  println(io, "    Critical value: ", round(r.asymp_crit, digits=4))
  if isnan(r.asymp_pval)
    println(io, "    p-value:        not available (use test_op_bp_bootstrap)")
  else
    println(io, "    p-value:        ", round(r.asymp_pval, digits=4))
  end
  print(io,   "    Reject H₀:      ", r.asymp_reject)
end

# --- Null distribution of the BP statistic ---

# The BP statistic aggregates the chart statistics over the delays d = 1, …, w and is
# upper-tailed for every chart: the entropy charts enter as (maximum − statistic) and the
# β-, τ-, γ- and δ-charts enter as squares (see `stat_op_bp`).
#
# Under H₀ the null distribution is available in closed form only in the cases below.
# For the remaining m = 3 charts the individual statistics are *correlated across the
# delays* d = 1, …, w, so the null is not a scaled χ²(w); `crit_val_op_bp` therefore falls
# back on values tabulated at α = 0.05 (Weiß, 2022) and no p-value can be reported.
#
# Returns `(dist, scale)` such that `scale * stat` is asymptotically `dist` under H₀, or
# `nothing` when the null distribution is unknown.
function _op_bp_null(chart_choice, m::Int, w::Int)
  if m == 2 && (chart_choice isa Shannon || chart_choice isa DistanceToWhiteNoise)
    # 6 * stat ~ Chisq(w); matches crit_val_op_bp = 1/6 * quantile(Chisq(w), 1 - α)
    return (Chisq(w), 6.0)
  elseif m == 3 && chart_choice isa UpDownBalance
    # The β-statistics are asymptotically independent across delays (Var = 1/3 each),
    # so 3 * stat ~ Chisq(w); matches crit_val_op_bp = 1/3 * quantile(Chisq(w), 1 - α).
    return (Chisq(w), 3.0)
  elseif m == 3 && w == 1
    # A single delay leaves no cross-delay correlation, so the null coincides with the
    # one of the corresponding single-delay test in `op_test_functions.jl`.
    if chart_choice isa Union{Shannon,ShannonExtropy,DistanceToWhiteNoise}
      return (_gc_op, 1.0)
    elseif chart_choice isa Persistence
      return (Chisq(1), 45 / 8)   # Var(τ̂) = 8/45
    elseif chart_choice isa RotationalAsymmetry
      return (Chisq(1), 5 / 2)    # Var(γ̂) = 2/5
    elseif chart_choice isa UpDownScaling
      return (Chisq(1), 3 / 2)    # Var(δ̂) = 2/3
    end
  end
  return nothing
end

# `crit_val_op_bp` only honours `alpha` where it evaluates a χ²(w) quantile; every other
# supported combination returns a value tabulated at α = 0.05.
function _op_bp_alpha_supported(chart_choice, m::Int)
  return (m == 2 && (chart_choice isa Shannon || chart_choice isa DistanceToWhiteNoise)) ||
         (m == 3 && chart_choice isa UpDownBalance)
end

# --- User-facing test function ---

"""
    test_op_bp(data, w; chart_choice, m=3, alpha=0.05, ljung_box=false)

Perform the asymptotic Box-Pierce type test for serial dependence based on ordinal
patterns, aggregating the chart statistics over the delays `d = 1, …, w`, and return an
[`OPBPTestResult`](@ref) with the test statistic, the asymptotic critical value, the
p-value, and the reject decision. The test is upper-tailed for every chart.

- `data`: the time series.
- `w::Int`: maximal delay; the individual statistics for delays `1:w` are aggregated.
- `chart_choice`: one of `Shannon()`, `ShannonExtropy()`, `DistanceToWhiteNoise()`,
  `UpDownBalance()`, `Persistence()`, `RotationalAsymmetry()`, `UpDownScaling()`.
  For `Shannon` and `ShannonExtropy`, the logarithm base must be larger than 1. The
  statistic does not depend on it.
- `m::Int=3`: length of the ordinal patterns.
- `alpha=0.05`: significance level.
- `ljung_box::Bool=false`: if `true`, use Ljung-Box (BL) weights instead of the constant
  Box-Pierce weight. The asymptotic null distribution is unchanged.

# Availability of critical values and p-values

Closed-form null distributions — and hence p-values and arbitrary `alpha` — are available
for `m = 2` with `Shannon()`/`DistanceToWhiteNoise()` and for `m = 3` with
`UpDownBalance()`. For the remaining `m = 3` charts the individual statistics are
correlated across the delays `1:w`, so [`crit_val_op_bp`](@ref) uses values tabulated at
`alpha = 0.05` (Weiß, 2022) for `w ∈ 1:5`; passing a different `alpha` throws an error,
and `asymp_pval` is `NaN` unless `w == 1`. Use [`test_op_bp_bootstrap`](@ref) to obtain a
p-value for any chart, `w` and `alpha`.
"""
function test_op_bp(data, w::Int; chart_choice, m::Int=3, alpha=0.05, ljung_box::Bool=false)

  if !_op_bp_alpha_supported(chart_choice, m) && alpha != 0.05
    throw(ArgumentError(
      "test_op_bp: only alpha = 0.05 is available for $(chart_choice) with m = $m, " *
      "because crit_val_op_bp relies on tabulated critical values. " *
      "Use test_op_bp_bootstrap for other significance levels."
    ))
  end

  crit_val = crit_val_op_bp(w; chart_choice=chart_choice, m=m, alpha=alpha)

  if crit_val === nothing
    throw(ArgumentError(
      "test_op_bp: no critical value available for $(chart_choice) with m = $m and " *
      "w = $w. Tabulated values cover w ∈ 1:5 for m = 3. " *
      "Use test_op_bp_bootstrap instead."
    ))
  end

  test_stat = stat_op_bp(data, w; chart_choice=chart_choice, m=m, ljung_box=ljung_box)

  null = _op_bp_null(chart_choice, m, w)
  p_val = null === nothing ? NaN : 1.0 - cdf(null[1], null[2] * test_stat)

  return OPBPTestResult(chart_choice, test_stat, crit_val, p_val, test_stat > crit_val)
end
