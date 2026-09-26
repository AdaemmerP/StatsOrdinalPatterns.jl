# The entropy-type SOP statistics have a generalized chi-squared null whose eigenvalues
# are fixed constants — one set per classification scheme (Weiß and Kim 2025). The
# distributions are therefore built once, as on the OP side (see `_gc_op`).
const _GC_SOP_CLASSICAL = _GChisqDist([2 / 5, 16 / 45], ones(2), zeros(2), 0.0, 0.0)
const _GC_SOP_ROTATION = _GChisqDist([1 / 5, 8 / 45, 13 / 90, 2 / 15], ones(4), zeros(4), 0.0, 0.0)
const _GC_SOP_DIRECTION = _GChisqDist([4 / 15, 1 / 5, 8 / 45, 19 / 630], [1, 1, 2, 1], zeros(4), 0.0, 0.0)
const _GC_SOP_DIAGONAL = _GChisqDist([1 / 5, 8 / 45, 19 / 126, 4 / 45], ones(4), zeros(4), 0.0, 0.0)

_gc_sop(::OrdinaryType) = _GC_SOP_CLASSICAL
_gc_sop(::RotationType) = _GC_SOP_ROTATION
_gc_sop(::DirectionType) = _GC_SOP_DIRECTION
_gc_sop(::DiagonalType) = _GC_SOP_DIAGONAL

# Evaluating a quantile of these distributions is not cheap: it is a numerical root-find
# over a numerically integrated cdf, some 340 ms per call. The value depends only on the
# classification scheme and the significance level, so the answers are cached — a size
# study that calls `test_sop` a thousand times at one alpha pays for it once.
#
# The lock guards the *writes*. `get!` inserts whenever a (scheme, alpha) pair is seen for
# the first time, and an unsynchronised insert can corrupt a Dict that another thread is
# reading. Nothing inside the package calls this from a thread — the ARL routines use
# `chart_stat_sop` and a caller-supplied control limit — but a user putting `test_sop` in
# a `Threads.@threads` loop would have every thread arrive at an empty cache together and
# insert at once. The lock is uncontended after that first insert and costs nanoseconds.
const _QUP22_SOP_CACHE = Dict{Tuple{DataType,Float64},Float64}()
const _QUP22_SOP_LOCK = ReentrantLock()

function qup22_sop_value(refinement::SOPClassification, alpha)
  key = (typeof(refinement), Float64(alpha))
  return lock(_QUP22_SOP_LOCK) do
    get!(_QUP22_SOP_CACHE, key) do
      quantile(_gc_sop(refinement), 1 - alpha)
    end
  end
end


# --- 3. Multiple Dispatch Implementation of crit_val_sop() ---

# ==========================================================================
# The dispatch logic is split based on the calculation method:
# 1. Tau/Kappa metrics: normal quantile with the exact finite-sample variance,
#    and theory only for the classical classification (::OrdinaryType).
# 2. Information metrics: generalized chi-squared quantile, one per scheme.
# ==========================================================================

# The public `crit_val_sop` below takes `chart_choice`, `refinement` and `alpha` as
# keywords (package-wide convention); the chart-specific formulas are selected by
# dispatch on the internal `_crit_val_sop` methods.

"""
    crit_val_sop(M, N, d1, d2; chart_choice, refinement=OrdinaryType(), alpha=0.05)

Compute the critical value for the asymptotic test based on spatial ordinal patterns
(SOPs); see [`test_sop`](@ref).

- `M::Int`: number of rows of the data matrix. The SOP matrix has `m = M - d1` rows.
- `N::Int`: number of columns of the data matrix. The SOP matrix has `n = N - d2` columns.
- `d1::Int`: row delay.
- `d2::Int`: column delay.
- `chart_choice`: one of [`TauHat`](@ref)`()`, [`KappaHat`](@ref)`()`,
  [`TauTilde`](@ref)`()`, [`KappaTilde`](@ref)`()`, `Shannon()`, `ShannonExtropy()`,
  `DistanceToWhiteNoise()`.
  The critical value does not depend on the logarithm base of `Shannon` and
  `ShannonExtropy`.
- `refinement`: [`OrdinaryType`](@ref)`()` for the classical SOP classification, or one of
  [`RotationType`](@ref)`()`, [`DirectionType`](@ref)`()`, [`DiagonalType`](@ref)`()`
  (only for the entropy-type charts).
- `alpha=0.05`: significance level.

# Examples
```julia-repl
crit_val_sop(11, 11, 1, 1; chart_choice=TauHat())
```
"""
function crit_val_sop(
  M, N, d1::Int, d2::Int;
  chart_choice,
  refinement::SOPClassification=OrdinaryType(),
  alpha=0.05
)
  return _crit_val_sop(M, N, d1, d2, chart_choice, refinement, alpha)
end

function _crit_val_sop(M, N, d1::Int, d2::Int, ::TauHat, ::OrdinaryType, alpha)
  m = M - d1
  n = N - d2
  correction = 1 - 1 / (2 * m) - 1 / (2 * n)
  term = sqrt(2 / 9 + 1 / 45 * correction) / sqrt(m * n)
  return quantile(Normal(0, 1), 1 - alpha / 2) * term
end

function _crit_val_sop(M, N, d1::Int, d2::Int, ::KappaHat, ::OrdinaryType, alpha)
  m = M - d1
  n = N - d2
  correction = 1 - 1 / (2 * m) - 1 / (2 * n)
  term = sqrt(2 / 3 + 1 / 9 * correction) / sqrt(m * n)
  return quantile(Normal(0, 1), 1 - alpha / 2) * term
end

function _crit_val_sop(M, N, d1::Int, d2::Int, ::TauTilde, ::OrdinaryType, alpha)
  m = M - d1
  n = N - d2
  correction = 1 - 1 / (2 * m) - 1 / (2 * n)
  term = sqrt(2 / 9 + 2 / 45 * correction) / sqrt(m * n)
  return quantile(Normal(0, 1), 1 - alpha / 2) * term
end

function _crit_val_sop(M, N, d1::Int, d2::Int, ::KappaTilde, ::OrdinaryType, alpha)
  m = M - d1
  n = N - d2
  correction = 1 - 1 / (2 * m) - 1 / (2 * n)
  term = sqrt(2 / 3 + 2 / 45 * correction) / sqrt(m * n)
  return quantile(Normal(0, 1), 1 - alpha / 2) * term
end

# A2. Dispatch for Entropy metrics (classical and refined classification alike)
function _crit_val_sop(
  M, N, d1::Int, d2::Int,
  chart_choice::Union{Shannon,ShannonExtropy,DistanceToWhiteNoise},
  refinement::SOPClassification,
  alpha
)

  m = M - d1
  n = N - d2

  return qup22_sop_value(refinement, alpha) / (m * n)
end

# --- Result type for SOP asymptotic test ---

"""
    SOPTestResult

Result of the asymptotic test based on spatial ordinal patterns [`test_sop`](@ref).

Fields:
- `chart`: the chart choice the test was computed for.
- `stat::Float64`: value of the test statistic.
- `asymp_crit::Float64`: asymptotic critical value.
- `asymp_pval::Float64`: asymptotic p-value.
- `asymp_reject::Bool`: whether the null hypothesis is rejected at the chosen level.
"""
struct SOPTestResult{C}
  chart::C
  stat::Float64
  asymp_crit::Float64
  asymp_pval::Float64
  asymp_reject::Bool
end

function Base.show(io::IO, r::SOPTestResult)
  println(io, "SOPTestResult")
  println(io, "  Chart:            ", r.chart)
  println(io, "  Statistic:        ", round(r.stat,       digits=4))
  println(io, "  ─────────────────────────────")
  println(io, "  Asymptotic test")
  println(io, "    Critical value: ", round(r.asymp_crit, digits=4))
  println(io, "    p-value:        ", round(r.asymp_pval, digits=4))
  print(io,   "    Reject H₀:      ", r.asymp_reject)
end

# (the GenChisq nulls used for the entropy-chart p-values are the `_gc_sop` constants
# defined at the top of this file)

# Asymptotic p-value for any chart type.
# For Tau/Kappa (Normal): derive SE from crit_val.
# For entropy charts (GenChisq): m_pat * n_pat * test_stat ~ GenChisq under H₀.
function _sop_asymp_pval(chart_choice, test_stat, crit_val, refinement, alpha, m_pat, n_pat)
  if chart_choice isa Union{TauHat, KappaHat, TauTilde, KappaTilde}
    z_alpha = quantile(Normal(0, 1), 1 - alpha / 2)
    return 2.0 * (1.0 - cdf(Normal(), abs(test_stat) * z_alpha / crit_val))
  else
    return 1.0 - cdf(_gc_sop(refinement), m_pat * n_pat * test_stat)
  end
end

"""
    test_sop(data, d1, d2; chart_choice, refinement=OrdinaryType(), alpha=0.05, add_noise=false)

Perform the asymptotic hypothesis test for spatial dependence based on spatial ordinal
patterns (SOPs) and return a [`SOPTestResult`](@ref) with the test statistic, the
asymptotic critical value, the p-value, and the reject decision.

- `data`: data matrix (spatial field).
- `d1::Int`: row delay.
- `d2::Int`: column delay.
- `chart_choice`: one of [`TauHat`](@ref)`()`, [`KappaHat`](@ref)`()`,
  [`TauTilde`](@ref)`()`, [`KappaTilde`](@ref)`()` (two-sided test), or `Shannon()`,
  `ShannonExtropy()`, `DistanceToWhiteNoise()` (one-sided, upper-tail test with
  rescaled statistic).
  For `Shannon` and `ShannonExtropy`, the logarithm base must be larger than 1. The
  statistic does not depend on it.
- `refinement`: [`OrdinaryType`](@ref)`()` for the classical SOP classification, or one of
  [`RotationType`](@ref)`()`, [`DirectionType`](@ref)`()`, [`DiagonalType`](@ref)`()`
  (only for the entropy-type charts).
- `alpha=0.05`: significance level.
- `add_noise::Bool=false`: add uniform noise to the data to break ties (recommended for
  discrete-valued data).
"""
function test_sop(
  data, d1::Int, d2::Int;
  chart_choice,
  refinement::SOPClassification=OrdinaryType(),
  alpha=0.05,
  add_noise::Bool=false
)
  return _test_sop(data, d1, d2, chart_choice, refinement, alpha, add_noise)
end

# ---- Internal Method 1: Tau/Kappa - two-sided test, no rescaling ----
function _test_sop(
  data, d1::Int, d2::Int,
  chart_choice::Union{TauHat,KappaHat,TauTilde,KappaTilde},
  refinement::OrdinaryType, alpha, add_noise::Bool
)
  M = size(data, 1)
  N = size(data, 2)
  m_pat = M - d1
  n_pat = N - d2
  crit_val  = _crit_val_sop(M, N, d1, d2, chart_choice, refinement, alpha)
  test_stat = stat_sop(data, d1, d2; chart_choice=chart_choice, refinement=refinement, add_noise=add_noise)[1]
  p_val     = _sop_asymp_pval(chart_choice, test_stat, crit_val, refinement, alpha, m_pat, n_pat)
  return SOPTestResult(chart_choice, test_stat, crit_val, p_val, abs(test_stat) > crit_val)
end

# ---- Internal Method 2: Entropy - one-sided (upper-tail) test, rescaling needed ----
function _test_sop(
  data, d1::Int, d2::Int,
  chart_choice::Union{Shannon,ShannonExtropy,DistanceToWhiteNoise},
  refinement::SOPClassification, alpha, add_noise::Bool
)
  M = size(data, 1)
  N = size(data, 2)
  m_pat = M - d1
  n_pat = N - d2
  crit_val  = _crit_val_sop(M, N, d1, d2, chart_choice, refinement, alpha)
  raw       = stat_sop(data, d1, d2; chart_choice=chart_choice, refinement=refinement, add_noise=add_noise)
  # `rescale_sop` alone: it is `m_pat * n_pat` times this quantity that converges to the
  # generalized chi-squared null, and both the critical value (`qup22 / (m * n)`) and the
  # p-value (which multiplies by `m_pat * n_pat` itself) are expressed on the unscaled
  # scale. Multiplying here as well squared the factor and made the test always reject.
  test_stat = rescale_sop(raw[1], length(raw[2]), chart_choice)
  p_val     = _sop_asymp_pval(chart_choice, test_stat, crit_val, refinement, alpha, m_pat, n_pat)
  return SOPTestResult(chart_choice, test_stat, crit_val, p_val, test_stat > crit_val)
end

# ---- Fallback: report the unsupported combination in terms of the public API ----
# The refined SOP classifications only have asymptotic theory for the entropy-type
# charts, so Tau/Kappa + a `RefinedType` has no method above.
function _test_sop(::Any, ::Int, ::Int, chart_choice, refinement, ::Any, ::Bool)
  throw(ArgumentError(
    "test_sop: no asymptotic test available for chart_choice = $(chart_choice) with " *
    "refinement = $(refinement). The refined classifications (RotationType, " *
    "DirectionType, DiagonalType) are only supported for Shannon(), ShannonExtropy() " *
    "and DistanceToWhiteNoise(). Use test_sop_bootstrap instead."
  ))
end
