module StatsOrdinalPatternsTimeseriesSurrogatesExt

using StatsOrdinalPatterns
using StatsOrdinalPatterns: OPBuffer, stat_op!, _op_boot_crit, _op_boot_pval, reject
using TimeseriesSurrogates: Surrogate, surrogenerator
using Random: AbstractRNG, default_rng

# ------------------------------------------------------------------------------
# 1. SURROGATE REFERENCE ENSEMBLE
# ------------------------------------------------------------------------------
# Generic over the statistic: `statfun` maps a surrogate series to a scalar, so the
# same loop can back test_sop_surrogate(), test_sacf_surrogate() etc. later on.
function _surrogate_ensemble(
  statfun,
  data::Vector{Float64},
  method::Surrogate,
  n_surrogates::Int,
  rng::AbstractRNG
)
  sgen = surrogenerator(data, method, rng)
  results = zeros(Float64, n_surrogates)
  for i in 1:n_surrogates
    results[i] = statfun(sgen())
  end
  return results
end

# ------------------------------------------------------------------------------
# 2. SURROGATE TEST (docstring on the stub in src/op/op_surrogate_functions.jl)
# ------------------------------------------------------------------------------
function StatsOrdinalPatterns.test_op_surrogate(
  data::Vector{Float64},
  method::Surrogate,
  n_surrogates::Int;
  chart_choice,
  m::Int = 3,
  d::Int = 1,
  alpha::Float64 = 0.05,
  rng::AbstractRNG = default_rng()
)
  buffer    = OPBuffer(m)
  stat      = stat_op!(buffer, data; chart_choice=chart_choice, m=m, d=d)
  statfun   = s -> stat_op!(buffer, s; chart_choice=chart_choice, m=m, d=d)
  surr_dist = _surrogate_ensemble(statfun, data, method, n_surrogates, rng)
  s_crit    = _op_boot_crit(chart_choice, surr_dist, alpha)
  s_pval    = _op_boot_pval(chart_choice, stat, surr_dist)
  s_reject  = reject(chart_choice, stat, s_crit)
  return OPTestResultSurrogate(chart_choice, method, stat, s_crit, s_pval, s_reject, n_surrogates, surr_dist)
end

end
