export test_op_surrogate, OPTestResultSurrogate

# ------------------------------------------------------------------------------
# SURROGATE TEST (implementation in ext/StatsOrdinalPatternsTimeseriesSurrogatesExt.jl)
# ------------------------------------------------------------------------------
"""
    OPTestResultSurrogate

Result of the surrogate-data ordinal-pattern test [`test_op_surrogate`](@ref).

Fields:
- `chart`: the chart choice the test was computed for.
- `method`: the surrogate method the reference ensemble was generated with.
- `stat::Float64`: value of the test statistic.
- `surr_crit::Float64`: surrogate critical value.
- `surr_pval::Float64`: surrogate p-value.
- `surr_reject::Bool`: whether the null hypothesis is rejected at the chosen level.
- `n_surrogates::Int`: number of surrogate replications.
- `surr_dist::Vector{Float64}`: the `n_surrogates` statistics of the surrogate ensemble
  (surrogate null distribution). With Makie loaded, `plot(res)` draws it as a histogram.
"""
struct OPTestResultSurrogate{C,S}
  chart::C
  method::S
  stat::Float64
  surr_crit::Float64
  surr_pval::Float64
  surr_reject::Bool
  n_surrogates::Int
  surr_dist::Vector{Float64}
end

function Base.show(io::IO, r::OPTestResultSurrogate)
  println(io, "OPTestResultSurrogate")
  println(io, "  Chart:            ", r.chart)
  println(io, "  Statistic:        ", round(r.stat,      digits=4))
  println(io, "  ─────────────────────────────")
  println(io, "  Surrogate  (", r.method, ", n_surrogates = ", r.n_surrogates, ")")
  println(io, "    Critical value: ", round(r.surr_crit, digits=4))
  println(io, "    p-value:        ", round(r.surr_pval, digits=4))
  print(io,   "    Reject H₀:      ", r.surr_reject)
end

"""
    test_op_surrogate(data, method, n_surrogates; chart_choice, m=3, d=1, alpha=0.05, rng)

Compute a surrogate-data hypothesis test for ordinal patterns and return an
`OPTestResultSurrogate` with the surrogate critical value, p-value, and reject decision.

Unlike `test_op()` and `test_op_bootstrap()`, which test against an i.i.d. null, the null
hypothesis here is determined by the surrogate method: e.g. `RandomFourier()` tests against
a stationary linear Gaussian process, and `AAFT()`/`IAAFT()` against a monotonic static
transform of one — making this a nonlinearity test rather than a generic dependence test.

- `data`: the time series.
- `method`: a surrogate method from TimeseriesSurrogates.jl, e.g. `RandomFourier()`,
  `AAFT()`, `IAAFT()`, `RandomShuffle()`.
- `n_surrogates`: number of surrogate replications.
- `chart_choice`: one of `Persistence()`, `UpDownBalance()`, `RotationalAsymmetry()`,
  `UpDownScaling()`, `DistanceToWhiteNoise()`, `Shannon()`, `ShannonExtropy()`.
- `m=3`: length of the ordinal patterns.
- `d=1`: delay between observations of a pattern.
- `alpha`: significance level (default `0.05`).
- `rng`: random number generator used for the surrogate generation.

**Note:** This function is provided as a package extension: it becomes available once
TimeseriesSurrogates.jl is loaded, i.e. after `using TimeseriesSurrogates`.
"""
function test_op_surrogate end
