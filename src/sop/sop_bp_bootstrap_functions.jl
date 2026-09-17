export bootstrap_sop_bp, test_sop_bp_bootstrap, SOPBPTestResultBoot

# ------------------------------------------------------------------------------
# 1. STATISTIC CALCULATION
# ------------------------------------------------------------------------------
# SOPBuffer and resample_2d! are defined in sop_bootstrap_functions.jl (included earlier)
function stat_sop_bp!(buffer::SOPBuffer, data, w::Int; chart_choice, refinement=OrdinaryType())
  (; lookup_array_sop, sop, win, sop_freq, p_hat, index_sop) = buffer

  M_rows = size(data, 1)
  N_cols = size(data, 2)
  bp_stat = 0.0

  for (d1, d2) in Iterators.product(1:w, 1:w)
    fill!(sop_freq, 0)
    fill!(p_hat, 0)
    fill!(win, 0)

    m = M_rows - d1
    n = N_cols - d2

    sop_frequencies!(m, n, d1, d2, lookup_array_sop, data, sop, win, sop_freq)
    fill_p_hat!(p_hat, chart_choice, refinement, sop_freq, m, n, index_sop)
    bp_stat += chart_stat_sop(p_hat, chart_choice)^2
  end

  return bp_stat
end

# ------------------------------------------------------------------------------
# 2. BOOTSTRAP WRAPPER
# ------------------------------------------------------------------------------
"""
    bootstrap_sop_bp(
      data::Matrix{<:Real}, n_boot::Int, w::Int;
      chart_choice=TauTilde(), refinement=OrdinaryType(), block_size::Int=1
    )

Generate a bootstrap distribution of the BP-SOP statistic for a single spatial image.

- `data`: The 2D image (M × N matrix).
- `n_boot`: Number of bootstrap replications.
- `w::Int`: Window size; all (d1, d2) ∈ {1,…,w}² are included in the BP sum.
- `chart_choice`: The chart statistic type (e.g. `TauTilde()`, `KappaTilde()`).
- `refinement`: [`OrdinaryType`](@ref)`()` for classical types, or a `RefinedType` instance.
- `block_size`: Set > 1 to use 2D block bootstrap and preserve spatial dependencies.
"""
function bootstrap_sop_bp(
  data::Matrix{<:Real},
  n_boot::Int,
  w::Int;
  chart_choice=TauTilde(),
  refinement=OrdinaryType(),
  block_size::Int=1
)

  M, N = size(data)
  results = zeros(Float64, n_boot)
  buffer = SOPBuffer(M, N; refinement=refinement)

  for b in 1:n_boot
    resample_2d!(buffer.resampled_data, data, block_size)
    results[b] = stat_sop_bp!(buffer, buffer.resampled_data, w;
      chart_choice=chart_choice, refinement=refinement)
  end

  return results
end

# ------------------------------------------------------------------------------
# 3. BOOTSTRAP TEST
# ------------------------------------------------------------------------------
"""
    SOPBPTestResultBoot

Result of the bootstrap Box-Pierce type test based on spatial ordinal patterns
[`test_sop_bp_bootstrap`](@ref).

Fields:
- `chart`: the chart choice the test was computed for.
- `stat::Float64`: value of the test statistic.
- `boot_crit::Float64`: bootstrap critical value.
- `boot_pval::Float64`: bootstrap p-value.
- `boot_reject::Bool`: whether the null hypothesis is rejected at the chosen level.
- `n_boot::Int`: number of bootstrap replications.
- `boot_dist::Vector{Float64}`: the `n_boot` resampled statistics (bootstrap null
  distribution). With Makie loaded, `plot(res)` draws it as a histogram.
"""
struct SOPBPTestResultBoot{C}
  chart::C
  stat::Float64
  boot_crit::Float64
  boot_pval::Float64
  boot_reject::Bool
  n_boot::Int
  boot_dist::Vector{Float64}
end

function Base.show(io::IO, r::SOPBPTestResultBoot)
  println(io, "SOPBPTestResultBoot")
  println(io, "  Chart:            ", r.chart)
  println(io, "  Statistic:        ", round(r.stat,      digits=4))
  println(io, "  ─────────────────────────────")
  println(io, "  Bootstrap  (n_boot = ", r.n_boot, ")")
  println(io, "    Critical value: ", round(r.boot_crit, digits=4))
  println(io, "    p-value:        ", round(r.boot_pval, digits=4))
  print(io,   "    Reject H₀:      ", r.boot_reject)
end

# The BP-SOP statistic sums the squared chart statistics over all (d1, d2) ∈ {1,…,w}²,
# so it is non-negative and upper-tailed for every chart — no direction dispatch needed.
function _sop_bp_boot_crit(boot::Vector{Float64}, alpha::Float64)::Float64
  return quantile(boot, 1.0 - alpha)
end

function _sop_bp_boot_pval(stat::Float64, boot::Vector{Float64})::Float64
  n = length(boot)
  count = 0
  @inbounds for b in boot; count += (b >= stat); end
  return count / n
end

"""
    test_sop_bp_bootstrap(data, n_boot, w; chart_choice, refinement, alpha, block_size)

Compute a bootstrap Box-Pierce type hypothesis test for spatial ordinal patterns (SOPs),
aggregating the squared chart statistics over all delay combinations
`(d1, d2) ∈ {1,…,w}²`, and return a `SOPBPTestResultBoot` with the bootstrap critical
value, p-value, and reject decision.

No asymptotic theory is available for the BP-SOP statistic, so — unlike the single-delay
[`test_sop`](@ref) — this bootstrap test is the only classical test provided for it.

- `data`: the 2D image (M × N matrix).
- `n_boot`: number of bootstrap replications.
- `w`: window size; all `(d1, d2) ∈ {1,…,w}²` are included in the BP sum.
- `chart_choice`: one of [`TauHat`](@ref)`()`, [`KappaHat`](@ref)`()`,
  [`TauTilde`](@ref)`()`, [`KappaTilde`](@ref)`()`.
- `refinement`: [`OrdinaryType`](@ref)`()` for the classical SOP classification, or a `RefinedType` instance.
- `alpha`: significance level (default `0.05`).
- `block_size`: set `> 1` for a 2D block bootstrap that preserves spatial dependencies.
"""
function test_sop_bp_bootstrap(
  data::Matrix{<:Real},
  n_boot::Int,
  w::Int;
  chart_choice = TauTilde(),
  refinement = OrdinaryType(),
  alpha::Float64 = 0.05,
  block_size::Int = 1
)
  buffer    = SOPBuffer(size(data, 1), size(data, 2); refinement=refinement)
  stat      = stat_sop_bp!(buffer, data, w; chart_choice=chart_choice, refinement=refinement)
  boot_dist = bootstrap_sop_bp(
    data, n_boot, w; chart_choice=chart_choice, refinement=refinement, block_size=block_size
  )
  b_crit = _sop_bp_boot_crit(boot_dist, alpha)
  b_pval = _sop_bp_boot_pval(stat, boot_dist)
  return SOPBPTestResultBoot(chart_choice, stat, b_crit, b_pval, stat > b_crit, n_boot, boot_dist)
end
