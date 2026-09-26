export bootstrap_op_bp, test_op_bp_bootstrap, OPBPTestResultBoot

# ------------------------------------------------------------------------------
# 1. BOOTSTRAP WRAPPER
# ------------------------------------------------------------------------------
"""
    bootstrap_op_bp(data, n_boot, w; chart_choice, m=3, ljung_box=false, block_size=1)

Compute the bootstrap distribution of the Box-Pierce type ordinal-pattern statistic for
the time series `data` and return a vector of `n_boot` bootstrap statistics.

- `data::Vector{Float64}`: the time series.
- `n_boot::Int`: number of bootstrap replications.
- `w::Int`: maximal delay; the individual statistics for delays `1:w` are aggregated.
- `chart_choice`: one of `Shannon()`, `ShannonExtropy()`, `DistanceToWhiteNoise()`,
  `UpDownBalance()`, `Persistence()`, `RotationalAsymmetry()`, `UpDownScaling()`.
  For `Shannon` and `ShannonExtropy`, the logarithm base must be larger than 1. The
  statistic does not depend on it.
- `m::Int=3`: length of the ordinal patterns.
- `ljung_box::Bool=false`: if `true`, use Ljung-Box (BL) weights instead of the constant
  Box-Pierce weight.
- `block_size::Int=1`: block length for the resampling. `1` corresponds to an i.i.d.
  bootstrap; values `> 1` yield a moving-block bootstrap that preserves serial dependence.
"""
function bootstrap_op_bp(
  data::Vector{Float64},
  n_boot::Int,
  w::Int;
  chart_choice,
  m::Int = 3,
  ljung_box::Bool = false,
  block_size::Int = 1
)
  n = length(data)
  results = zeros(Float64, n_boot)
  resampled_data = similar(data)

  for b in 1:n_boot
    if block_size == 1
      for i in 1:n
        resampled_data[i] = data[rand(1:n)]
      end
    else
      curr_idx = 1
      while curr_idx <= n
        start_pos = rand(1:(n-block_size+1))
        for j in 0:(block_size-1)
          if curr_idx <= n
            resampled_data[curr_idx] = data[start_pos+j]
            curr_idx += 1
          end
        end
      end
    end
    results[b] = stat_op_bp(
      resampled_data, w; chart_choice=chart_choice, m=m, ljung_box=ljung_box
    )
  end

  return results
end

# ------------------------------------------------------------------------------
# 2. BOOTSTRAP TEST
# ------------------------------------------------------------------------------
"""
    OPBPTestResultBoot

Result of the bootstrap Box-Pierce type ordinal-pattern test
[`test_op_bp_bootstrap`](@ref).

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
struct OPBPTestResultBoot{C}
  chart::C
  stat::Float64
  boot_crit::Float64
  boot_pval::Float64
  boot_reject::Bool
  n_boot::Int
  boot_dist::Vector{Float64}
end

function Base.show(io::IO, r::OPBPTestResultBoot)
  println(io, "OPBPTestResultBoot")
  println(io, "  Chart:            ", r.chart)
  println(io, "  Statistic:        ", round(r.stat,      digits=4))
  println(io, "  ─────────────────────────────")
  println(io, "  Bootstrap  (n_boot = ", r.n_boot, ")")
  println(io, "    Critical value: ", round(r.boot_crit, digits=4))
  println(io, "    p-value:        ", round(r.boot_pval, digits=4))
  print(io,   "    Reject H₀:      ", r.boot_reject)
end

# The BP statistic is upper-tailed for every chart (entropy charts enter as
# maximum − statistic, the remaining charts as squares), so no direction dispatch
# is needed here — unlike the single-delay `test_op_bootstrap`.
function _op_bp_boot_crit(boot::Vector{Float64}, alpha::Float64)::Float64
  return quantile(boot, 1.0 - alpha)
end

function _op_bp_boot_pval(stat::Float64, boot::Vector{Float64})::Float64
  n = length(boot)
  count = 0
  @inbounds for b in boot; count += (b >= stat); end
  return count / n
end

"""
    test_op_bp_bootstrap(data, n_boot, w; chart_choice, m=3, alpha=0.05,
      ljung_box=false, block_size=1)

Compute a bootstrap Box-Pierce type hypothesis test for ordinal patterns and return an
`OPBPTestResultBoot` with the bootstrap critical value, p-value, and reject decision.

Unlike [`test_op_bp`](@ref), this does not rely on tabulated critical values and
therefore works for any chart, any pattern length `m`, any maximal delay `w` and any
significance level `alpha`.

- `data`: the time series.
- `n_boot`: number of bootstrap replications.
- `w`: maximal delay; the individual statistics for delays `1:w` are aggregated.
- `chart_choice`: one of `Shannon()`, `ShannonExtropy()`, `DistanceToWhiteNoise()`,
  `UpDownBalance()`, `Persistence()`, `RotationalAsymmetry()`, `UpDownScaling()`.
  For `Shannon` and `ShannonExtropy`, the logarithm base must be larger than 1. The
  statistic does not depend on it.
- `m::Int=3`: length of the ordinal patterns.
- `alpha`: significance level (default `0.05`).
- `ljung_box::Bool=false`: if `true`, use Ljung-Box (BL) weights.
- `block_size`: set `> 1` for a block bootstrap that preserves serial dependencies.
"""
function test_op_bp_bootstrap(
  data::Vector{Float64},
  n_boot::Int,
  w::Int;
  chart_choice,
  m::Int = 3,
  alpha::Float64 = 0.05,
  ljung_box::Bool = false,
  block_size::Int = 1
)
  stat      = stat_op_bp(data, w; chart_choice=chart_choice, m=m, ljung_box=ljung_box)
  boot_dist = bootstrap_op_bp(
    data, n_boot, w; chart_choice=chart_choice, m=m, ljung_box=ljung_box,
    block_size=block_size
  )
  b_crit = _op_bp_boot_crit(boot_dist, alpha)
  b_pval = _op_bp_boot_pval(stat, boot_dist)
  return OPBPTestResultBoot(chart_choice, stat, b_crit, b_pval, stat > b_crit, n_boot, boot_dist)
end
