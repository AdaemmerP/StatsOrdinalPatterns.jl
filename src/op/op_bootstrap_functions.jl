export bootstrap_op, test_op_bootstrap, OPTestResultBoot

# ------------------------------------------------------------------------------
# 1. BUFFER MANAGEMENT
# ------------------------------------------------------------------------------
struct OPBuffer
  p_count::Vector{Int}      # Counts for each of the m! permutations
  bin::Vector{Int}          # Temporary bin for internal calculations
  win::Vector{Int}          # Stores the current sliding window indices
  idx_used::Vector{Int}     # Tracking vector for Lehmer Index calculation
  p_rel::Vector{Float64}    # The resulting probability distribution
end

function OPBuffer(m)
  m_fact = factorial(m)
  return OPBuffer(
    zeros(Int, m_fact),
    zeros(Int, m_fact),
    zeros(Int, m),
    zeros(Int, m),
    zeros(Float64, m_fact)
  )
end

# ------------------------------------------------------------------------------
# 2. STATISTIC CALCULATION
# ------------------------------------------------------------------------------
function stat_op!(buffer::OPBuffer, data; chart_choice, m::Int=3, d::Int=1)

  (; p_count, win, idx_used, p_rel) = buffer

  fill!(p_count, 0)
  number_of_patterns = length(data) - (m - 1) * d

  for i in 1:number_of_patterns
    unit_range = i:d:(i+(m-1)*d)
    x_long = view(data, unit_range)
    sortperm!(win, x_long)
    fill!(idx_used, 0)
    index = perm_to_lehm_idx!(win, idx_used)
    p_count[index] += 1
  end

  s = sum(p_count)
  @. p_rel = p_count / s
  return chart_stat_op(p_rel, chart_choice)
end

# ------------------------------------------------------------------------------
# 3. BOOTSTRAP WRAPPER
# ------------------------------------------------------------------------------
"""
    bootstrap_op(data, n_boot; chart_choice, m=3, d=1, block_size=1)

Compute the bootstrap distribution of the ordinal-pattern chart statistic for the time
series `data` and return a vector of `n_boot` bootstrap statistics.

- `data::Vector{Float64}`: the time series.
- `n_boot::Int`: number of bootstrap replications.
- `chart_choice`: one of `Shannon()`, `ShannonExtropy()`, `DistanceToWhiteNoise()`,
  `UpDownBalance()`, `Persistence()`, `RotationalAsymmetry()`, `UpDownScaling()`.
- `m=3`: length of the ordinal patterns.
- `d=1`: delay between observations of a pattern.
- `block_size::Int=1`: block length for the resampling. `1` corresponds to an i.i.d.
  bootstrap; values `> 1` yield a moving-block bootstrap that preserves serial dependence.
"""
function bootstrap_op(
  data::Vector{Float64},
  n_boot::Int;
  chart_choice,
  m=3,
  d=1,
  block_size::Int=1
)
  n = length(data)
  results = zeros(Float64, n_boot)
  buffer = OPBuffer(m)
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
    results[b] = stat_op!(buffer, resampled_data; chart_choice=chart_choice, m=m, d=d)
  end

  return results
end

# ------------------------------------------------------------------------------
# 4. BOOTSTRAP TEST
# ------------------------------------------------------------------------------
"""
    OPTestResultBoot

Result of the bootstrap ordinal-pattern test [`test_op_bootstrap`](@ref).

Fields:
- `chart`: the chart choice the test was computed for.
- `stat::Float64`: value of the test statistic.
- `boot_crit::Float64`: bootstrap critical value.
- `boot_pval::Float64`: bootstrap p-value.
- `boot_reject::Bool`: whether the null hypothesis is rejected at the chosen level.
- `n_boot::Int`: number of bootstrap replications.
- `boot_dist::Vector{Float64}`: the `n_boot` resampled statistics, i.e. the bootstrap
  null distribution the critical value and the p-value were computed from.

With Makie loaded, `plot(res)` draws a histogram of `boot_dist` with the rejection region,
the critical value and the observed statistic; see the plotting tutorial in the docs.
"""
struct OPTestResultBoot{C}
  chart::C
  stat::Float64
  boot_crit::Float64
  boot_pval::Float64
  boot_reject::Bool
  n_boot::Int
  boot_dist::Vector{Float64}
end

function Base.show(io::IO, r::OPTestResultBoot)
  println(io, "OPTestResultBoot")
  println(io, "  Chart:            ", r.chart)
  println(io, "  Statistic:        ", round(r.stat,      digits=4))
  println(io, "  ─────────────────────────────")
  println(io, "  Bootstrap  (n_boot = ", r.n_boot, ")")
  println(io, "    Critical value: ", round(r.boot_crit, digits=4))
  println(io, "    p-value:        ", round(r.boot_pval, digits=4))
  print(io,   "    Reject H₀:      ", r.boot_reject)
end

# Direction-aware bootstrap critical value
function _op_boot_crit(chart, boot::Vector{Float64}, alpha::Float64)::Float64
  if chart isa Union{UpDownBalance, Persistence, RotationalAsymmetry, UpDownScaling}
    return quantile(abs.(boot), 1.0 - alpha)
  elseif chart isa DistanceToWhiteNoise
    return quantile(boot, 1.0 - alpha)
  else  # Shannon, ShannonExtropy
    return quantile(boot, alpha)
  end
end

# Direction-aware bootstrap p-value (allocation-free loop)
function _op_boot_pval(chart, stat::Float64, boot::Vector{Float64})::Float64
  n = length(boot)
  count = 0
  if chart isa Union{UpDownBalance, Persistence, RotationalAsymmetry, UpDownScaling}
    abs_stat = abs(stat)
    @inbounds for b in boot; count += (abs(b) >= abs_stat); end
  elseif chart isa DistanceToWhiteNoise
    @inbounds for b in boot; count += (b >= stat); end
  else
    @inbounds for b in boot; count += (b <= stat); end
  end
  return count / n
end

"""
    test_op_bootstrap(data, n_boot; chart_choice, m=3, d=1, alpha=0.05, block_size=1)

Compute a bootstrap hypothesis test for ordinal patterns and return an `OPTestResultBoot`
with the bootstrap critical value, p-value, and reject decision.

Unlike `test_op()`, this function does not rely on asymptotic distributions and therefore
works for any pattern length `m`, including `m > 3` where no asymptotic theory is available.

- `data`: the time series.
- `n_boot`: number of bootstrap replications.
- `chart_choice`: one of `Persistence()`, `UpDownBalance()`, `RotationalAsymmetry()`,
  `UpDownScaling()`, `DistanceToWhiteNoise()`, `Shannon()`, `ShannonExtropy()`.
- `alpha`: significance level (default `0.05`).
- `block_size`: set `> 1` for a block bootstrap that preserves serial dependencies.
"""
function test_op_bootstrap(
  data::Vector{Float64},
  n_boot::Int;
  chart_choice,
  m::Int = 3,
  d::Int = 1,
  alpha::Float64 = 0.05,
  block_size::Int = 1
)
  buffer    = OPBuffer(m)
  stat      = stat_op!(buffer, data; chart_choice=chart_choice, m=m, d=d)
  boot_dist = bootstrap_op(data, n_boot; chart_choice=chart_choice, m=m, d=d, block_size=block_size)
  b_crit    = _op_boot_crit(chart_choice, boot_dist, alpha)
  b_pval    = _op_boot_pval(chart_choice, stat, boot_dist)
  b_reject  = reject(chart_choice, stat, b_crit)
  return OPTestResultBoot(chart_choice, stat, b_crit, b_pval, b_reject, n_boot, boot_dist)
end
