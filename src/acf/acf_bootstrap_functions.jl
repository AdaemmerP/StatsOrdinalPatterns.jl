export bootstrap_acf, test_acf_bootstrap, ACFTestResultBoot

# ------------------------------------------------------------------------------
# 1. BOOTSTRAP WRAPPER
# ------------------------------------------------------------------------------
"""
    bootstrap_acf(data, n_boot, h; block_size=1)

Compute the bootstrap distribution of the classical autocorrelation statistic at lag
`h` for the time series `data` and return a vector of `n_boot` bootstrap statistics.

- `data::Vector{Float64}`: the time series.
- `n_boot::Int`: number of bootstrap replications.
- `h::Int`: lag.
- `block_size::Int=1`: block length for the resampling. `1` corresponds to an i.i.d.
  bootstrap; values `> 1` yield a moving-block bootstrap that preserves serial dependence.
"""
function bootstrap_acf(
  data::Vector{Float64},
  n_boot::Int,
  h::Int;
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
    results[b] = stat_acf(resampled_data, h)
  end

  return results
end

# ------------------------------------------------------------------------------
# 2. BOOTSTRAP TEST
# ------------------------------------------------------------------------------
"""
    ACFTestResultBoot

Result of the bootstrap test based on the classical autocorrelation function
[`test_acf_bootstrap`](@ref).

Fields:
- `stat::Float64`: value of the test statistic.
- `boot_crit::Float64`: bootstrap critical value.
- `boot_pval::Float64`: bootstrap p-value.
- `boot_reject::Bool`: whether the null hypothesis is rejected at the chosen level.
- `n_boot::Int`: number of bootstrap replications.
- `boot_dist::Vector{Float64}`: the `n_boot` resampled statistics (bootstrap null
  distribution). With Makie loaded, `plot(res)` draws it as a histogram.
"""
struct ACFTestResultBoot
  stat::Float64
  boot_crit::Float64
  boot_pval::Float64
  boot_reject::Bool
  n_boot::Int
  boot_dist::Vector{Float64}
end

function Base.show(io::IO, r::ACFTestResultBoot)
  println(io, "ACFTestResultBoot")
  println(io, "  Statistic:        ", round(r.stat,      digits=4))
  println(io, "  ─────────────────────────────")
  println(io, "  Bootstrap  (n_boot = ", r.n_boot, ")")
  println(io, "    Critical value: ", round(r.boot_crit, digits=4))
  println(io, "    p-value:        ", round(r.boot_pval, digits=4))
  print(io,   "    Reject H₀:      ", r.boot_reject)
end

# Two-sided: reject when |stat| > crit
function _acf_boot_crit(boot::Vector{Float64}, alpha::Float64)::Float64
  return quantile(abs.(boot), 1.0 - alpha)
end

function _acf_boot_pval(stat::Float64, boot::Vector{Float64})::Float64
  n = length(boot)
  abs_stat = abs(stat)
  count = 0
  @inbounds for b in boot; count += (abs(b) >= abs_stat); end
  return count / n
end

"""
    test_acf_bootstrap(data, n_boot, h; alpha=0.05, block_size=1)

Compute a bootstrap hypothesis test for the classical autocorrelation at lag `h` and
return an `ACFTestResultBoot` with the bootstrap critical value, p-value, and reject
decision.

Unlike `test_acf()`, this does not rely on the asymptotic N(0, 1/n) distribution and is
therefore more reliable for short time series or non-Gaussian data.

- `data`: the time series.
- `n_boot`: number of bootstrap replications.
- `h`: lag.
- `alpha`: significance level (default `0.05`).
- `block_size`: set `> 1` for a block bootstrap that preserves serial dependencies.
"""
function test_acf_bootstrap(
  data::Vector{Float64},
  n_boot::Int,
  h::Int;
  alpha::Float64 = 0.05,
  block_size::Int = 1
)
  stat      = stat_acf(data, h)
  boot_dist = bootstrap_acf(data, n_boot, h; block_size=block_size)
  b_crit    = _acf_boot_crit(boot_dist, alpha)
  b_pval    = _acf_boot_pval(stat, boot_dist)
  return ACFTestResultBoot(stat, b_crit, b_pval, abs(stat) > b_crit, n_boot, boot_dist)
end
