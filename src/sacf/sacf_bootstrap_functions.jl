export bootstrap_sacf, bootstrap_sacf_bp, test_sacf_bootstrap, test_sacf_bp_bootstrap, SACFTestResultBoot, SACFBPTestResultBoot

# ------------------------------------------------------------------------------
# 1. BUFFER MANAGEMENT
# ------------------------------------------------------------------------------
struct SACFBuffer{T<:Real}
  resampled_data::Matrix{T}
  X_centered::Matrix{T}
end

function SACFBuffer(M::Int, N::Int)
  return SACFBuffer(
    zeros(Float64, M, N),
    zeros(Float64, M, N)
  )
end

# resample_2d! is defined in sop/sop_bootstrap_functions.jl (included earlier)

# ------------------------------------------------------------------------------
# 2. BOOTSTRAP WRAPPERS
# ------------------------------------------------------------------------------
"""
    bootstrap_sacf(
      data::Matrix{T}, n_boot::Int, d1::Int, d2::Int;
      block_size::Int=1
    ) where {T<:Real}

Generate a bootstrap distribution of the SACF statistic for a single spatial image.

- `data`: The 2D image (M × N matrix).
- `n_boot`: Number of bootstrap replications.
- `d1::Int`: Row delay for the SACF.
- `d2::Int`: Column delay for the SACF.
- `block_size`: Set > 1 to use 2D block bootstrap and preserve spatial dependencies.
"""
function bootstrap_sacf(
  data::Matrix{<:Real},
  n_boot::Int,
  d1::Int,
  d2::Int;
  block_size::Int=1
)

  M, N = size(data)
  results = zeros(Float64, n_boot)
  buffer = SACFBuffer(M, N)

  for b in 1:n_boot
    resample_2d!(buffer.resampled_data, data, block_size)
    buffer.X_centered .= buffer.resampled_data .- mean(buffer.resampled_data)
    results[b] = sacf(buffer.X_centered, d1, d2)
  end

  return results
end


"""
    bootstrap_sacf_bp(
      data::Matrix{T}, n_boot::Int, w::Int;
      block_size::Int=1
    ) where {T<:Real}

Generate a bootstrap distribution of the BP-SACF statistic for a single spatial image.

- `data`: The 2D image (M × N matrix).
- `n_boot`: Number of bootstrap replications.
- `w::Int`: Window size for the BP-statistic.
- `block_size`: Set > 1 to use 2D block bootstrap and preserve spatial dependencies.
"""
function bootstrap_sacf_bp(
  data::Matrix{<:Real},
  n_boot::Int,
  w::Int;
  block_size::Int=1
)

  M, N = size(data)
  results = zeros(Float64, n_boot)
  buffer = SACFBuffer(M, N)

  for b in 1:n_boot
    resample_2d!(buffer.resampled_data, data, block_size)
    results[b] = stat_sacf_bp(buffer.resampled_data, w)
  end

  return results
end

# ------------------------------------------------------------------------------
# 3. BOOTSTRAP TEST
# ------------------------------------------------------------------------------
"""
    SACFTestResultBoot

Result of the bootstrap test based on the spatial autocorrelation function
[`test_sacf_bootstrap`](@ref).

Fields:
- `stat::Float64`: value of the test statistic.
- `boot_crit::Float64`: bootstrap critical value.
- `boot_pval::Float64`: bootstrap p-value.
- `boot_reject::Bool`: whether the null hypothesis is rejected at the chosen level.
- `n_boot::Int`: number of bootstrap replications.
- `boot_dist::Vector{Float64}`: the `n_boot` resampled statistics (bootstrap null
  distribution). With Makie loaded, `plot(res)` draws it as a histogram.
"""
struct SACFTestResultBoot
  stat::Float64
  boot_crit::Float64
  boot_pval::Float64
  boot_reject::Bool
  n_boot::Int
  boot_dist::Vector{Float64}
end

function Base.show(io::IO, r::SACFTestResultBoot)
  println(io, "SACFTestResultBoot")
  println(io, "  Statistic:        ", round(r.stat,      digits=4))
  println(io, "  ─────────────────────────────")
  println(io, "  Bootstrap  (n_boot = ", r.n_boot, ")")
  println(io, "    Critical value: ", round(r.boot_crit, digits=4))
  println(io, "    p-value:        ", round(r.boot_pval, digits=4))
  print(io,   "    Reject H₀:      ", r.boot_reject)
end

"""
    SACFBPTestResultBoot

Result of the bootstrap Box-Pierce type test based on the spatial autocorrelation
function [`test_sacf_bp_bootstrap`](@ref).

Fields:
- `stat::Float64`: value of the test statistic.
- `boot_crit::Float64`: bootstrap critical value.
- `boot_pval::Float64`: bootstrap p-value.
- `boot_reject::Bool`: whether the null hypothesis is rejected at the chosen level.
- `n_boot::Int`: number of bootstrap replications.
- `boot_dist::Vector{Float64}`: the `n_boot` resampled statistics (bootstrap null
  distribution). With Makie loaded, `plot(res)` draws it as a histogram.
"""
struct SACFBPTestResultBoot
  stat::Float64
  boot_crit::Float64
  boot_pval::Float64
  boot_reject::Bool
  n_boot::Int
  boot_dist::Vector{Float64}
end

function Base.show(io::IO, r::SACFBPTestResultBoot)
  println(io, "SACFBPTestResultBoot")
  println(io, "  Statistic:        ", round(r.stat,      digits=4))
  println(io, "  ─────────────────────────────")
  println(io, "  Bootstrap  (n_boot = ", r.n_boot, ")")
  println(io, "    Critical value: ", round(r.boot_crit, digits=4))
  println(io, "    p-value:        ", round(r.boot_pval, digits=4))
  print(io,   "    Reject H₀:      ", r.boot_reject)
end

# Two-sided: reject when |stat| > crit
function _sacf_boot_crit(boot::Vector{Float64}, alpha::Float64)::Float64
  return quantile(abs.(boot), 1.0 - alpha)
end

function _sacf_boot_pval(stat::Float64, boot::Vector{Float64})::Float64
  n = length(boot)
  abs_stat = abs(stat)
  count = 0
  @inbounds for b in boot; count += (abs(b) >= abs_stat); end
  return count / n
end

# Upper-tail: reject when stat > crit
function _sacf_bp_boot_crit(boot::Vector{Float64}, alpha::Float64)::Float64
  return quantile(boot, 1.0 - alpha)
end

function _sacf_bp_boot_pval(stat::Float64, boot::Vector{Float64})::Float64
  n = length(boot)
  count = 0
  @inbounds for b in boot; count += (b >= stat); end
  return count / n
end

"""
    test_sacf_bootstrap(data, n_boot, d1, d2; alpha, block_size)

Compute a bootstrap hypothesis test for the SACF at lag (d1, d2) and return a
`SACFTestResultBoot` with the bootstrap critical value, p-value, and reject decision.

Unlike `test_sacf()`, this does not rely on the asymptotic N(0, 1/MN) distribution
and is therefore more reliable for small images.

- `data`: the 2D image (M × N matrix).
- `n_boot`: number of bootstrap replications.
- `d1`, `d2`: row and column delays.
- `alpha`: significance level (default `0.05`).
- `block_size`: set `> 1` for a 2D block bootstrap that preserves spatial dependencies.
"""
function test_sacf_bootstrap(
  data::Matrix{<:Real},
  n_boot::Int,
  d1::Int,
  d2::Int;
  alpha::Float64 = 0.05,
  block_size::Int = 1
)
  stat      = stat_sacf(data, d1, d2)
  boot_dist = bootstrap_sacf(data, n_boot, d1, d2; block_size=block_size)
  b_crit    = _sacf_boot_crit(boot_dist, alpha)
  b_pval    = _sacf_boot_pval(stat, boot_dist)
  return SACFTestResultBoot(stat, b_crit, b_pval, abs(stat) > b_crit, n_boot, boot_dist)
end

"""
    test_sacf_bp_bootstrap(data, n_boot, w; alpha, block_size)

Compute a bootstrap hypothesis test for the BP-SACF statistic with window `w` and
return a `SACFBPTestResultBoot` with the bootstrap critical value, p-value, and
reject decision.

Unlike `test_sacf_bp()`, this does not rely on the asymptotic χ²(2w(w+1)) distribution
and is therefore more reliable for small images or large `w`.

- `data`: the 2D image (M × N matrix).
- `n_boot`: number of bootstrap replications.
- `w`: lag window size (lags up to w in L∞ norm).
- `alpha`: significance level (default `0.05`).
- `block_size`: set `> 1` for a 2D block bootstrap that preserves spatial dependencies.
"""
function test_sacf_bp_bootstrap(
  data::Matrix{<:Real},
  n_boot::Int,
  w::Int;
  alpha::Float64 = 0.05,
  block_size::Int = 1
)
  stat      = stat_sacf_bp(data, w)
  boot_dist = bootstrap_sacf_bp(data, n_boot, w; block_size=block_size)
  b_crit    = _sacf_bp_boot_crit(boot_dist, alpha)
  b_pval    = _sacf_bp_boot_pval(stat, boot_dist)
  return SACFBPTestResultBoot(stat, b_crit, b_pval, stat > b_crit, n_boot, boot_dist)
end
