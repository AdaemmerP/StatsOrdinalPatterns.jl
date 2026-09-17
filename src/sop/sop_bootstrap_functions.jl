export bootstrap_sop, test_sop_bootstrap, SOPTestResultBoot

# ------------------------------------------------------------------------------
# 1. 2D RESAMPLING (shared with sop_bp_bootstrap_functions.jl and sacf_bootstrap_functions.jl)
# ------------------------------------------------------------------------------
function resample_2d!(resampled::Matrix, data::Matrix, block_size::Int)
  M, N = size(data)
  if block_size == 1
    # IID bootstrap: sample each pixel independently from the full image.
    # Linear indexing (one rand call per pixel) is faster than sampling row/col separately.
    n_pixels = length(data)
    for j in 1:N
      for i in 1:M
        resampled[i, j] = data[rand(1:n_pixels)]
      end
    end
  else
    # 2D block bootstrap: tile the image with randomly drawn rectangular blocks.
    # Outer loop over columns, inner over rows — column-major order for Julia.
    j = 1
    while j <= N
      i = 1
      while i <= M
        start_row = rand(1:(M - block_size + 1))
        start_col = rand(1:(N - block_size + 1))
        for dj in 0:(block_size - 1)
          for di in 0:(block_size - 1)
            if i + di <= M && j + dj <= N
              resampled[i+di, j+dj] = data[start_row+di, start_col+dj]
            end
          end
        end
        i += block_size
      end
      j += block_size
    end
  end
end

# ------------------------------------------------------------------------------
# 2. BUFFER MANAGEMENT
# ------------------------------------------------------------------------------
struct SOPBuffer
  lookup_array_sop::Array{Int,4}   # Precomputed 4D lookup array (computed once)
  sop::Vector{Float64}             # 2x2 neighbourhood values
  win::Vector{Int}                 # sortperm result for the neighbourhood
  sop_freq::Vector{Int}            # Absolute SOP frequencies (length 24)
  p_hat::Vector{Float64}           # Relative type frequencies (length 3 or 6)
  index_sop::Vector{Vector{Int}}   # Precomputed type-to-index mapping
  resampled_data::Matrix{Float64}  # Pre-allocated resampled image
end

function SOPBuffer(M::Int, N::Int; refinement=OrdinaryType())
  n_size = _n_sop_types(refinement)
  return SOPBuffer(
    compute_lookup_array_sop(),
    zeros(4),
    zeros(Int, 4),
    zeros(Int, 24),
    zeros(n_size),
    create_index_sop(refinement),
    zeros(M, N)
  )
end

# ------------------------------------------------------------------------------
# 3. STATISTIC CALCULATION
# ------------------------------------------------------------------------------
function stat_sop!(buffer::SOPBuffer, data, d1::Int, d2::Int; chart_choice, refinement=OrdinaryType())
  (; lookup_array_sop, sop, win, sop_freq, p_hat, index_sop) = buffer

  fill!(sop_freq, 0)
  fill!(p_hat, 0)

  m = size(data, 1) - d1
  n = size(data, 2) - d2

  sop_frequencies!(m, n, d1, d2, lookup_array_sop, data, sop, win, sop_freq)
  fill_p_hat!(p_hat, chart_choice, refinement, sop_freq, m, n, index_sop)

  return chart_stat_sop(p_hat, chart_choice)
end

# ------------------------------------------------------------------------------
# 4. BOOTSTRAP WRAPPER
# ------------------------------------------------------------------------------
"""
    bootstrap_sop(
      data::Matrix{<:Real}, n_boot::Int, d1::Int, d2::Int;
      chart_choice=TauTilde(), refinement=OrdinaryType(), block_size::Int=1
    )

Generate a bootstrap distribution of the SOP statistic for a single spatial image.

- `data`: The 2D image (M × N matrix).
- `n_boot`: Number of bootstrap replications.
- `d1::Int`: Row delay.
- `d2::Int`: Column delay.
- `chart_choice`: The chart statistic type (e.g. `TauTilde()`, `KappaTilde()`).
- `refinement`: [`OrdinaryType`](@ref)`()` for classical types, or a `RefinedType` instance.
- `block_size`: Set > 1 to use 2D block bootstrap and preserve spatial dependencies.
"""
function bootstrap_sop(
  data::Matrix{<:Real},
  n_boot::Int,
  d1::Int,
  d2::Int;
  chart_choice=TauTilde(),
  refinement=OrdinaryType(),
  block_size::Int=1
)

  M, N = size(data)
  results = zeros(Float64, n_boot)
  buffer = SOPBuffer(M, N; refinement=refinement)

  for b in 1:n_boot
    resample_2d!(buffer.resampled_data, data, block_size)
    results[b] = stat_sop!(buffer, buffer.resampled_data, d1, d2;
      chart_choice=chart_choice, refinement=refinement)
  end

  return results
end

# ------------------------------------------------------------------------------
# 5. BOOTSTRAP TEST
# ------------------------------------------------------------------------------
"""
    SOPTestResultBoot

Result of the bootstrap test based on spatial ordinal patterns
[`test_sop_bootstrap`](@ref).

Fields:
- `chart`: the chart choice the test was computed for.
- `stat::Float64`: value of the test statistic.
- `boot_crit::Float64`: bootstrap critical value.
- `boot_pval::Float64`: bootstrap p-value.
- `boot_reject::Bool`: whether the null hypothesis is rejected at the chosen level.
- `n_boot::Int`: number of bootstrap replications.
- `boot_dist::Vector{Float64}`: the `n_boot` resampled statistics (bootstrap null
  distribution), on the same scale as `stat` and `boot_crit`. With Makie loaded,
  `plot(res)` draws it as a histogram.
"""
struct SOPTestResultBoot{C}
  chart::C
  stat::Float64
  boot_crit::Float64
  boot_pval::Float64
  boot_reject::Bool
  n_boot::Int
  boot_dist::Vector{Float64}
end

function Base.show(io::IO, r::SOPTestResultBoot)
  println(io, "SOPTestResultBoot")
  println(io, "  Chart:            ", r.chart)
  println(io, "  Statistic:        ", round(r.stat,      digits=4))
  println(io, "  ─────────────────────────────")
  println(io, "  Bootstrap  (n_boot = ", r.n_boot, ")")
  println(io, "    Critical value: ", round(r.boot_crit, digits=4))
  println(io, "    p-value:        ", round(r.boot_pval, digits=4))
  print(io,   "    Reject H₀:      ", r.boot_reject)
end

# Apply rescale_sop for entropy charts so the bootstrap sits on the same scale as
# the asymptotic critical value (qup22 / (m·n)).
# DistanceToWhiteNoise: rescale_sop is the identity → no change.
# Shannon / ShannonExtropy: rescale_sop maps raw entropy/extropy to a non-negative
#   value that is upper-tail under H₁ and has 95th percentile ≈ qup22/(m·n) under H₀.
# Tau / Kappa charts: pass the raw stat unchanged.
function _sop_scale_stat(chart, raw::Float64, q::Int)::Float64
  if chart isa Union{Shannon, ShannonExtropy, DistanceToWhiteNoise}
    return rescale_sop(raw, q, chart)
  else
    return raw
  end
end

# Direction-aware bootstrap critical value (all entropy charts are upper-tail after rescaling)
function _sop_boot_crit(chart, boot::Vector{Float64}, alpha::Float64)::Float64
  if chart isa Union{TauHat, KappaHat, TauTilde, KappaTilde}
    return quantile(abs.(boot), 1.0 - alpha)   # two-sided
  else
    return quantile(boot, 1.0 - alpha)          # upper-tail (all entropy after rescaling)
  end
end

# Direction-aware bootstrap p-value (allocation-free loop)
function _sop_boot_pval(chart, stat::Float64, boot::Vector{Float64})::Float64
  n = length(boot)
  count = 0
  if chart isa Union{TauHat, KappaHat, TauTilde, KappaTilde}
    abs_stat = abs(stat)
    @inbounds for b in boot; count += (abs(b) >= abs_stat); end
  else  # all entropy charts: upper-tail after rescaling
    @inbounds for b in boot; count += (b >= stat); end
  end
  return count / n
end

"""
    test_sop_bootstrap(data, n_boot, d1, d2; chart_choice, refinement, alpha, block_size)

Compute a bootstrap hypothesis test for spatial ordinal patterns and return an
`SOPTestResultBoot` with the bootstrap critical value, p-value, and reject decision.

For Tau/Kappa charts the raw statistic is used. For entropy charts (`Shannon`,
`ShannonExtropy`, `DistanceToWhiteNoise`) the same `rescale_sop` transformation used
by `test_sop` is applied so that the bootstrap critical value is on the same scale as
the asymptotic critical value and the two can be compared directly.

- `data`: the 2D image (M × N matrix).
- `n_boot`: number of bootstrap replications.
- `d1`, `d2`: row and column delays.
- `block_size`: set `> 1` for a 2D block bootstrap that preserves spatial dependencies.
"""
function test_sop_bootstrap(
  data::Matrix{<:Real},
  n_boot::Int,
  d1::Int,
  d2::Int;
  chart_choice = TauTilde(),
  refinement = OrdinaryType(),
  alpha::Float64 = 0.05,
  block_size::Int = 1
)
  M, N    = size(data)
  buffer  = SOPBuffer(M, N; refinement=refinement)
  q       = length(buffer.p_hat)   # 3 (classical) or 6 (refined)

  raw_stat  = stat_sop!(buffer, data, d1, d2; chart_choice=chart_choice, refinement=refinement)
  stat      = _sop_scale_stat(chart_choice, raw_stat, q)

  raw_boot  = bootstrap_sop(data, n_boot, d1, d2;
                chart_choice=chart_choice, refinement=refinement, block_size=block_size)
  boot_dist = [_sop_scale_stat(chart_choice, b, q) for b in raw_boot]

  b_crit   = _sop_boot_crit(chart_choice, boot_dist, alpha)
  b_pval   = _sop_boot_pval(chart_choice, stat, boot_dist)
  b_reject = chart_choice isa Union{TauHat, KappaHat, TauTilde, KappaTilde} ?
               abs(stat) > b_crit : stat > b_crit
  return SOPTestResultBoot(chart_choice, stat, b_crit, b_pval, b_reject, n_boot, boot_dist)
end
