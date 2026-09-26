export test_gop_bootstrap

# ------------------------------------------------------------------------------
# 1. BUFFER MANAGEMENT
# ------------------------------------------------------------------------------
struct GOPBuffer
  p::Vector{Float64}           # EWMA smoothed probability estimate
  p0::Vector{Float64}          # Null distribution probabilities
  p_p0::Vector{Float64}        # Difference p - p0
  bin::Vector{Int}             # Binarization vector
  win::Vector{Int}             # Current window (compete rank result)
  ix::Vector{Int}              # Temporary index vector for competerank!
  lookup_array_gop::Array      # Precomputed lookup array (computed once)
  stats_all::Vector{Float64}   # Pre-allocated per-pattern statistic buffer
end

function GOPBuffer(m, n_patterns::Int, null_dist::DiscreteUnivariateDistribution)
  buf = GOPBuffer(
    zeros(Float64, 13),
    zeros(Float64, 13),
    zeros(Float64, 13),
    zeros(Int, 13),
    zeros(Int, m),
    zeros(Int, m),
    compute_lookup_array_gop(),
    zeros(Float64, n_patterns)
  )
  fill_p0!(buf.p0, null_dist)
  return buf
end

# ------------------------------------------------------------------------------
# 2. STATISTIC CALCULATION
# ------------------------------------------------------------------------------
function stat_gop!(buffer::GOPBuffer, data; chart_choice, lam::Float64, m::Int=3, d::Int=1, reduce=maximum)

  # Unpack buffer for easier access
  (; p, p0, p_p0, bin, win, ix, lookup_array_gop, stats_all) = buffer

  # Reset EWMA to null distribution at start of each resample
  p .= p0
  fill!(bin, 0)

  number_of_patterns = length(data) - (m - 1) * d

  for i in 1:number_of_patterns
    unit_range = range(i; step=d, length=m)
    x_seq = view(data, unit_range)

    competerank!(win, x_seq, ix)

    bin[lookup_array_gop[win[1], win[2], win[3]]] = 1
    @. p = lam * bin + (1 - lam) * p
    @. p_p0 = p - p0

    stats_all[i] = chart_stat_gop(p_p0, chart_choice)

    fill!(bin, 0)
  end

  return reduce(stats_all)
end

# ------------------------------------------------------------------------------
# 3. BOOTSTRAP WRAPPER
# ------------------------------------------------------------------------------
"""
    test_gop_bootstrap(
      data::Vector{Float64}, n_boot::Int,
      null_dist::DiscreteUnivariateDistribution, lam::Float64;
      chart_choice, m::Int=3, d::Int=1,
      block_size::Int=1, reduce=maximum
    )

Generate a bootstrap distribution of the GOP statistic. Each resampled series is
monitored with the EWMA chart, and its sequence of chart statistics is summarized by
`reduce`. Returns the vector of the `n_boot` resampled summaries.

- `data`: The in-control time series.
- `n_boot`: Number of bootstrap replications.
- `null_dist`: The null (in-control) discrete distribution.
- `lam`: EWMA smoothing parameter.
- `chart_choice`: [`D_Chart`](@ref)`()` or `Persistence()`.
- `m::Int=3`: length of the ordinal patterns.
- `d::Int=1`: delay between observations of a pattern.
- `block_size`: Set > 1 to use block bootstrap and preserve time-series dependencies.
- `reduce=maximum`: function that maps the vector of EWMA chart statistics of one
  resampled series to a single number.
"""
function test_gop_bootstrap(
  data::Vector{Float64},
  n_boot::Int,
  null_dist::DiscreteUnivariateDistribution,
  lam::Float64;
  chart_choice,
  m::Int=3,
  d::Int=1,
  block_size::Int=1,
  reduce=maximum
)
  n = length(data)
  n_patterns = n - (m - 1) * d
  results = zeros(Float64, n_boot)
  buffer = GOPBuffer(m, n_patterns, null_dist)
  resampled_data = similar(data)

  for b in 1:n_boot
    if block_size == 1
      for i in 1:n
        resampled_data[i] = data[rand(1:n)]
      end
    else
      curr_idx = 1
      while curr_idx <= n
        start_pos = rand(1:(n - block_size + 1))
        for j in 0:(block_size - 1)
          if curr_idx <= n
            resampled_data[curr_idx] = data[start_pos+j]
            curr_idx += 1
          end
        end
      end
    end

    results[b] = stat_gop!(buffer, resampled_data;
      chart_choice=chart_choice, lam=lam, m=m, d=d, reduce=reduce)
  end

  return results
end
