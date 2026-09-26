"""
    stat_op(data; chart_choice, m=3, d=1, add_noise=false)
    stat_op(data, lam; chart_choice, m=3, d=1, add_noise=false)

Compute the ordinal-pattern chart statistic for the time series `data`. The ordinal
patterns are indexed via the Lehmer code (see [`perm_to_lehm_idx!`](@ref)).

The first method computes the statistic once for the whole series and returns the vector
`[stat, p_rel]`, where `stat` is the chart statistic and `p_rel` the vector of relative
ordinal-pattern frequencies.

The second method additionally applies EWMA smoothing with parameter `lam` (Equation (5)
in Weiß and Testik (2023)) and returns the tuple `(stats_all, p_rel)`, where `stats_all`
contains the sequentially computed EWMA chart statistics.

- `data`: time series (vector).
- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `chart_choice`: one of `Shannon()`, `ShannonExtropy()`, `DistanceToWhiteNoise()`,
  `UpDownBalance()`, `Persistence()`, `RotationalAsymmetry()`, `UpDownScaling()`.
  For `Shannon` and `ShannonExtropy`, the statistic is in the logarithm base of the
  chart, which must be larger than 1. Both default to base 2 in ComplexityMeasures.jl;
  use `Shannon(base=exp(1))` for the natural logarithm used in the papers.
- `m::Int=3`: length of the ordinal patterns.
- `d::Int=1`: delay between observations of a pattern.
- `add_noise::Bool=false`: add uniform noise to `data` to break ties (recommended for
  discrete-valued series).
"""
function stat_op(data; chart_choice, m::Int=3, d::Int=1, add_noise::Bool=false) # uses the lehmer code to compute the ordinal pattern

  # pre-allocate
  m_fact = factorial(m)
  p_vec = Vector{Float64}(undef, m_fact)
  p_count = zeros(Int, m_fact)
  fill!(p_vec, 1 / m_fact)
  bin = zeros(Int, m_fact)
  win = zeros(Int, m)
  idx_used = zeros(Int, m)
  number_of_patterns = length(data) - (m - 1) * d

  if add_noise
    data = data .+ rand(length(data))
  end

  # Loop over all possible ordinal patterns
  for i in 1:number_of_patterns #for (i, j) in enumerate(dindex_ranges)

    # Reset binarization vector
    fill!(bin, 0)

    # create unit range for indexing data
    unit_range = range(i; step=d, length=m)
    x_long = view(data, unit_range)

    # compute ordinal pattern based on permutations    
    sortperm!(win, x_long)

    # Convert permutation to lehmer index
    index = perm_to_lehm_idx!(win, idx_used)
    fill!(idx_used, 0) # reset idx_used

    # Binarization of ordinal pattern
    bin[index] = 1

    @. p_count += bin

  end

  p_rel = p_count ./ sum(p_count) #length(dindex_ranges)
  stat = chart_stat_op(p_rel, chart_choice)
  return [stat, p_rel]

end



# Function to compute EWMA chart statistic
function stat_op(
  data, lam; chart_choice, m::Int=3, d::Int=1, add_noise::Bool=false
)

  m_fact = factorial(m)

  p_vec = Vector{Float64}(undef, m_fact)
  p_count = zeros(Int, m_fact)
  fill!(p_vec, 1 / m_fact)
  bin = Vector{Int64}(undef, m_fact)
  win = Vector{Int64}(undef, m)
  idx_used = zeros(Int, m)
  number_of_patterns = length(data) - (m - 1) * d
  stats_all = zeros(Float64, number_of_patterns)

  if add_noise
    data = data .+ rand(length(data))
  end

  for i in 1:number_of_patterns # (i, j) in enumerate(dindex_ranges)

    # Reset binarization vector
    fill!(bin, 0)

    # create unit range for indexing data
    unit_range = range(i; step=d, length=m)

    x_long = view(data, unit_range)

    # compute ordinal pattern based on permutations    
    sortperm!(win, x_long)

    # Convert permutation to lehmer index
    index = perm_to_lehm_idx!(win, idx_used)
    fill!(idx_used, 0)

    # Binarization of ordinal pattern
    bin[index] = 1

    # Compute EWMA statistic for binarized ordinal pattern, Equation (5), page 342, Weiss and Testik (2023)
    @. p_vec = lam * bin + (1 - lam) * p_vec
    @. p_count += bin

    # statistic based on smoothed p-estimate
    stat = chart_stat_op(p_vec, chart_choice)

    # Save temporary test statistic
    stats_all[i] = stat
  end

  p_rel = p_count ./ sum(p_count) #length(dindex_ranges)
  return (stats_all, p_rel) # [stats_all, p_rel]

end
