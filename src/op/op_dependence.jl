
"""
    count_uv_op(ts; m::Int=3, d=1)

Count the ordinal patterns of a single time series `ts`, indexed by their Lehmer code.

Returns the tuple `([p_rel], p_count)`, where `p_count` is the vector of the `m!`
pattern counts and `p_rel = p_count / n_patterns` the vector of relative frequencies,
wrapped in a one element vector.

- `ts::Vector{Float64}`: Time series for which the ordinal patterns are counted.
- `m::Int=3`: Length of the ordinal patterns. Default is 3. Minimum is 2, maximum is 4.
- `d::Int=1`: Time delay. Default is 1.

```julia

ts = rand(100)
count_uv_op(ts; m=3, d=1)
```
"""
function count_uv_op(ts; m::Int=3, d=1)

  m! = factorial(m)
  number_of_patterns = length(ts) - (m - 1) * d
  p_count = zeros(Int, m!)
  bin = zeros(Int, m!)
  win = Vector{Int}(undef, m)
  idx_used = zeros(Int, m)

  for i in 1:number_of_patterns

    unit_range = range(i; step=d, length=m)
    x_long = view(ts, unit_range)
    fill!(bin, 0)
    sortperm!(win, x_long)

    index = perm_to_lehm_idx!(win, idx_used)
    fill!(idx_used, 0)
    bin[index] = 1

    @. p_count += bin
  end

  # Return tuple with relative frequencies and counts
  return ([p_count ./ number_of_patterns], p_count)

end

"""
    count_mv_op(tsx, tsy; m::Int=3, d=1)

Count the number of ordinal patterns in bins for two time series `tsx` and `tsy`. The output will be used
to compute the ordinal pattern dependence coefficient by Schnurr and Dehling (2017) <doi:10.1080/01621459.2016.1164706>.

Returns the tuple `(count_x, count_y, count_yrev, count_eq, count_neq, pattern_seq_tsx,
pattern_seq_tsy)`: the pattern counts of `tsx`, of `tsy` and of the reversed patterns of
`tsy`, the counts of time points at which both series show the same pattern
(`count_eq`) and the reversed pattern (`count_neq`), and the sequences of pattern
indices of both series.

- `tsx`: First time series for which the ordinal patterns are counted.
- `tsy`: Second time series for which the ordinal patterns are counted.
- `m::Int=3`: Length of the ordinal patterns. Default is 3. Minimum is 2, maximum is 4.
- `d::Int=1`: Time delay. Default is 1.

```julia
tsx = rand(100)
tsy = rand(100)
count_mv_op(tsx, tsy; m=3, d=1)
```
"""
function count_mv_op(tsx, tsy; m::Int=3, d=1)

  # Assert that time series have the same length
  @assert length(tsx) == length(tsy) "The time series must have the same length"

  # Assert that 2 <= m <= 4
  @assert 2 <= m <= 4 "This function is only implemented for pattern lengths of 2, 3 and 4"

  m! = factorial(m)
  number_of_patterns = length(tsx) - (m - 1) * d

  # Vectors to store counts for op of x, y and reversed y
  count_x = zeros(Int, m!)
  count_y = zeros(Int, m!)
  count_yrev = zeros(Int, m!)

  # Vectors to store counts for equal and non-equal op
  count_eq = zeros(Int, m!)
  count_neq = zeros(Int, m!)

  # Vectors for storing op match
  bin_x = zeros(Int, m!)
  bin_y = zeros(Int, m!)

  # Vectors for storing ordered sequence
  win_x = Vector{Int}(undef, m)
  win_y = Vector{Int}(undef, m)
  idx_used = zeros(Int, m)

  pattern_seq_tsx = Vector{Int}(undef, number_of_patterns)
  pattern_seq_tsy = Vector{Int}(undef, number_of_patterns)

  for i in 1:number_of_patterns
    unit_range = range(i; step=d, length=m)
    seq_x = view(tsx, unit_range)
    seq_y = view(tsy, unit_range)
    fill!(bin_x, 0)
    fill!(bin_y, 0)

    sortperm!(win_x, seq_x)
    sortperm!(win_y, seq_y)

    index_x = perm_to_lehm_idx!(win_x, idx_used)
    fill!(idx_used, 0)
    index_y = perm_to_lehm_idx!(win_y, idx_used)
    fill!(idx_used, 0)

    pattern_seq_tsx[i] = index_x
    pattern_seq_tsy[i] = index_y

    bin_x[index_x] = 1
    bin_y[index_y] = 1

    @. count_x += bin_x
    @. count_y += bin_y
    @. count_eq = count_eq + bin_x * bin_y

    # Reverse tsy to account for negative dependence
    reverse!(win_y)
    fill!(bin_y, 0)

    index_yrev = perm_to_lehm_idx!(win_y, idx_used)
    fill!(idx_used, 0)
    bin_y[index_yrev] = 1

    @. count_yrev += bin_y
    @. count_neq = count_neq + bin_x * bin_y

  end

  # Create return array
  return_array = (count_x, count_y, count_yrev, count_eq, count_neq, pattern_seq_tsx, pattern_seq_tsy)

  return return_array

end


"""
    dependence_op(tsx, tsy; m::Int=3, d=1)

Compute the ordinal pattern dependence coefficient by Schnurr and Dehling (2017) <doi:10.1080/01621459.2016.1164706>.

Returns the tuple `(dependence, pattern_seq_tsx, pattern_seq_tsy)`, where `dependence`
is the standardized ordinal pattern dependence in `[-1, 1]` and the other two entries
are the sequences of pattern indices of both series.

- `tsx`: first time series.
- `tsy`: second time series, of the same length as `tsx`.
- `m::Int=3`: length of the ordinal patterns (2, 3 or 4).
- `d::Int=1`: time delay.
"""
function dependence_op(tsx, tsy; m::Int=3, d=1)

  @assert length(tsx) == length(tsy) "The time series must have the same length"

  results_count = count_mv_op(tsx, tsy; m=m, d=d)

  count_x = results_count[1] # all pattern counts for x
  count_y = results_count[2] # all pattern counts for y
  count_yrev = results_count[3] # all pattern counts for y reversed
  count_eq = results_count[4] # all pattern counts for equal patterns
  count_neq = results_count[5] # all pattern counts for non-equal patterns
  pattern_seq_tsx = results_count[6]
  pattern_seq_tsy = results_count[7]

  # Convert count matrices to relative frequencies
  n = length(pattern_seq_tsx)
  p_x = count_x ./ n
  p_y = count_y ./ n
  p_yrev = count_yrev ./ n

  n_same = sum(count_eq)
  n_neq = sum(count_neq)

  # Same notation as Schnurr & Dehling (2017), p. 707 
  p = n_same / n
  q = sum(p_x .* p_y)
  r = n_neq / n
  s = sum(p_x .* p_yrev)

  α = p - q
  β = r - s

  cor_pos = α / (1 - q)
  cor_neg = β / (1 - s)
  cor_standard = maximum([cor_pos, 0]) - maximum([cor_neg, 0])

  return (cor_standard, pattern_seq_tsx, pattern_seq_tsy)

end

# Kernel function that is used for the changepoint detection
function kernel_change(x)
  return maximum([0, 1 - abs(x)])
end

# Weight function that is used for the changepoint detection
# Based on https://github.com/cran/ordinalpattern/blob/17b24cfe203893c3ceb41e867de8021760fea1e4/R/Pattern.R#L175 
function weightfun(maxdif, x)
  return return ((maxdif - x) / maxdif)
end

"""
    changepoint_op(tsx, tsy; conf_level=0.95, weight=true, bn=log(length(tsx)), m::Int=3, d=1)

Compute the changepoint in dependence between two time series based on Schnurr and Dehling (2017) <doi:10.1080/01621459.2016.1164706>.

Returns the tuple `(Tn_max, changepoint, p_value, conf_iv)`: the maximum of the CUSUM
statistic, the time index at which it is attained (the estimated changepoint), the
asymptotic p-value based on the Kolmogorov distribution, and the interval
`(-q, q)` with `q` the `conf_level` quantile of that distribution.

- `tsx`: First time series for which the ordinal patterns are counted.
- `tsy`: Second time series for which the ordinal patterns are counted.
- `conf_level::Float64=0.95`: Confidence level for the changepoint detection. Default is 0.95.
- `weight::Bool=true`: Whether to use a weight function. Default is true.
- `bn::Float64=log(length(tsx))`: Bandwidth for the kernel function. Default is log(length(tsx)).
- `m::Int=3`: Length of the ordinal patterns. Default is 3. Minimum is 2, maximum is 4.
- `d::Int=1`: Time delay. Default is 1.

```julia
tsx = rand(100)
tsy = rand(100)
changepoint_op(tsx, tsy; conf_level=0.95, weight=true, bn=log(length(tsx)), m=3, d=1)
```
"""
function changepoint_op(tsx, tsy; conf_level=0.95, weight=true, bn=log(length(tsx)), m::Int=3, d=1)

  # Defining standard weight function
  # Based on https://github.com/cran/ordinalpattern/blob/17b24cfe203893c3ceb41e867de8021760fea1e4/R/Pattern.R#L173
  if (weight == true)
    maxdif = floor(m / 2) * (floor(m / 2) + 1) + floor((m - 1) / 2) * (floor((m - 1) / 2) + 1)
  end

  results_count = count_mv_op(tsx, tsy; m=m, d=d)
  pattern_x_index::Vector{Int64} = results_count[6]
  pattern_index_y::Vector{Int64} = results_count[7]

  # Pre-allocate vectors for L1 norm computation
  L1_vec = Vector{Int}(undef, length(pattern_x_index))
  rks_x = Vector{Int}(undef, m)
  rks_y = Vector{Int}(undef, m)
  ix = Vector{Int}(undef, m)   # scratch for competerank!
  x_minus_y = Vector{Int}(undef, m)

  # Loop to compute L1 norm for each pattern using rank vectors (matching R's patternseq)
  for i in eachindex(pattern_x_index)

    unit_range = range(i; step=d, length=m)
    competerank!(rks_x, view(tsx, unit_range), ix)
    competerank!(rks_y, view(tsy, unit_range), ix)

    for j in 1:m
      x_minus_y[j] = abs(rks_x[j] - rks_y[j])
    end

    L1_vec[i] = sum(x_minus_y)

  end

  # Check whether to use weight function
  if weight == true
    obs = broadcast(weightfun, maxdif, L1_vec)
  else
    obs = L1_vec .== 0
  end

  # Calculation of long-run-variance
  n = length(obs)
  acfv = StatsBase.autocov(obs, 0:floor(Int, bn), demean=true)
  sigma = acfv[1] + 2 * sum(acfv[2:floor(Int, bn)])

  # Calculation of Cusum statistic
  Tn = 1 / sqrt(n) * abs.(cumsum(obs .- mean(obs))) ./ sqrt(sigma)
  Tn_max = findmax(Tn)
  changepoint = Tn_max[2]
  Tnmax = Tn_max[1]
  p_value = 1 - cdf(Kolmogorov(), Tnmax)
  conf_iv = (-1, 1) .* quantile(Kolmogorov(), conf_level)

  return (Tnmax, changepoint, p_value, conf_iv)

end
