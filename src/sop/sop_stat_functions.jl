#===============================================

Multiple Dispatch for 'stat_sop()':
  1. data is only one picture -> data::Matrix{T}
  2. data is a three dimensional array -> data::Array{T, 3}

================================================#

# 1. Method to compute test statistic for one picture without refinement
"""
    stat_sop(data, d1, d2; chart_choice, refinement=OrdinaryType(), add_noise=false,
      noise_dist=Uniform(0, 1))

Compute the test statistic based on spatial ordinal patterns (SOPs) for a single picture
(matrix). Returns the tuple `(stat, p_hat)` with the chart statistic and the vector of
relative SOP type frequencies.

- `data::Union{SubArray,Matrix{<:Real}}`: A 2D array of data.
- `d1::Int`: The delay value for the rows.
- `d2::Int`: The delay value for the columns.
- `chart_choice`: one of [`TauHat`](@ref)`()`, [`KappaHat`](@ref)`()`,
  [`TauTilde`](@ref)`()`, [`KappaTilde`](@ref)`()`, `Shannon()`, `ShannonExtropy()`,
  `DistanceToWhiteNoise()`.
- `refinement`: [`OrdinaryType`](@ref)`()` for the classical SOP classification, or one of
  [`RotationType`](@ref)`()`, [`DirectionType`](@ref)`()`, [`DiagonalType`](@ref)`()`.
- `add_noise::Bool`: A boolean value to add noise to the data.
- `noise_dist::UnivariateDistribution`: The distribution for the noise.

# Examples
```julia-repl
data = rand(20, 20);

stat_sop(data, 1, 1; chart_choice=TauTilde())
```
"""
function stat_sop(
  data::Union{SubArray,Matrix{<:Real}},
  d1::Int, d2::Int;
  chart_choice,
  refinement::SOPClassification=OrdinaryType(),
  add_noise::Bool=false,
  noise_dist::UnivariateDistribution=Uniform(0, 1)
)

  # TODO Check input parameters


  # Pre-allocate
  p_hat = zeros(_n_sop_types(refinement))

  lookup_array_sop = compute_lookup_array_sop()
  sop = zeros(4)
  win = zeros(Int, 4)
  sop_freq = zeros(Int, 24) # factorial(4)  

  # Compute m and n based on data
  m = size(data, 1) - d1
  n = size(data, 2) - d2

  # indices for sum of frequencies
  index_sop = create_index_sop(refinement)

  # Add noise?
  if add_noise
    data = data .+ rand(noise_dist, size(data, 1), size(data, 2))
  end

  # Compute frequencies of sops    
  sop_frequencies!(m, n, d1, d2, lookup_array_sop, data, sop, win, sop_freq)

  # Fill 'p_hat' with sop-frequencies and compute relative frequencies
  fill_p_hat!(p_hat, chart_choice, refinement, sop_freq, m, n, index_sop)

  # Compute test statistic
  stat = chart_stat_sop(p_hat, chart_choice)

  return (stat, p_hat)
end

# 2. Method to compute test statistic for multiple pictures
"""
    stat_sop(data, lam, d1, d2; chart_choice=TauTilde(), refinement=OrdinaryType(),
      add_noise=false, noise_dist=Uniform(0, 1), type_freq_init=nothing)

Compute the sequence of EWMA-smoothed test statistics based on spatial ordinal patterns
(SOPs) for a 3D array of data (image sequence, third dimension = time).

- `data::Array{Float64,3}`: A 3D array of data.
- `lam::Float64`: The lambda value for the EWMA.
- `d1::Int`: The delay value for the rows.
- `d2::Int`: The delay value for the columns.
- `chart_choice`: one of [`TauHat`](@ref)`()`, [`KappaHat`](@ref)`()`,
  [`TauTilde`](@ref)`()`, [`KappaTilde`](@ref)`()`.
- `refinement`: [`OrdinaryType`](@ref)`()` for the classical SOP classification, or one of
  [`RotationType`](@ref)`()`, [`DirectionType`](@ref)`()`, [`DiagonalType`](@ref)`()`.
- `add_noise::Bool`: A boolean value to add noise to the data.
- `noise_dist::UnivariateDistribution`: The distribution for the noise.
- `type_freq_init`: The initial type frequencies. Defaults to the uniform value `1/q`, where
  `q` is the number of SOP types of the classification (3 or 6).
"""
function stat_sop(
  data::Array{T,3},
  lam,
  d1::Int,
  d2::Int;
  chart_choice=TauTilde(),
  refinement::SOPClassification=OrdinaryType(),
  add_noise::Bool=false,
  noise_dist::UnivariateDistribution=Uniform(0, 1),
  type_freq_init=nothing
) where {T<:Real}

  # TODO Check input parameters


  # Compute lookup cube
  lookup_array_sop = compute_lookup_array_sop()

  # Pre-allocate. The EWMA vector has one entry per SOP type: 3 for the classical
  # classification, 6 for the refined ones. It starts at the uniform in-control value
  # unless the caller supplies its own start.
  n_size = _n_sop_types(refinement)
  p_hat = zeros(n_size)
  p_ewma = zeros(n_size)
  p_ewma .= isnothing(type_freq_init) ? 1 / n_size : type_freq_init

  sop = zeros(4)
  stats_all = zeros(size(data, 3))
  sop_freq = zeros(Int, 24) # factorial(4)
  win = zeros(Int, 4)

  # indices for sum of frequencies
  index_sop = create_index_sop(refinement)

  # Compute m and n based on data
  data_tmp = similar(data[:, :, 1])
  rand_tmp = similar(data_tmp)
  m = size(data, 1) - d1
  n = size(data, 2) - d2

  for i = axes(data, 3)

    # add noise?
    if add_noise
      data_tmp .= view(data, :, :, i) .+ rand!(noise_dist, rand_tmp)
    else
      data_tmp .= view(data, :, :, i)
    end

    # Compute frequencies of sops    
    sop_frequencies!(m, n, d1, d2, lookup_array_sop, data_tmp, sop, win, sop_freq)

    # Fill 'p_hat' with sop-frequencies and compute relative frequencies
    fill_p_hat!(p_hat, chart_choice, refinement, sop_freq, m, n, index_sop)

    # Compute test statistic
    @. p_ewma = (1 - lam) * p_ewma + lam * p_hat

    stat_tmp = chart_stat_sop(p_ewma, chart_choice)

    # Save temporary test statistic
    stats_all[i] = stat_tmp

    # Reset win and sop_freq
    fill!(win, 0)
    fill!(sop_freq, 0)
    fill!(p_hat, 0)
  end

  return stats_all

end
