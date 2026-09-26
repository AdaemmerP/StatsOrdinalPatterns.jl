
"""
    compute_lookup_array_sop()

Compute a 4D array to lookup the index of the sops. The original SOPs are based on ranks. Here we use sortperm which computes the order of the elements in the vector.
"""
function compute_lookup_array_sop()

  p_sops = zeros(Int, 24, 4)
  sort_tmp = zeros(Int, 4)

  for (i, j) in enumerate(collect(permutations(1:4)))
    sortperm!(sort_tmp, j)
    @views p_sops[i, :] = sort_tmp
  end

  # Construct multi-dimensional lookup array 
  lookup_array = zeros(Int, 4, 4, 4, 4)

  for i in axes(p_sops, 1)
    @views lookup_array[p_sops[i, :][1], p_sops[i, :][2], p_sops[i, :][3], p_sops[i, :][4]] = i
  end

  return lookup_array

end



# Number of pattern-type frequencies: 3 for the classical classification, 6 for a
# refined one. Used to size `p_hat` / `p_ewma` consistently with `create_index_sop`.
_n_sop_types(::OrdinaryType) = 3
_n_sop_types(::RefinedType) = 6

"""
    create_index_sop(refinement)

Create and return the index of the sops for sortperm values.
The type frequencies are based on the ranks of the sops, but we use sortperm to
compute the order of the elements in the vector.

`refinement` is a [`SOPClassification`](@ref): [`OrdinaryType`](@ref)`()` for the
classical classification, or one of [`RotationType`](@ref)`()`,
[`DirectionType`](@ref)`()`, [`DiagonalType`](@ref)`()`.
"""
function create_index_sop end

# Classical approach as in Weiss and Kim (2024) and Adämmer et al. (2024)
function create_index_sop(::OrdinaryType)
  s_1 = [1, 3, 8, 11, 14, 17, 22, 24]
  s_2 = [2, 5, 7, 9, 16, 18, 20, 23]
  s_3 = [4, 6, 10, 12, 13, 15, 19, 21]
  return [s_1, s_2, s_3]
end

# RotationType -> Equation (8) in Weiss and Kim (2025)
function create_index_sop(::RotationType)
  s_11 = [1, 11, 14, 24]
  s_12 = [3, 8, 17, 22]
  s_21 = [2, 9, 18, 20]
  s_22 = [5, 7, 16, 23]
  s_31 = [4, 12, 15, 19]
  s_32 = [6, 10, 13, 21]
  return [s_11, s_12, s_21, s_22, s_31, s_32]
end

# "Direction types" -> Equation (9) in Weiss and Kim (2025)
function create_index_sop(::DirectionType)
  s_11 = [1, 8, 17, 24]
  s_12 = [3, 11, 14, 22]
  s_21 = [2, 7, 18, 23]
  s_22 = [5, 9, 16, 20]
  s_31 = [4, 6, 10, 12]
  s_32 = [13, 15, 19, 21]
  return [s_11, s_12, s_21, s_22, s_31, s_32]
end

# "Diagonal types" -> Equation (10) in Weiss and Kim (2025)
function create_index_sop(::DiagonalType)
  s_11 = [1, 3, 22, 24]
  s_12 = [8, 11, 14, 17]
  s_21 = [7, 9, 20, 23]
  s_22 = [2, 5, 16, 18]
  s_31 = [13, 15, 19, 21]
  s_32 = [4, 6, 10, 12]
  return [s_11, s_12, s_21, s_22, s_31, s_32]
end

"""
    compute_p_array(data, d1, d2; chart_choice=TauTilde(), refinement=OrdinaryType(),
      add_noise=false)

Compute the relative SOP type frequencies of every image in an image sequence at lag
`(d1, d2)`. These frequencies are the input of the bootstrap functions
[`arl_sop_bootstrap`](@ref) and [`cl_sop_bootstrap`](@ref).

Returns a matrix with one row per image and one column per SOP type (3 for
[`OrdinaryType`](@ref), 6 for a refined classification).

- `data::Array{T,3}`: image sequence (third dimension = time).
- `d1::Int`: row delay.
- `d2::Int`: column delay.
- `chart_choice=TauTilde()`: chart the frequencies are computed for. It determines how
  the SOP types are grouped; see [`stat_sop`](@ref) for the available charts.
- `refinement=OrdinaryType()`: [`OrdinaryType`](@ref)`()` or one of
  [`RotationType`](@ref)`()`, [`DirectionType`](@ref)`()`, [`DiagonalType`](@ref)`()`.
- `add_noise=false`: add uniform noise to the data to break ties.
"""
function compute_p_array(data::Array{T,3}, d1::Int, d2::Int; chart_choice=TauTilde(), refinement::SOPClassification=OrdinaryType(), add_noise=false) where {T<:Real}

  # pre-allocate
  m = size(data, 1) - d1
  n = size(data, 2) - d2
  lookup_array_sop = compute_lookup_array_sop()
  n_size = _n_sop_types(refinement)
  p_mat = zeros(size(data, 3), n_size)

  # indices for sum of frequencies
  index_sop = create_index_sop(refinement)

  # Add noise?
  if add_noise
    data = data .+ rand(size(data, 1), size(data, 2), size(data, 3))
  end

  # Function to fill p_mat with p_hat values used in parallel computation with Threads.@threads
  function fill_p_mat!(
    i, data_tmp, p_mat, lookup_array_sop, m, n, d1, d2, s_all, chart_choice, refinement
  )

    n_size = _n_sop_types(refinement)
    p_hat = zeros(1, n_size)
    sop = zeros(4)
    sop_freq = zeros(Int, 24)
    win = zeros(Int, 4)

    # Compute frequencies of sops
    sop_frequencies!(m, n, d1, d2, lookup_array_sop, data_tmp, sop, win, sop_freq)

    # Compute sum of frequencies for each group
    fill_p_hat!(p_hat, chart_choice, refinement, sop_freq, m, n, s_all)

    p_mat[i, :] = p_hat

    # Reset win and sop_freq
    fill!(win, 0)
    fill!(sop_freq, 0)
    fill!(p_hat, 0)

  end

  # Fill p_mat in parallel
  Threads.@threads for i in axes(data, 3)

    # Compute frequencies of sops
    @views data_tmp = data[:, :, i]
    fill_p_mat!(
      i, data_tmp, p_mat, lookup_array_sop, m, n, d1, d2, index_sop, chart_choice, refinement
    )

  end

  return p_mat

end


"""
    compute_p_array_bp(data, w; chart_choice, refinement, add_noise=false)

Compute the relative SOP type frequencies of every image in an image sequence for all
delay combinations `(d1, d2) ∈ {1,…,w} × {1,…,w}`. These frequencies are the input of
the bootstrap functions for the BP statistics, [`arl_sop_bp_bootstrap`](@ref) and
[`cl_sop_bp_bootstrap`](@ref).

Returns a 3-dimensional array of size `(number of images, number of SOP types, w²)`.

- `data::Array{T,3}`: image sequence (third dimension = time).
- `w::Int`: maximal delay in both directions.
- `chart_choice`: chart the frequencies are computed for; see [`stat_sop_bp`](@ref).
- `refinement`: [`OrdinaryType`](@ref)`()` or one of [`RotationType`](@ref)`()`,
  [`DirectionType`](@ref)`()`, [`DiagonalType`](@ref)`()`.
- `add_noise=false`: add uniform noise to the data to break ties.
"""
function compute_p_array_bp(data::Array{T,3}, w::Int; chart_choice, refinement::SOPClassification, add_noise=false) where {T<:Real}

  # pre-allocate
  lookup_array_sop = compute_lookup_array_sop()
  d1_d2_combinations = Iterators.product(1:w, 1:w)
  n_size = _n_sop_types(refinement)
  p_array = zeros(size(data, 3), n_size, length(d1_d2_combinations))

  # indices for sum of type frequencies  
  index_sop = create_index_sop(refinement)

  # Add noise?
  if add_noise
    data = data .+ rand(size(data, 1), size(data, 2), size(data, 3))
  end

  # Function to fill 'p_array' with 'p_hat' values. 
  # This function will be called in parallel via Threads.@threads below
  function fill_p_array_bp!(
    i, data_tmp, p_array, d1_d2_combinations, lookup_array_sop, s_all, chart_choice, refinement
  )

    # Initialize thread-local variables
    M_rows = size(data_tmp, 1)
    N_cols = size(data_tmp, 2)
    n_size = _n_sop_types(refinement)
    sop = zeros(4)
    win = zeros(Int, 4)
    sop_freq = zeros(Int, 24)
    p_hat = zeros(1, n_size)

    for (j, (d1, d2)) in enumerate(d1_d2_combinations)

      m = M_rows - d1
      n = N_cols - d2

      # Compute frequencies of sops
      sop_frequencies!(m, n, d1, d2, lookup_array_sop, data_tmp, sop, win, sop_freq)

      # Compute sum of frequencies for each group
      fill_p_hat!(p_hat, chart_choice, refinement, sop_freq, m, n, s_all)

      p_array[i, :, j] = p_hat

      # Reset win and sop_freq
      fill!(sop_freq, 0)
      fill!(p_hat, 0)
    end

  end

  # Fill p_array in parallel
  Threads.@threads for i in axes(data, 3)

    @views fill_p_array_bp!(
      i, data[:, :, i], p_array, d1_d2_combinations, lookup_array_sop, index_sop, chart_choice, refinement
    )

  end

  return p_array

end


"""
    sop_frequencies!(m, n, d1, d2, lookup_array_sop, data, sop, win, sop_freq)

Count the absolute frequencies of the spatial ordinal patterns (SOPs) of the data matrix
`data` in-place.

- `m`, `n`: number of SOP rows (`m = M - d1`) and columns (`n = N - d2`).
- `d1`, `d2`: row and column delays.
- `lookup_array_sop`: lookup array from [`compute_lookup_array_sop`](@ref).
- `data`: data matrix.
- `sop`: work vector of length 4 holding the current 2×2 window values.
- `win`: work vector of length 4 for `sortperm!`.
- `sop_freq`: vector of length 24 that the pattern counts are added to.

Returns `sop_freq`.
"""
function sop_frequencies!(m, n, d1, d2, lookup_array_sop, data, sop, win, sop_freq)

  # Loop through data to fill sop vector
  for j in 1:n
    for i in 1:m

      sop[1] = data[i, j]
      sop[2] = data[i, j+d2]
      sop[3] = data[i+d1, j]
      sop[4] = data[i+d1, j+d2]

      # Order 'sop' in-place and save results in 'win'
      sortperm!(win, sop)

      # Get index
      ind2 = lookup_array_sop[win[1], win[2], win[3], win[4]]

      # Add 1 to index position
      sop_freq[ind2] += 1
    end
  end

  return sop_freq

end


# Fill p_hat with the sum of frequencies and compute relative frequencies.
# Dispatch on chart_choice type (classical, ::OrdinaryType refinement) and on RefinedType.

function fill_p_hat!(p_hat, ::TauHat, ::OrdinaryType, sop_freq, m, n, s_all)
  for i in s_all[1]
    p_hat[1] += sop_freq[i]
  end
  p_hat ./= m * n
end

function fill_p_hat!(p_hat, ::KappaHat, ::OrdinaryType, sop_freq, m, n, s_all)
  for (i, j) in zip(s_all[2], s_all[3])
    p_hat[2] += sop_freq[i]
    p_hat[3] += sop_freq[j]
  end
  p_hat ./= m * n
end

function fill_p_hat!(p_hat, ::TauTilde, ::OrdinaryType, sop_freq, m, n, s_all)
  for i in s_all[3]
    p_hat[3] += sop_freq[i]
  end
  p_hat ./= m * n
end

function fill_p_hat!(p_hat, ::KappaTilde, ::OrdinaryType, sop_freq, m, n, s_all)
  for (i, j) in zip(s_all[1], s_all[2])
    p_hat[1] += sop_freq[i]
    p_hat[2] += sop_freq[j]
  end
  p_hat ./= m * n
end

# Covers Shannon, ShannonExtropy, DistanceToWhiteNoise (all <: InformationMeasure)
function fill_p_hat!(p_hat, ::Union{Shannon,ShannonExtropy,DistanceToWhiteNoise}, ::OrdinaryType, sop_freq, m, n, s_all)
  for (i, j, k) in zip(s_all[1], s_all[2], s_all[3])
    p_hat[1] += sop_freq[i]
    p_hat[2] += sop_freq[j]
    p_hat[3] += sop_freq[k]
  end
  p_hat ./= m * n
end

# Refined computations: same for all chart types
function fill_p_hat!(p_hat, ::Any, ::RefinedType, sop_freq, m, n, s_all)
  for (i1, i2, i3, i4, i5, i6) in zip(s_all[1], s_all[2], s_all[3], s_all[4], s_all[5], s_all[6])
    p_hat[1] += sop_freq[i1]
    p_hat[2] += sop_freq[i2]
    p_hat[3] += sop_freq[i3]
    p_hat[4] += sop_freq[i4]
    p_hat[5] += sop_freq[i5]
    p_hat[6] += sop_freq[i6]
  end
  p_hat ./= m * n
end
