# SACF-BP-statistic for one images
"""
    stat_sacf_bp(data, w)
    stat_sacf_bp(data, lam, w)

Compute the Box-Pierce (BP) type statistic of the spatial autocorrelation function
(SACF), which aggregates the squared autocorrelations over all lags up to `w`.

The statistic is 2 times the sum of the squared spatial autocorrelations over the
`2w(w + 1)` lag pairs `(h1, h2)` with `max(|h1|, |h2|) ≤ w` in one half plane, namely
`1 ≤ h1 ≤ w, 0 ≤ h2 ≤ w` and `-w ≤ h1 ≤ 0, 1 ≤ h2 ≤ w`.

The first method takes a single image (matrix) and returns the statistic as a
`Float64`. The second method takes a sequence of images (3-dimensional array, third
dimension = time), smooths every autocorrelation with an EWMA with parameter `lam` and
returns the vector of sequentially computed BP statistics, one per image.

- `data`: data matrix (single image) or 3-dimensional array (image sequence).
- `lam`: smoothing parameter of the EWMA statistic (second method only).
- `w::Int`: maximal lag.

The critical value is provided by [`crit_val_sacf_bp`](@ref).
"""
function stat_sacf_bp(data::Union{SubArray,Matrix{<:Real}}, w::Int)

  # Compute all relevant h1-h2 combinations
  set_1 = Iterators.product(1:w, 0:w)
  set_2 = Iterators.product(-w:0, 1:w)
  h1_h2_combinations = Iterators.flatten(Iterators.zip(set_1, set_2))

  # pre-allocate
  X_centered = data .- mean(data)
  bp_stat = 0.0

  for (h1, h2) in h1_h2_combinations
    bp_stat += 2 * sacf(X_centered, h1, h2)^2
  end

  return bp_stat

end

# EWMA SACF-BP-statistic for multiple images (documented together with the method above)
function stat_sacf_bp(data::Array{T,3}, lam, w::Int) where {T<:Real}

  # Compute all relevant h1-h2 combinations
  set_1 = Iterators.product(1:w, 0:w)
  set_2 = Iterators.product(-w:0, 1:w)
  h1_h2_combinations = Iterators.flatten(Iterators.zip(set_1, set_2))

  # pre-allocate
  X_centered = zeros(size(data[:, :, 1]))
  rho_hat_all = zeros(length(h1_h2_combinations))
  bp_stats = zeros(size(data, 3))

  # compute sequential BP-statistic
  for i in axes(data, 3)

    # Center the data
    X_centered .= view(data, :, :, i) .- mean(view(data, :, :, i))

    # Compute the BP-statistic       
    bp_stat = 0.0  # Initialize BP-sum
    for (j, (h1, h2)) in enumerate(h1_h2_combinations)
      rho_hat_all[j] = (1 - lam) * rho_hat_all[j] + lam * sacf(X_centered, h1, h2)
      bp_stat += 2 * rho_hat_all[j]^2
    end

    bp_stats[i] = bp_stat

  end

  return bp_stats

end

