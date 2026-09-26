"""
    sacf(X_centered, d1::Int, d2::Int)

Compute the spatial autocorrelation of a centered data matrix at lag `(d1, d2)`.
Negative lags are allowed. Returns the autocorrelation as a `Float64`, or `1.0` if all
entries of `X_centered` are equal.

- `X_centered`: the centered (de-meaned) data matrix.
- `d1::Int`: row lag.
- `d2::Int`: column lag.

Use [`stat_sacf`](@ref) to compute the statistic directly from uncentered data.
"""
function sacf(X_centered, d1::Int, d2::Int)

  M = size(X_centered, 1)
  N = size(X_centered, 2)

  # Lag 0x0
  @views cov_00 = dot(X_centered[1:M, 1:N], X_centered[1:M, 1:N]) #/ (M * N)

  # Lag d1xd2
  if d1 >= 0
    if d2 >= 0
      @views cov_d1d2 = dot(X_centered[1:(M-d1), 1:(N-d2)], X_centered[(1+d1):M, (1+d2):N]) #/ (M * N)
    else
      @views cov_d1d2 = dot(X_centered[1:(M-d1), (1+abs(d2)):N], X_centered[(1+d1):M, 1:(N-abs(d2))]) #/ (M * N)
    end
  else
    if d2 >= 0
      @views cov_d1d2 = dot(X_centered[(1+abs(d1)):M, 1:(N-d2)], X_centered[1:(M-abs(d1)), (1+d2):N]) #/ (M * N)
    else
      @views cov_d1d2 = dot(X_centered[(1+abs(d1)):M, (1+abs(d2)):N], X_centered[1:(M-abs(d1)), 1:(N-abs(d2))]) #/ (M * N)
    end
  end

  # Return the SACF value
  if allequal(X_centered)
    return 1.0
  else
    return cov_d1d2 / cov_00
  end
end

# Compute SACF for one picture
"""
    stat_sacf(data, d1, d2)
    stat_sacf(data, lam, d1, d2)

Compute the spatial autocorrelation function (SACF) statistic at lag `(d1, d2)`.

The first method takes a single image (matrix), centers it and returns the spatial
autocorrelation as a `Float64`.

The second method takes a sequence of images (3-dimensional array, third dimension =
time), applies EWMA smoothing with parameter `lam` and returns the vector of
sequentially computed EWMA statistics, one per image.

- `data`: data matrix (single image) or 3-dimensional array (image sequence).
- `lam`: smoothing parameter of the EWMA statistic (second method only).
- `d1::Int`: row delay.
- `d2::Int`: column delay.

The asymptotic critical value is provided by [`crit_val_sacf`](@ref).
"""
function stat_sacf(data::Union{SubArray,Matrix{<:Real}}, d1::Int, d2::Int)

  # pre-allocate
  X_centered = data .- mean(data)

  return sacf(X_centered, d1, d2)

end

# Compute SACF for multiple images (documented together with the method above)
function stat_sacf(data::Array{T,3}, lam, d1::Int, d2::Int) where {T<:Real}

  # pre-allocate
  data_tmp = similar(data[:, :, 1])
  X_centered = zeros(size(data_tmp))
  rho_hat = 0.0
  sacf_vals = zeros(size(data, 3))

  # loop over images
  for i in axes(data, 3)
    data_tmp .= view(data, :, :, i)
    X_centered .= data_tmp .- mean(data_tmp)
    rho_hat = (1 - lam) * rho_hat + lam * sacf(X_centered, d1, d2)
    sacf_vals[i] = rho_hat
  end

  return sacf_vals

end


"""
    crit_val_sacf(M, N; alpha=0.05)

Return the asymptotic two sided critical value of the SACF statistic
[`stat_sacf`](@ref) for an `M × N` image. The null hypothesis of no spatial dependence
is rejected when the absolute value of the statistic exceeds it.

- `M::Int`: number of rows of the data matrix.
- `N::Int`: number of columns of the data matrix.
- `alpha=0.05`: significance level.

# Examples
```julia-repl
# compute critical value
crit_val_sacf(11, 11)
```
"""
function crit_val_sacf(M, N; alpha=0.05)
  quantile(Normal(0, 1), 1 - alpha / 2) / sqrt(M * N)
end
