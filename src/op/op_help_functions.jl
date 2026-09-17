

# --- Function to select abort criterium --- #
# see Equation (6), page 342, Weiss and Testik (2023)
function abort_criterium_op(stat, cl, ::Union{Shannon,ShannonExtropy})
  # Permutation Entropy-H chart: Equation (3), page 342, Weiss and Testik (2023)
  return stat < cl

end

function abort_criterium_op(stat, cl, ::DistanceToWhiteNoise)
  # Permutation Entropy-H chart: Equation (3), page 342, Weiss and Testik (2023)
  return stat > cl

end

function abort_criterium_op(stat, cl, ::Union{UpDownBalance,Persistence,RotationalAsymmetry,UpDownScaling})
  # Permutation Entropy-H chart: Equation (3), page 342, Weiss and Testik (2023)
  return abs(stat) > cl

end

# Side of the rejection region of an OP chart, as a symbol: `:lower` (signal when the
# statistic falls below the limit), `:upper` (above) or `:two_sided` (absolute value
# above). This is the same rule as `abort_criterium_op` and `reject`, in a form the
# plotting code can read off without evaluating it.
_tail_op(::Union{Shannon,ShannonExtropy}) = :lower
_tail_op(::DistanceToWhiteNoise) = :upper
_tail_op(::Union{UpDownBalance,Persistence,RotationalAsymmetry,UpDownScaling}) = :two_sided

"""
    add_noise!(vec, dist)

Add standard-uniform noise to each element of `vec` in-place if `dist` is a
`DiscreteDistribution`, in order to break ties before computing ordinal patterns. If
`dist` is a `ContinuousDistribution`, `vec` is returned unchanged.
"""
function add_noise!(vec, ::DiscreteDistribution)
  for i in axes(vec, 1)
    vec[i] += rand()
  end
end

# Method for continous distribution -> do nothing
function add_noise!(vec, ::ContinuousDistribution)
  return vec
end
