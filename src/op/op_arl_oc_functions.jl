
"""
    arl_op_oc(op_dgp, lam, cl, reps=10_000; chart_choice, d=1, m=3, ced=false, ad=100,
      rl_max=typemax(Int))

Compute the out-of-control average run length (ARL) of the EWMA chart based on ordinal
patterns via simulation, following Weiß and Testik (2023). The computation is
multithreaded.

- `op_dgp`: out-of-control DGP (e.g. `AR1`, `MA1`, `MA2`, `TEAR1`, `QAR1`).
- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the chart.
- `reps::Int=10_000`: number of replications.
- `chart_choice`: one of `Shannon()`, `ShannonExtropy()`, `DistanceToWhiteNoise()`,
  `UpDownBalance()`, `Persistence()`, `RotationalAsymmetry()`, `UpDownScaling()`
  (see [`chart_stat_op`](@ref)).
  For `Shannon` and `ShannonExtropy`, the statistic is in the logarithm base of the
  chart, which must be larger than 1. Both default to base 2 in ComplexityMeasures.jl;
  use `Shannon(base=exp(1))` for the natural logarithm used in the papers. The control
  limit `cl` must be given in the same base.
- `d::Int=1`: delay between observations of a pattern.
- `m::Int=3`: length of the ordinal patterns.
- `ced::Bool=false`: use conditional expected delay initialization.
- `ad::Int=100`: number of in-control iterations for `ced`.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns the tuple `(ARL, standard error)`.
"""
function arl_op_oc(
  op_dgp, lam, cl, reps=10_000; chart_choice, d::Int=1, m::Int=3, ced=false, ad=100, rl_max::Int=typemax(Int)
)

  # Number of chunks for load balancing
  n_chunks = Threads.nthreads() * 4

  # Make chunks for separate tasks (based on number of threads)
  chunks = Iterators.partition(1:reps, div(reps, n_chunks))

  par_results = map(chunks) do i
    Threads.@spawn rl_op_oc(
      op_dgp, lam, cl, i, op_dgp.dist, chart_choice,
      d, m, ced, ad, rl_max
    )
  end

  # Collect results from tasks
  rls = fetch.(par_results)
  rlvec = Iterators.flatten(rls) |> collect
  return (mean(rlvec), std(rlvec) / sqrt(reps))
end



"""
    rl_op_oc(op_dgp, lam, cl, p_reps, op_dgp_dist, chart_choice, d, m, ced, ad,
      rl_max=typemax(Int))

Compute out-of-control run lengths of the EWMA chart based on ordinal patterns for a
chunk of replications. This is the single-threaded worker used by [`arl_op_oc`](@ref).

- `op_dgp`: out-of-control DGP (e.g. `AR1`, `MA1`, `MA2`, `TEAR1`, `QAR1`).
- `lam::Float64`: smoothing parameter of the EWMA statistic.
- `cl::Float64`: control limit of the chart.
- `p_reps`: range of replication indices to process.
- `op_dgp_dist::UnivariateDistribution`: distribution of the DGP.
- `chart_choice`: chart choice (see [`chart_stat_op`](@ref)).
- `d::Int`: delay between observations of a pattern.
- `m::Int`: length of the ordinal patterns.
- `ced::Bool`: use conditional expected delay initialization.
- `ad::Int`: number of in-control iterations for `ced`.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns a vector of run lengths.
"""
function rl_op_oc(
  op_dgp, lam, cl, p_reps, op_dgp_dist, chart_choice, d, m, ced, ad, rl_max::Int=typemax(Int)
)
  m_fact = factorial(m)
  rls = zeros(Int, length(p_reps))
  p = zeros(Float64, m_fact)
  bin = zeros(Int, m_fact)
  win = zeros(Int, m)
  idx_used = similar(win)

  if ced
    pool_vector = Vector{Float64}(undef, 10_000)
    init_dgp_op!(op_dgp, pool_vector, op_dgp_dist, 1)
  else
    pool_vector = Float64[]
  end

  x_seq = Vector{Float64}(undef, m + (d > 1 ? d : 0))
  # xbiv not needed for generic OP DGPs

  for r in axes(p_reps, 1)
    if ced
      icrun = true
      while icrun
        fill!(p, 1 / m_fact)
        seq = init_dgp_op_ced!(op_dgp, x_seq, pool_vector, d)
        falarm = false
        for _ in 1:ad
          bin .= 0
          sortperm!(win, seq)
          index = perm_to_lehm_idx!(win, idx_used)
          bin[index] = 1
          fill!(idx_used, 0)

          @. p = lam * bin + (1.0 - lam) * p
          stat = chart_stat_op(p, chart_choice)
          seq = update_dgp_op_ced!(op_dgp, x_seq, pool_vector, d)

          if abort_criterium_op(stat, cl, chart_choice)
            falarm = true
            break
          end
        end
        !falarm && (icrun = false)
      end
    else
      fill!(p, 1 / m_fact)
      # Standard init without eps_long
      seq = init_dgp_op!(op_dgp, x_seq, op_dgp_dist, d)
      stat = chart_stat_op(p, chart_choice)
    end

    rl = 0
    while !abort_criterium_op(stat, cl, chart_choice)
      rl += 1
      bin .= 0
      sortperm!(win, seq)
      index = perm_to_lehm_idx!(win, idx_used)
      bin[index] = 1
      fill!(idx_used, 0)

      @. p = lam * bin + (1.0 - lam) * p
      stat = chart_stat_op(p, chart_choice)
      # Standard update
      seq = update_dgp_op!(op_dgp, x_seq, op_dgp_dist, d)

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end
    rls[r] = rl
  end
  return rls
end


# Length of the burn-in prerun used to initialize the AAR(1) and QAR(1) sequences.
const _OP_OC_BURN_IN = 100

# Methods for the DGPs whose init/update take the extra `eps_long` vector: MA1 and MA2
# need it for the epsilons, AAR1 and QAR1 additionally need the burn-in vector `xbiv`.
function rl_op_oc(
  op_dgp::Union{MA1,MA2,TEAR1,AAR1,QAR1}, lam, cl, p_reps, op_dgp_dist, chart_choice, d, m, ced, ad, rl_max::Int=typemax(Int)
)
  m_fact = factorial(m)
  rls = zeros(Int, length(p_reps))
  p = zeros(Float64, m_fact)
  bin = zeros(Int, m_fact)
  win = zeros(Int, m)
  idx_used = similar(win)

  # Determine sequence lengths: the MA states start at index 2 (MA1) resp. 3 (MA2),
  # the autoregressive-type states at index 1.
  offset = op_dgp isa Union{MA1,MA2} ? _ma_offset(op_dgp) : 0
  x_seq = Vector{Float64}(undef, m + (d > 1 ? d : 0) + offset)
  eps_long = similar(x_seq)
  rand!(op_dgp_dist, eps_long)
  # Burn-in vector; only used by the AAR(1) and QAR(1) initialization.
  xbiv = Vector{Float64}(undef, _OP_OC_BURN_IN)

  # Pre-allocate pool if CED is used
  if ced
    pool_vector = Vector{Float64}(undef, 10_000)
    init_dgp_op!(op_dgp, pool_vector, similar(pool_vector), op_dgp_dist, 1, xbiv)
    # The MA initialization leaves the first `offset` entries undefined.
    offset > 0 && (pool_vector = pool_vector[(offset+1):end])
  else
    pool_vector = Float64[]
  end

  for r in axes(p_reps, 1)
    if ced
      icrun = true
      while icrun
        fill!(p, 1 / m_fact)
        seq = init_dgp_op_ced!(op_dgp, x_seq, pool_vector, d)
        falarm = false
        for _ in 1:ad
          bin .= 0
          sortperm!(win, seq)
          index = perm_to_lehm_idx!(win, idx_used)
          bin[index] = 1
          fill!(idx_used, 0)

          @. p = lam * bin + (1.0 - lam) * p
          stat = chart_stat_op(p, chart_choice)
          seq = update_dgp_op_ced!(op_dgp, x_seq, pool_vector, d)

          if abort_criterium_op(stat, cl, chart_choice)
            falarm = true
            break
          end
        end
        !falarm && (icrun = false)
      end
    else
      fill!(p, 1 / m_fact)
      seq = init_dgp_op!(op_dgp, x_seq, eps_long, op_dgp_dist, d, xbiv)
      stat = chart_stat_op(p, chart_choice)
    end

    rl = 0
    while !abort_criterium_op(stat, cl, chart_choice)
      rl += 1
      bin .= 0
      sortperm!(win, seq)
      index = perm_to_lehm_idx!(win, idx_used)
      bin[index] = 1
      fill!(idx_used, 0)

      @. p = lam * bin + (1.0 - lam) * p
      stat = chart_stat_op(p, chart_choice)
      # Specific update for MA processes using eps_long
      seq = update_dgp_op!(op_dgp, x_seq, eps_long, op_dgp_dist, d)

      # Break while loop when rl exceeds rl_max
      if rl > rl_max
        break
      end
    end
    rls[r] = rl
  end
  return rls
end
