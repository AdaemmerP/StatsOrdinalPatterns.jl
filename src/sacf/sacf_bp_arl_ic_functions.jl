"""
    arl_sacf_bp_ic(sp_dgp, lam, cl, w, reps=10_000; rl_max=typemax(Int))

Compute the in-control average run length (ARL) of the EWMA chart based on the
Box-Pierce type SACF statistic [`stat_sacf_bp`](@ref) via simulation. The computation is
multithreaded.

- `sp_dgp::ICSTS`: in-control spatial data generating process.
- `lam`: smoothing parameter of the EWMA statistic.
- `cl`: control limit of the chart, typically obtained from [`cl_sacf_bp`](@ref).
- `w::Int`: maximal lag of the BP statistic (see [`stat_sacf_bp`](@ref)).
- `reps=10_000`: number of replications.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns the tuple `(ARL, standard error)`.
"""
function arl_sacf_bp_ic(sp_dgp::ICSTS, lam, cl, w::Int, reps=10_000; rl_max::Int=typemax(Int))

    # Extract distribution        
    dist_error = sp_dgp.dist

    # Number of chunks for load balancing
    n_chunks = Threads.nthreads() * 4

    # Make chunks for separate tasks (based on number of threads)
    chunks = Iterators.partition(1:reps, div(reps, n_chunks))

    par_results = map(chunks) do i
        Threads.@spawn rl_sacf_bp_ic(sp_dgp, lam, cl, w, i, dist_error, rl_max)
    end

    # Collect results from tasks
    rls = fetch.(par_results)
    rlvec = Iterators.flatten(rls) |> collect
    return (mean(rlvec), std(rlvec) / sqrt(reps))
end

"""
    rl_sacf_bp_ic(sp_dgp, lam, cl, w, p_reps, dist_error, rl_max=typemax(Int))

Simulate in-control run lengths of the EWMA chart based on the Box-Pierce type SACF
statistic [`stat_sacf_bp`](@ref). Internal helper of [`arl_sacf_bp_ic`](@ref).

- `sp_dgp::ICSTS`: in-control spatial data generating process.
- `lam`: smoothing parameter of the EWMA statistic.
- `cl`: control limit of the chart.
- `w::Int`: maximal lag of the BP statistic.
- `p_reps::UnitRange`: replications handled by this call. [`arl_sacf_bp_ic`](@ref) splits
  its replications into ranges like this and processes them in parallel.
- `dist_error::UnivariateDistribution`: distribution of the error term of the process.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns the vector of run lengths, one per replication in `p_reps`.
"""
function rl_sacf_bp_ic(
    sp_dgp::ICSTS, lam, cl, w::Int, p_reps::UnitRange, dist_error::UnivariateDistribution, rl_max::Int=typemax(Int)
)

    # Extract matrix size and pre-allocate data matrices
    M = sp_dgp.M_rows
    N = sp_dgp.N_cols
    data = zeros(M, N)
    X_centered = similar(data)
    rls = zeros(Int, length(p_reps))

    # Compute all relevant h1-h2 combinations
    set_1 = Iterators.product(1:w, 0:w)
    set_2 = Iterators.product(-w:0, 1:w)
    h1_h2_combinations = Iterators.flatten(Iterators.zip(set_1, set_2))
    rho_hat_all = zeros(length(h1_h2_combinations))

    for r in axes(p_reps, 1)

        fill!(rho_hat_all, 0.0)
        bp_stat = 0.0
        rl = 0

        while bp_stat < cl
            rl += 1

            # fill data matrix 
            rand!(dist_error, data)

            # Demean data for SACF
            X_centered .= data .- mean(data)

            # compute BP-statistic using all h1-h2 combinations
            bp_stat = 0.0 # Initialize BP-sum
            for (i, (h1, h2)) in enumerate(h1_h2_combinations)

                rho_hat_all[i] = (1 - lam) * rho_hat_all[i] + lam * sacf(X_centered, h1, h2)
                bp_stat += 2 * rho_hat_all[i]^2

            end

            # Break while loop when rl exceeds rl_max
            if rl > rl_max
                break
            end
        end

        rls[r] = rl

    end

    return rls

end
