
"""
    arl_sacf_ic(sp_dgp, lam, cl, d1, d2, reps=10_000; rl_max=typemax(Int))

Compute the in-control average run length (ARL) of the EWMA chart based on the spatial
autocorrelation function (SACF) at lag `(d1, d2)` via simulation. The computation is
multithreaded.

- `sp_dgp::ICSTS`: in-control spatial data generating process.
- `lam`: smoothing parameter of the EWMA statistic.
- `cl`: control limit of the chart, typically obtained from [`cl_sacf`](@ref).
- `d1::Int`: row delay.
- `d2::Int`: column delay.
- `reps=10_000`: number of replications.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns the tuple `(ARL, standard error)`.
"""
function arl_sacf_ic(
    sp_dgp::ICSTS, lam, cl, d1::Int, d2::Int, reps=10_000; rl_max::Int=typemax(Int)
)

    # Extract        
    dist_error = sp_dgp.dist

    # Number of chunks for load balancing
    n_chunks = Threads.nthreads() * 4

    # Make chunks for separate tasks (based on number of threads)
    chunks = Iterators.partition(1:reps, div(reps, n_chunks))

    par_results = map(chunks) do i
        Threads.@spawn rl_sacf_ic(sp_dgp, lam, cl, d1, d2, i, dist_error, rl_max)
    end

    # Collect results from tasks
    rls = fetch.(par_results)
    rlvec = Iterators.flatten(rls) |> collect
    return (mean(rlvec), std(rlvec) / sqrt(reps))
end


"""
    rl_sacf_ic(sp_dgp, lam, cl, d1, d2, p_reps, dist_error, rl_max=typemax(Int))

Simulate in-control run lengths of the EWMA chart based on the spatial autocorrelation
function (SACF) at lag `(d1, d2)`. Internal helper of [`arl_sacf_ic`](@ref).

- `sp_dgp::ICSTS`: in-control spatial data generating process.
- `lam`: smoothing parameter of the EWMA statistic.
- `cl`: control limit of the chart.
- `d1::Int`: row delay.
- `d2::Int`: column delay.
- `p_reps::UnitRange`: replications handled by this call. [`arl_sacf_ic`](@ref) splits
  its replications into ranges like this and processes them in parallel.
- `dist_error::UnivariateDistribution`: distribution of the error term of the process.
- `rl_max::Int=typemax(Int)`: maximal run length after which a replication is stopped.

Returns the vector of run lengths, one per replication in `p_reps`.
"""
function rl_sacf_ic(
    sp_dgp::ICSTS, lam, cl, d1::Int, d2::Int, p_reps::UnitRange, dist_error::UnivariateDistribution, rl_max::Int=typemax(Int)
)

    # Extract matrix size and pre-allocate matrices
    M = sp_dgp.M_rows
    N = sp_dgp.N_cols
    data = zeros(M, N)
    X_centered = similar(data)
    rls = zeros(Int, length(p_reps))

    for r in axes(p_reps, 1)

        rho_hat = 0.0
        rl = 0

        while abs(rho_hat) < cl
            rl += 1

            # fill matrix with iid values
            rand!(dist_error, data)
            X_centered .= data .- mean(data)

            # compute ρ(d1,d2)-EWMA
            rho_hat = (1 - lam) * rho_hat + lam * sacf(X_centered, d1, d2)

            # Break while loop when rl exceeds rl_max
            if rl > rl_max
                break
            end
        end

        rls[r] = rl

    end

    return rls

end
