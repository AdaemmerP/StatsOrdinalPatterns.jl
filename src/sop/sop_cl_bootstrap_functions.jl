"""
    cl_sop_bootstrap(data::Array{T,3}, lam, L0, cl_init, d1, d2;
                     reps_final, reps_bracket, bracket_step,
                     arl_truncation_factor, chart_choice, refinement,
                     verbose, cl_tol, seed) where {T<:Real}

Compute the critical limit `cl` for an EWMA-SOP control chart using a
bootstrap approach, such that the in-control ARL equals `L0`.

Relative type frequencies are pre-computed once from `data` before the
search, so all ARL evaluations reuse the same empirical distribution.

The search proceeds in two phases:
1. **Bracketing**: A coarse Monte Carlo estimate (using `reps_bracket` replications)
   steps away from `cl_init` in increments of `bracket_step` until an interval
   `[a, b]` is found where the ARL crosses `L0`.
2. **Refinement**: The ITP root-finding algorithm narrows the bracket to
   within `cl_tol` using full Monte Carlo estimates (`reps_final` replications).

Using a fixed `seed` ensures the ARL function behaves smoothly across
evaluations within a single call. With large `reps_final`, the result is accurate
regardless of which seed is used. Pass `seed=nothing` (default) for independent
results across calls, or fix the seed for reproducibility.

# Arguments
- `data::Array{T,3}`: 3D array of images used to compute the empirical
  relative type frequencies for bootstrapping.
- `lam`: EWMA smoothing parameter λ ∈ (0, 1].
- `L0`: Target in-control ARL.
- `cl_init`: Initial guess for the critical limit.
- `d1`: First delay dimension.
- `d2`: Second delay dimension.

# Keyword Arguments
- `reps_final=10_000`: Replications used during the ITP refinement phase.
- `reps_bracket=1_000`: Replications used during the bracketing phase.
- `bracket_step=0.001`: Step size for the bracketing search.
- `arl_truncation_factor=50`: Individual simulation runs are capped at
  `arl_truncation_factor * L0` steps during bracketing.
- `chart_choice=TauTilde()`: Control chart statistic to use.
- `refinement=OrdinaryType()`: SOP classification; a `RefinedType` applies a refined chart computation.
- `cl_tol=1e-4`: Absolute convergence tolerance on `cl` for the ITP phase.
- `seed=nothing`: Random seed for reproducibility.
- `verbose=false`: If `true`, prints progress information at each evaluation.

# Returns
- `cl::Float64`: The critical limit achieving an in-control ARL of `L0`.

# Example
```julia
cl = cl_sop_bootstrap(data, 0.1, 370.0, 2.5, 3, 5; reps_final=50_000, seed=42)
```
"""
function cl_sop_bootstrap(
    data::Array{T,3}, lam, L0, cl_init, d1, d2;
    reps_final=10_000,
    reps_bracket=1_000,
    bracket_step=0.001,
    arl_truncation_factor=50,
    chart_choice=TauTilde(),
    refinement=OrdinaryType(),
    verbose=false,
    cl_tol=1e-4,
    seed=nothing
) where {T<:Real}

    # Pre-compute relative type frequencies once from data.
    p_array = compute_p_array(data, d1, d2; chart_choice=chart_choice, refinement=refinement)

    # Evaluates ARL(cl) via MC simulation;
    # fixed seed ensures comparable noise across calls.
    function get_arl(cl, current_reps, current_truncate)
        Random.seed!(seed)
        res = arl_sop_bootstrap(
            p_array, lam, cl, current_reps;
            chart_choice=chart_choice, refinement=refinement, rl_max=current_truncate
        )
        if verbose
            println("  cl = ", round(cl, digits=6), " | ARL = ", round(res[1], digits=4))
        end
        return res[1]
    end

    # Ensure we have a seed for reproducibility.
    seed = isnothing(seed) ? rand(UInt64) : seed # Random.seed! rejects negative seeds on Julia 1.10

    # Cap ARL runs during bracketing to avoid wasting reps far from the root.
    trunc_val = round(Int, arl_truncation_factor * L0) # rl_max must be an Int
    cap_val = trunc_val / 10

    # Step 1: Find an interval [a, b] that brackets the root using coarse MC.
    if verbose
        println("\n" * "="^60)
        println(" STEP 1: Bracket Search")
        println(" reps = $reps_bracket | truncation cap = $trunc_val | step size = $bracket_step")
        println("="^60)
    end

    a = cl_init
    f_a = get_arl(a, reps_bracket, trunc_val) - L0

    # If cl_init hits the truncation cap, shift down until we get a real ARL estimate.
    if f_a >= cap_val
        if verbose
            println("  [!] cl_init hits truncation cap — shifting down:")
        end
        while f_a >= cap_val
            a -= bracket_step
            f_a = get_arl(a, reps_bracket, trunc_val) - L0
            if verbose
                println("      → cl = ", round(a, digits=6), " | ARL - L0 = ", round(f_a, digits=4), " (cap = $cap_val)")
            end
        end
    end

    direction = (f_a < 0) ? 1.0 : -1.0
    b = a + (direction * bracket_step)
    f_b = get_arl(b, reps_bracket, trunc_val) - L0

    search_iter = 0
    while f_a * f_b > 0
        b += (direction * bracket_step)
        f_b = get_arl(b, reps_bracket, trunc_val) - L0
        search_iter += 1
        if search_iter > 100
            error("Could not find a bracket. Check if cl_init is reasonable.")
        end
    end

    bracket = (min(a, b), max(a, b))
    if verbose
        println("-"^60)
        println(" Bracket found: [", round(bracket[1], digits=6), ", ", round(bracket[2], digits=6), "]")
        println("-"^60)
    end

    # Step 2: Refine to the root via ITP using full reps and no truncation.
    if verbose
        println("\n" * "="^60)
        println(" STEP 2: Root Finding via ITP")
        println(" reps = $reps_final | no truncation | tolerance = $cl_tol")
        println("="^60)
    end
    final_cl = find_zero(
        cl -> get_arl(cl, reps_final, typemax(Int)) - L0,
        bracket,
        Roots.ITP(),
        xatol=cl_tol,
        verbose=verbose
    )

    return final_cl
end
