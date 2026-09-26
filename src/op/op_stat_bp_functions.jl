
################################################################################
#                        Use information measures and lehmer                   #
################################################################################

"""
    stat_op_bp(data, w; chart_choice, m=3, ljung_box=false)

Compute the Box-Pierce (BP) type test statistic for ordinal patterns, which aggregates
the chart statistics over the delays `d = 1, …, w`.

- `data`: the time series.
- `w::Int`: maximal delay; the individual statistics for delays `1:w` are aggregated.
- `chart_choice`: one of `Shannon()`, `ShannonExtropy()`, `DistanceToWhiteNoise()`,
  `UpDownBalance()`, `Persistence()`, `RotationalAsymmetry()`, `UpDownScaling()`.
  For `Shannon` and `ShannonExtropy`, the logarithm base must be larger than 1. The
  statistic does not depend on it.
- `m::Int=3`: length of the ordinal patterns.
- `ljung_box::Bool=false`: if `true`, use Ljung-Box (BL) weights
  (delay-specific numbers of patterns) instead of the constant Box-Pierce weight.

Returns the aggregated test statistic. Critical values are provided by
[`crit_val_op_bp`](@ref).
"""
function stat_op_bp(data, w::Int; chart_choice, m::Int=3, ljung_box::Bool=false)

    # Compute lookup array and number of ops
    m_fact = factorial(m)

    # Pre-allocate
    bin = zeros(Int, m_fact)
    p_rel = zeros(Float64, m_fact)
    win = Vector{Int64}(undef, m)
    idx_used = zeros(Int, m)
    bp_stats_all = Vector{Float64}(undef, w)

    for (i, d) in enumerate(1:w)

        for range_start = 1:(length(data)-(m-1)*d) # iterate through time series

            # create unit range for indexing data
            unit_range = range(range_start; step=d, length=m)

            # create view of data based on unit range
            x_long = view(data, unit_range)

            # compute ordinal pattern based on permutations
            sortperm!(win, x_long)

            # Convert permutation to lehmer index
            index = perm_to_lehm_idx!(win, idx_used)
            fill!(idx_used, 0) # reset idx_used

            # Binarization of ordinal pattern
            bin[index] += 1

        end # end of range loop

        # Compute relative frequency of types. Entropies are converted back to the
        # natural logarithm, the scale of the constants below, so the BP statistic does
        # not depend on the logarithm base (see `log_base`).
        p_rel .= bin ./ sum(bin)
        bp_stats_all[i] = chart_stat_op(p_rel, chart_choice) * log_base(chart_choice)

        # reset bin for next iteration
        fill!(bin, 0)

    end # end of w loop

    bp_val = 0.0
    log_nr_perm = log(m_fact)

    # ---------------------------------------------------------------------------#
    #                   Sum up the individual test statistics                    # 
    # ---------------------------------------------------------------------------#
    # Weighting based on Ljung-Box?
    if ljung_box
        # Iterator object for BL-weights
        stat_weights = Iterators.map(d -> length(data) - (m - 1) * d, 1:w)
    else
        # Iterator object for BP-weights
        stat_weights = Iterators.repeated(length(data) - m + 1, w)
    end

    # (1) H-chart 
    if chart_choice isa Shannon
        for (i, weight) in enumerate(stat_weights)
            bp_val += weight * (log_nr_perm - bp_stats_all[i])
        end
        return 2 / m_fact * bp_val

        # (2) Hex-chart
    elseif chart_choice isa ShannonExtropy
        term = (m_fact - 1) * log(m_fact / (m_fact - 1))
        for (i, weight) in enumerate(stat_weights)
            bp_val += weight * (term - bp_stats_all[i])
        end
        return (2 * (m_fact - 1) / m_fact) * bp_val

        # (3) Δ-chart  
    elseif chart_choice isa DistanceToWhiteNoise
        for (i, weight) in enumerate(stat_weights)
            bp_val += weight * bp_stats_all[i]
        end
        return bp_val

        # (4) β-chart, (5) τ-chart, (6) γ-chart, (7) δ-chart  
    elseif chart_choice isa UpDownBalance || chart_choice isa Persistence ||
           chart_choice isa RotationalAsymmetry || chart_choice isa UpDownScaling

        for (i, weight) in enumerate(stat_weights)
            bp_val += weight * bp_stats_all[i]^2
        end
        return bp_val

    end # end of if statement
    # ---------------------------------------------------------------------------#

end # end of function


"""
    crit_val_op_bp(w; chart_choice, m=3, alpha=0.05)

Return the critical value for the Box-Pierce type ordinal-pattern test statistic computed
by [`stat_op_bp`](@ref).

- `w::Int`: maximal delay used in the test statistic (supported values depend on the
  chart).
- `chart_choice`: one of `Shannon()`, `ShannonExtropy()`, `DistanceToWhiteNoise()`,
  `UpDownBalance()`, `Persistence()`, `RotationalAsymmetry()`, `UpDownScaling()`.
  The critical value does not depend on the logarithm base of `Shannon` and
  `ShannonExtropy`.
- `m::Int=3`: length of the ordinal patterns (`2` or `3`, depending on the chart).
- `alpha=0.05`: significance level. For the Shannon-, extropy- and Δ-charts with `m = 3`,
  tabulated values for `alpha = 0.05` are used.
"""
function crit_val_op_bp(w::Int; chart_choice, m::Int=3, alpha=0.05)

    # ---------------------------------------------------------------------------#
    #                               op-length of 2
    # ---------------------------------------------------------------------------#

    if m == 2
        # Check if chart_choice is an instance of Shannon (any base) or DistanceToWhiteNoise
        if chart_choice isa Shannon || chart_choice isa DistanceToWhiteNoise
            return 1 / 6 * quantile(Chisq(w), 1 - alpha)
        end
    end



    # ---------------------------------------------------------------------------#
    #                               op-length of 3
    # ---------------------------------------------------------------------------#
    if m == 3
        # for H-chart, Hex-chart and Δ-chart
        if chart_choice isa Shannon || chart_choice isa ShannonExtropy || chart_choice isa DistanceToWhiteNoise
            if w == 1
                return 1.484224
            elseif w == 2
                return 2.533081
            elseif w == 3
                return 3.345710
            elseif w == 4
                return 4.207398
            elseif w == 5
                return 4.946716
            end
        end

        # critical value for β-chart
        if chart_choice isa UpDownBalance # 4
            return 1 / 3 * quantile(Chisq(w), 1 - alpha)
        end

        # critical value for τ-chart
        if chart_choice isa Persistence #5
            if w == 1
                return 0.6829163
            elseif w == 2
                return 1.0825457
            elseif w == 3
                return 1.4059123
            elseif w == 4
                return 1.7176861
            elseif w == 5
                return 1.9966817
            end
        end

        # critical value for γ-chart
        if chart_choice isa RotationalAsymmetry #6
            if w == 1
                return 1.536584
            elseif w == 2
                return 2.413527
            elseif w == 3
                return 3.142336
            elseif w == 4
                return 3.825928
            elseif w == 5
                return 4.456852
            end
        end

        # critical value for δ-chart
        if chart_choice isa UpDownScaling # 7
            if w == 1
                return 2.560972
            elseif w == 2
                return 4.282555
            elseif w == 3
                return 5.467196
            elseif w == 4
                return 6.781190
            elseif w == 5
                return 7.795281
            end
        end

    end
end
