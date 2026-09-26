
"""
    D_Chart()

Chart choice for the D-chart based on generalized ordinal patterns (GOPs). The statistic
is the squared Euclidean distance between the estimated GOP distribution and the
in-control GOP distribution; see Weiß and Schnurr (2024).
"""
struct D_Chart <: ComplexityEstimator end
struct G_Chart <: ComplexityEstimator end


"""
    chart_stat_gop(p_p0, chart_choice)
    chart_stat_gop(p_p0, G1G, ::G_Chart)

Compute the chart statistic for generalized ordinal patterns (GOPs) from the vector of
deviations `p_p0 = p - p0`, where `p` holds the estimated GOP frequencies and `p0` the
in-control GOP distribution (see [`fill_p0!`](@ref)).

- `p_p0`: vector of length 13 with the deviations of the estimated GOP frequencies from
  the in-control distribution.
- `chart_choice`: [`D_Chart`](@ref)`()` or `Persistence()`.
- `G1G`: weighting matrix `G'G` used by the G-chart method.

Returns the value of the chart statistic.
"""
function chart_stat_gop(p_p0, ::D_Chart)
  # D-chart: Equation (18), page 7 in the paper
  return dot(p_p0, p_p0)

end

function chart_stat_gop(p_p0, G1G, ::G_Chart)

  # G-chart: Equation (20), page 7 in the paper
  return dot(p_p0, G1G, p_p0)

end

function chart_stat_gop(p_p0, ::Persistence)

  # Persistence for ordinal patterns
  idx = (1, 6, 8, 10, 11, 13)

  stat = 0.0
  for i in idx
    stat += p_p0[i]^2
  end

  return stat
end
