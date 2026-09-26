# Monitoring an observed series or image sequence with a calibrated control chart.
#
# The sequential `stat_*(data, lam, ...)` functions return the EWMA chart statistics and
# the `cl_*` functions return the control limit, but nothing in the package tied the two
# together: the user had to apply the alarm rule by hand. `ControlChartResult` records the
# statistics, the limit, the alarm rule and the first alarm in one object, which is also
# what the Makie extension plots.

"""
    ControlChartResult

Result of monitoring an observed series or image sequence with an EWMA control chart;
returned by [`monitor_op`](@ref) and [`monitor_sop`](@ref).

Fields:
- `family::Symbol`: `:op` for a time series monitored with ordinal patterns, `:sop` for an
  image sequence monitored with spatial ordinal patterns.
- `chart`: the chart choice the statistics were computed for.
- `lam::Float64`: EWMA smoothing parameter.
- `cl::Float64`: control limit.
- `stats::Vector{Float64}`: the sequentially computed EWMA chart statistics.
- `time::UnitRange{Int}`: for each entry of `stats`, the index of the observation at which
  it becomes available. For ordinal patterns this is the last observation of the pattern
  window, so `time` starts at `(m - 1) * d + 1`; for image sequences it is the image index.
- `alarm::Union{Int,Nothing}`: position in `stats` of the first signal, or `nothing` if the
  chart never signals. The corresponding observation index is `time[alarm]`.
- `tail::Symbol`: side of the rejection region, `:lower` (signal when `stats[t] < cl`),
  `:upper` (`stats[t] > cl`) or `:two_sided` (`abs(stats[t]) > cl`).

With Makie loaded, `plot(res)` draws the statistics against the control limit(s) with the
rejection region shaded and the first alarm marked; see the plotting tutorial in the docs.
"""
struct ControlChartResult{C}
  family::Symbol
  chart::C
  lam::Float64
  cl::Float64
  stats::Vector{Float64}
  time::UnitRange{Int}
  alarm::Union{Int,Nothing}
  tail::Symbol
end

function Base.show(io::IO, r::ControlChartResult)
  println(io, "ControlChartResult")
  println(io, "  Family:           ", r.family === :sop ? "spatial ordinal patterns (SOP)" :
                                      "ordinal patterns (OP)")
  println(io, "  Chart:            ", r.chart)
  println(io, "  λ:                ", r.lam)
  println(io, "  Control limit:    ", round(r.cl, digits=4))
  println(io, "  Alarm rule:       ", _alarm_rule_string(r.tail))
  unit = r.family === :sop ? "image" : "observation"
  println(io, "  Statistics:       ", length(r.stats), " (", unit, "s ",
    first(r.time), " to ", last(r.time), ")")
  if isnothing(r.alarm)
    print(io, "  First alarm:      none")
  else
    print(io, "  First alarm:      ", unit, " ", r.time[r.alarm],
      " (statistic ", round(r.stats[r.alarm], digits=4), ")")
  end
end

_alarm_rule_string(tail::Symbol) =
  tail === :lower ? "stat < cl" :
  tail === :upper ? "stat > cl" : "|stat| > cl"

# The alarm rule in one place, so that `monitor_*`, the tests and the plots agree.
_signals(stat, cl, tail::Symbol) =
  tail === :lower ? stat < cl :
  tail === :upper ? stat > cl : abs(stat) > cl

"""
    monitor_op(data, lam, cl; chart_choice, m=3, d=1, add_noise=false)

Monitor the time series `data` with the EWMA control chart based on ordinal patterns and
return a [`ControlChartResult`](@ref) with the chart statistics, the control limit and
the first alarm.

The statistics are those of `stat_op(data, lam; ...)`, and the alarm rule is the one of
the chart: the chart signals when the statistic falls below `cl` for `Shannon()` and
`ShannonExtropy()`, exceeds `cl` for `DistanceToWhiteNoise()`, and exceeds `cl` in
absolute value for the four statistics of Bandt (2019).

- `data`: the time series.
- `lam::Float64`: EWMA smoothing parameter.
- `cl::Float64`: control limit, typically obtained from [`cl_op`](@ref).
- `chart_choice`: one of `Shannon()`, `ShannonExtropy()`, `DistanceToWhiteNoise()`,
  `UpDownBalance()`, `Persistence()`, `RotationalAsymmetry()`, `UpDownScaling()`.
  For `Shannon` and `ShannonExtropy`, the statistic is in the logarithm base of the
  chart, which must be larger than 1. Both default to base 2 in ComplexityMeasures.jl;
  use `Shannon(base=exp(1))` for the natural logarithm used in the papers. The control
  limit `cl` must be given in the same base.
- `m::Int=3`: length of the ordinal patterns.
- `d::Int=1`: delay between observations of a pattern.
- `add_noise::Bool=false`: add uniform noise to `data` to break ties.

```julia
x = randn(400)
res = monitor_op(x, 0.1, 0.25; chart_choice=Persistence())
res.alarm            # position of the first signal in res.stats, or nothing
```
"""
function monitor_op(data, lam, cl; chart_choice, m::Int=3, d::Int=1, add_noise::Bool=false)
  stats, _ = stat_op(data, lam; chart_choice=chart_choice, m=m, d=d, add_noise=add_noise)
  tail = _tail_op(chart_choice)
  alarm = findfirst(s -> _signals(s, cl, tail), stats)
  offset = (m - 1) * d
  time = (1 + offset):(length(stats) + offset)
  return ControlChartResult(:op, chart_choice, Float64(lam), Float64(cl), stats, time, alarm, tail)
end

"""
    monitor_sop(data, lam, cl, d1, d2; chart_choice=TauTilde(), refinement=OrdinaryType(),
      add_noise=false, noise_dist=Uniform(0, 1))

Monitor the image sequence `data` (a 3D array, rows × columns × time) with the EWMA
control chart based on spatial ordinal patterns and return a [`ControlChartResult`](@ref)
with the chart statistics, the control limit and the first alarm.

The statistics are those of `stat_sop(data, lam, d1, d2; ...)`. All four monitoring
statistics are two sided, so the chart signals when `abs(stat) > cl`.

- `data::Array{<:Real,3}`: the image sequence.
- `lam::Float64`: EWMA smoothing parameter.
- `cl::Float64`: control limit, typically obtained from [`cl_sop`](@ref).
- `d1::Int`, `d2::Int`: row and column delays.
- `chart_choice`: one of [`TauHat`](@ref)`()`, [`KappaHat`](@ref)`()`,
  [`TauTilde`](@ref)`()`, [`KappaTilde`](@ref)`()`.
- `refinement`: [`OrdinaryType`](@ref)`()` for the classical SOP classification, or one of
  [`RotationType`](@ref)`()`, [`DirectionType`](@ref)`()`, [`DiagonalType`](@ref)`()`.
  Must be the same as in the `cl_sop` call that produced `cl`.
- `add_noise::Bool=false`, `noise_dist`: add noise to break ties, as in `stat_sop`.

```julia
images = randn(11, 11, 40)
res = monitor_sop(images, 0.1, 0.03, 1, 1; chart_choice=TauTilde())
```
"""
function monitor_sop(
  data::Array{<:Real,3}, lam, cl, d1::Int, d2::Int;
  chart_choice=TauTilde(),
  refinement::SOPClassification=OrdinaryType(),
  add_noise::Bool=false,
  noise_dist::UnivariateDistribution=Uniform(0, 1)
)
  chart_choice isa Union{TauHat,KappaHat,TauTilde,KappaTilde} || throw(ArgumentError(
    "monitor_sop: chart_choice must be one of TauHat(), KappaHat(), TauTilde(), " *
    "KappaTilde(); got $(chart_choice)."
  ))
  stats = stat_sop(data, lam, d1, d2; chart_choice=chart_choice, refinement=refinement,
    add_noise=add_noise, noise_dist=noise_dist)
  tail = :two_sided
  alarm = findfirst(s -> _signals(s, cl, tail), stats)
  return ControlChartResult(:sop, chart_choice, Float64(lam), Float64(cl), stats, 1:length(stats), alarm, tail)
end
