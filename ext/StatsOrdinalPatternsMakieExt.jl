module StatsOrdinalPatternsMakieExt

# Plotting support, loaded automatically once Makie (or a backend such as CairoMakie or
# GLMakie) is loaded next to StatsOrdinalPatterns. It adds methods to `Makie.plot` and
# `Makie.plot!` for the package's result types:
#
#   plot(res::ControlChartResult)   chart statistics against the control limit(s), with
#                                   the rejection region shaded and the first alarm marked
#   plot(res::…TestResultBoot)      histogram of the resampled null distribution with the
#   plot(res::OPTestResultSurrogate) rejection region, critical value(s) and the observed
#                                   statistic
#
# Every `plot` builds a Figure with one Axis and returns the Figure. `plot!(ax, res)` draws
# into an existing Axis, and `plot(fig[i, j], res)` into a grid position. Which side is
# rejected, which vector holds the null sample and how the test is called are read from
# `src/other/plot_support.jl`, so no statistical rule is repeated here.

using StatsOrdinalPatterns
using StatsOrdinalPatterns: ControlChartResult, rejection_tail, null_sample, null_crit,
  null_pval, null_reject, null_label, _BootResult
using Makie

const _NullResult = Union{_BootResult,OPTestResultSurrogate}

# ------------------------------------------------------------------------------
# 1. SHARED HELPERS
# ------------------------------------------------------------------------------

# Range of `values` extended by `pad` on both sides, so that shaded regions and limit
# lines never sit on the axis border. A constant vector gets a range of its own.
function _padded_range(values; pad=0.06)
  lo, hi = extrema(values)
  width = hi - lo
  width == 0 && (width = max(abs(lo), 1.0))
  return lo - pad * width, hi + pad * width
end

# The limit(s) to draw: one line for a one-sided rule, ±limit for a two-sided one.
_limit_values(limit, tail::Symbol) = tail === :two_sided ? [-limit, limit] : [limit]

# Shade the rejection region along one axis direction. `span!` is `hspan!` for a limit on
# the y axis (control charts) and `vspan!` for a limit on the x axis (null distributions).
function _shade_rejection!(span!, ax, limit, tail::Symbol, lo, hi, color)
  if tail === :two_sided
    span!(ax, limit, hi; color=color, label="rejection region")
    span!(ax, lo, -limit; color=color)
  elseif tail === :upper
    span!(ax, limit, hi; color=color, label="rejection region")
  else
    span!(ax, lo, limit; color=color, label="rejection region")
  end
  return nothing
end

_fmt(x) = string(round(x, digits=4))

# Legend inside the axis at a Makie position such as `:rt`, or none for `false`. This is
# what `plot!` on a user-supplied axis offers; the `plot` entry points (section 4) place
# the legend outside the axis by default, where it cannot cover the data.
function _inside_legend!(ax, legend)
  legend === false && return nothing
  Makie.axislegend(ax; position=legend, framevisible=false)
  return nothing
end

# ------------------------------------------------------------------------------
# 2. CONTROL CHART: statistics against the control limit(s)
# ------------------------------------------------------------------------------

_time_label(r::ControlChartResult) = r.family === :sop ? "image t" : "observation t"

_stat_label(r::ControlChartResult) = "EWMA chart statistic ($(r.chart))"

function _chart_title(r::ControlChartResult)
  family = r.family === :sop ? "SOP" : "OP"
  return "$family control chart, λ = $(r.lam), cl = $(_fmt(r.cl))"
end

function _chart_subtitle(r::ControlChartResult)
  isnothing(r.alarm) && return "no alarm"
  return "first alarm at $(_time_label(r)) = $(r.time[r.alarm])"
end

_axis_defaults(r::ControlChartResult) = (
  xlabel=_time_label(r), ylabel=_stat_label(r),
  title=_chart_title(r), subtitle=_chart_subtitle(r),
)

"""
    plot!(ax::Axis, res::ControlChartResult; kwargs...)

Draw the chart statistics of `res` into `ax`, with the control limit(s) as dashed lines,
the rejection region shaded and the first alarm marked. Keyword arguments:

- `color=:black`, `linewidth=1.5`: the statistic line.
- `limit_color=:firebrick`: control limit line(s) and the alarm marker.
- `region_color=(:firebrick, 0.12)`: fill of the rejection region.
- `legend=:rt`: position of the legend inside the axis (a Makie position such as `:rt`
  or `:lb`), or `false` for none.
"""
function Makie.plot!(
  ax::Makie.Axis, r::ControlChartResult;
  color=:black, linewidth=1.5, limit_color=:firebrick, region_color=(:firebrick, 0.12),
  legend=:rt
)
  t = collect(r.time)
  limits = _limit_values(r.cl, r.tail)
  lo, hi = _padded_range(vcat(r.stats, limits))

  _shade_rejection!(Makie.hspan!, ax, r.cl, r.tail, lo, hi, region_color)
  Makie.hlines!(ax, limits; color=limit_color, linestyle=:dash, label="control limit")
  Makie.lines!(ax, t, r.stats; color=color, linewidth=linewidth, label="chart statistic")

  if !isnothing(r.alarm)
    t_alarm = r.time[r.alarm]
    Makie.vlines!(ax, [t_alarm]; color=limit_color, linestyle=:dot)
    Makie.scatter!(ax, [t_alarm], [r.stats[r.alarm]];
      color=limit_color, marker=:xcross, markersize=16, label="first alarm")
  end

  Makie.ylims!(ax, lo, hi)
  _inside_legend!(ax, legend)
  return ax
end

# ------------------------------------------------------------------------------
# 3. NULL DISTRIBUTION: histogram of the bootstrap or surrogate sample
# ------------------------------------------------------------------------------

function _null_subtitle(r::_NullResult)
  return "stat = $(_fmt(r.stat)), crit = $(_fmt(null_crit(r))), " *
         "p = $(_fmt(null_pval(r))), reject H₀: $(null_reject(r))"
end

_axis_defaults(r::_NullResult) = (
  xlabel="statistic", ylabel="count",
  title=null_label(r), subtitle=_null_subtitle(r),
)

"""
    plot!(ax::Axis, res; kwargs...)

Draw the resampled null distribution of a bootstrap or surrogate test result `res` into
`ax` as a histogram, with the rejection region shaded, the critical value(s) as dashed
lines and the observed statistic as a solid line. Keyword arguments:

- `nbins=40`: number of histogram bins.
- `color=(:steelblue, 0.8)`: histogram fill.
- `stat_color=:black`: the observed statistic.
- `crit_color=:firebrick`: critical value line(s).
- `region_color=(:firebrick, 0.12)`: fill of the rejection region.
- `legend=:rt`: position of the legend inside the axis (a Makie position such as `:rt`
  or `:lt`), or `false` for none.
"""
function Makie.plot!(
  ax::Makie.Axis, r::_NullResult;
  nbins::Int=40, color=(:steelblue, 0.8), stat_color=:black, crit_color=:firebrick,
  region_color=(:firebrick, 0.12), legend=:rt
)
  sample = null_sample(r)
  crit = null_crit(r)
  tail = rejection_tail(r)
  limits = _limit_values(crit, tail)
  lo, hi = _padded_range(vcat(sample, r.stat, limits))

  _shade_rejection!(Makie.vspan!, ax, crit, tail, lo, hi, region_color)
  Makie.hist!(ax, sample; bins=nbins, color=color, strokewidth=0.5, strokecolor=:white,
    label="resampled statistics")
  Makie.vlines!(ax, limits; color=crit_color, linestyle=:dash, label="critical value")
  Makie.vlines!(ax, [r.stat]; color=stat_color, linewidth=2, label="observed statistic")

  Makie.xlims!(ax, lo, hi)
  _inside_legend!(ax, legend)
  return ax
end

# ------------------------------------------------------------------------------
# 4. FIGURE LEVEL ENTRY POINTS (shared by both plot kinds)
# ------------------------------------------------------------------------------

const _Plottable = Union{ControlChartResult,_NullResult}

"""
    plot(res; figure=(;), axis=(;), legend=:outside, kwargs...)
    plot(fig[i, j], res; axis=(;), legend=:outside, kwargs...)

Plot a [`ControlChartResult`](@ref) or a bootstrap / surrogate test result. The first form
creates a `Figure` with one `Axis` and returns the `Figure`; the second draws into the
grid position of an existing figure and returns the `Axis`.

- `figure`: keyword arguments for `Figure`, e.g. `figure=(size=(800, 400),)`.
- `axis`: keyword arguments for `Axis`, overriding the default labels and title,
  e.g. `axis=(title="my chart",)`.
- `legend`: `:outside` (the default) places the legend to the right of the axis, where it
  never covers the data; a Makie position such as `:rt` or `:lb` places it inside the
  axis, and `false` omits it.
- `kwargs`: passed on to `plot!(ax, res; kwargs...)`.
"""
function Makie.plot(r::_Plottable; figure=(;), axis=(;), legend=:outside, kwargs...)
  fig = Makie.Figure(; size=(760, 400), figure...)
  Makie.plot(fig[1, 1], r; axis=axis, legend=legend, kwargs...)
  return fig
end

function Makie.plot(gp::Makie.GridPosition, r::_Plottable; axis=(;), legend=:outside, kwargs...)
  if legend === :outside
    # A nested layout at the grid position holds the axis and, to its right, the legend,
    # so the pair occupies the single cell the caller handed over.
    layout = Makie.GridLayout(gp)
    ax = Makie.Axis(layout[1, 1]; _axis_defaults(r)..., axis...)
    Makie.plot!(ax, r; legend=false, kwargs...)
    Makie.Legend(layout[1, 2], ax; framevisible=false)
  else
    ax = Makie.Axis(gp; _axis_defaults(r)..., axis...)
    Makie.plot!(ax, r; legend=legend, kwargs...)
  end
  return ax
end

end
