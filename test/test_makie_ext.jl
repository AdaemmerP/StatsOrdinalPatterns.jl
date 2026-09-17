using Makie
using TimeseriesSurrogates

# The figures are built but never rendered: no backend is loaded, which keeps the test
# independent of graphics libraries. The assertions check that every result type has a
# `plot` method, that the figure has the expected structure and that the drawn limits
# agree with the result.

# Blocks and plots of a one-axis figure
_only_axis(fig) = only(filter(c -> c isa Makie.Axis, fig.content))
_plots_of(T, ax) = filter(p -> p isa T, ax.scene.plots)

@testset "Makie extension: control chart" begin
  x = ar1_series(300, 0.6, 3)
  res = monitor_op(x, 0.1, 0.2; chart_choice=Persistence())
  @test !isnothing(res.alarm)   # AR(1) with φ = 0.6 signals long before t = 300

  fig = plot(res)
  @test fig isa Makie.Figure
  ax = _only_axis(fig)
  @test ax.xlabel[] == "observation t"
  @test occursin("OP control chart", ax.title[])
  @test occursin(string(res.time[res.alarm]), ax.subtitle[])
  @test length(_plots_of(Makie.Lines, ax)) == 1
  # two-sided rule: two shaded regions, limits at ±cl, alarm marker present
  @test length(_plots_of(Makie.HSpan, ax)) == 2
  @test _plots_of(Makie.HLines, ax)[1][1][] == [-0.2, 0.2]
  @test length(_plots_of(Makie.Scatter, ax)) == 1
  @test any(c -> c isa Makie.Legend, fig.content)

  # one-sided rules: one region, one limit; no alarm means no marker
  res_lo = monitor_op(x, 0.1, 0.0; chart_choice=Shannon())
  @test isnothing(res_lo.alarm)
  ax = _only_axis(plot(res_lo))
  @test length(_plots_of(Makie.HSpan, ax)) == 1
  @test _plots_of(Makie.HLines, ax)[1][1][] == [0.0]
  @test isempty(_plots_of(Makie.Scatter, ax))
  @test ax.subtitle[] == "no alarm"

  res_up = monitor_op(x, 0.1, 0.05; chart_choice=DistanceToWhiteNoise())
  ax = _only_axis(plot(res_up))
  @test length(_plots_of(Makie.HSpan, ax)) == 1

  # SOP monitoring uses the image index on the x axis
  images = randn(MersenneTwister(4), 11, 11, 20)
  res_sop = monitor_sop(images, 0.1, 0.03, 1, 1; chart_choice=TauTilde())
  ax = _only_axis(plot(res_sop))
  @test ax.xlabel[] == "image t"
  @test occursin("SOP control chart", ax.title[])

  # plot! into an existing axis, plot into a grid position, keyword overrides
  fig = Makie.Figure()
  ax = Makie.Axis(fig[1, 1])
  @test plot!(ax, res; legend=false) === ax
  @test !any(c -> c isa Makie.Legend, fig.content)
  ax2 = plot(fig[2, 1], res; axis=(title="custom",), color=:blue)
  @test ax2 isa Makie.Axis
  @test ax2.title[] == "custom"
  @test count(c -> c isa Makie.Legend, fig.content) == 1   # legend of ax2, outside
  ax3 = plot(fig[3, 1], res; legend=:lt)
  @test count(c -> c isa Makie.Legend, fig.content) == 2   # inside legend of ax3
  fig = plot(res; figure=(size=(300, 200),), legend=:lt)
  @test fig.scene.viewport[].widths == [300, 200]
  @test count(c -> c isa Makie.Legend, fig.content) == 1
  fig = plot(res; legend=false)
  @test !any(c -> c isa Makie.Legend, fig.content)
end

@testset "Makie extension: null distributions" begin
  x = ar1_series(200, 0.5, 5)
  img = smoothed_image(20, 6)
  n_boot = 100

  results = Any[
    test_op_bootstrap(x, n_boot; chart_choice=Persistence()),            # two-sided
    test_op_bootstrap(x, n_boot; chart_choice=Shannon()),                # lower tail
    test_op_bootstrap(x, n_boot; chart_choice=DistanceToWhiteNoise()),   # upper tail
    test_op_bootstrap(x, n_boot; chart_choice=Persistence(), block_size=10),
    test_op_bp_bootstrap(x, n_boot, 2; chart_choice=Persistence()),
    test_op_surrogate(x, RandomShuffle(), n_boot; chart_choice=Persistence(), rng=Xoshiro(1)),
    test_sop_bootstrap(img, n_boot, 1, 1; chart_choice=TauTilde()),
    test_sop_bootstrap(img, n_boot, 1, 1; chart_choice=Shannon()),
    test_sop_bp_bootstrap(img, n_boot, 2; chart_choice=TauTilde()),
    test_acf_bootstrap(x, n_boot, 1),
    test_sacf_bootstrap(img, n_boot, 1, 1),
    test_sacf_bp_bootstrap(img, n_boot, 2),
  ]

  for r in results
    sample = StatsOrdinalPatterns.null_sample(r)
    crit = StatsOrdinalPatterns.null_crit(r)
    tail = StatsOrdinalPatterns.rejection_tail(r)
    @test length(sample) == n_boot

    fig = plot(r; nbins=20)
    @test fig isa Makie.Figure
    ax = _only_axis(fig)
    @test ax.xlabel[] == "statistic"
    @test occursin("null", ax.title[])
    @test occursin("p = ", ax.subtitle[])
    @test length(_plots_of(Makie.Hist, ax)) == 1

    # regions and critical lines follow the rejection side of the result
    n_regions = tail === :two_sided ? 2 : 1
    @test length(_plots_of(Makie.VSpan, ax)) == n_regions
    vl = _plots_of(Makie.VLines, ax)
    @test length(vl) == 2
    @test vl[1][1][] == (tail === :two_sided ? [-crit, crit] : [crit])
    @test vl[2][1][] == [r.stat]
  end

  # Family-specific rejection sides
  @test StatsOrdinalPatterns.rejection_tail(results[1]) === :two_sided
  @test StatsOrdinalPatterns.rejection_tail(results[2]) === :lower
  @test StatsOrdinalPatterns.rejection_tail(results[3]) === :upper
  @test StatsOrdinalPatterns.rejection_tail(results[8]) === :upper   # SOP Shannon is rescaled
  @test StatsOrdinalPatterns.rejection_tail(results[10]) === :two_sided

  # The stored sample is the one the critical value was computed from
  r = results[3]
  @test StatsOrdinalPatterns._op_boot_crit(r.chart, r.boot_dist, 0.05) == r.boot_crit
  @test StatsOrdinalPatterns._op_boot_pval(r.chart, r.stat, r.boot_dist) == r.boot_pval
end
