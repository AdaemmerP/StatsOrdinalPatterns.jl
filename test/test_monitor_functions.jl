@testset "monitor_op" begin
  n, m, d, lam = 300, 3, 1, 0.1
  x = ar1_series(n, 0.6, 1)

  for chart in (Persistence(), DistanceToWhiteNoise(), Shannon())
    cl = chart isa Shannon ? 1.6 : 0.2
    res = monitor_op(x, lam, cl; chart_choice=chart, m=m, d=d)
    stats, _ = stat_op(x, lam; chart_choice=chart, m=m, d=d)

    @test res isa ControlChartResult
    @test res.family === :op
    @test res.chart === chart
    @test res.lam == lam
    @test res.cl == cl
    @test res.stats == stats
    # Statistic i covers observations i, …, i + (m - 1) d and is available at the last one.
    @test res.time == ((m - 1) * d + 1):n
    @test length(res.time) == length(res.stats)
    @test res.tail === StatsOrdinalPatterns._tail_op(chart)
    # The recorded alarm is the first violation of the chart's own alarm rule.
    expected = findfirst(s -> StatsOrdinalPatterns.abort_criterium_op(s, cl, chart), stats)
    @test res.alarm == expected
    if !isnothing(res.alarm)
      @test !any(s -> StatsOrdinalPatterns.abort_criterium_op(s, cl, chart), stats[1:res.alarm-1])
    end
  end

  # An unreachable limit gives no alarm; delays shift the time index.
  res = monitor_op(x, lam, 10.0; chart_choice=Persistence(), m=3, d=2)
  @test isnothing(res.alarm)
  @test first(res.time) == 5
  @test occursin("none", sprint(show, res))

  res = monitor_op(x, lam, 0.0; chart_choice=Persistence())
  @test res.alarm == 1
  @test occursin("First alarm", sprint(show, res))
end

@testset "monitor_sop" begin
  M, N, T, lam, cl = 11, 11, 30, 0.1, 0.03
  rng = MersenneTwister(2)
  images = randn(rng, M, N, T)

  res = monitor_sop(images, lam, cl, 1, 1; chart_choice=TauTilde())
  stats = stat_sop(images, lam, 1, 1; chart_choice=TauTilde())
  @test res isa ControlChartResult
  @test res.family === :sop
  @test res.stats == stats
  @test res.time == 1:T
  @test res.tail === :two_sided
  @test res.alarm == findfirst(s -> abs(s) > cl, stats)

  # Refined classifications: the EWMA vector must have one entry per refined type.
  for rf in (RotationType(), DirectionType(), DiagonalType())
    res_rf = monitor_sop(images, lam, cl, 1, 1; chart_choice=TauTilde(), refinement=rf)
    @test length(res_rf.stats) == T
    @test all(isfinite, res_rf.stats)
  end

  # The entropy charts are not monitoring statistics.
  @test_throws ArgumentError monitor_sop(images, lam, cl, 1, 1; chart_choice=Shannon())
end
