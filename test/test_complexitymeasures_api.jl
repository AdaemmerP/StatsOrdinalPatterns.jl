# Agreement between this package and ComplexityMeasures.jl.
#
# ComplexityMeasures.jl estimates ordinal pattern probabilities independently of this
# package. These tests check that both give the same probabilities, that the statistics
# computed through `complexity` and `information` equal those of `stat_op` and
# `stat_sop`, and that every chart type is classified as ComplexityMeasures requires. The data are continuous, so there are no ties: ComplexityMeasures
# breaks ties at random, `stat_op` by position.

using ComplexityMeasures: ComplexityMeasures, complexity, information, probabilities,
  allprobabilities_and_outcomes, OrdinalPatterns, Probabilities, ComplexityEstimator,
  InformationMeasure

# Series with different dependence structures: white noise, a persistent AR(1), a
# random walk and heavy tailed white noise.
function _cm_series(rng)
  n = 400
  ar = zeros(n)
  for t in 2:n
    ar[t] = 0.6 * ar[t-1] + randn(rng)
  end
  return [randn(rng, n), ar, cumsum(randn(rng, n)), rand(rng, TDist(2), n)]
end

const _BANDT = (UpDownBalance(), Persistence(), RotationalAsymmetry(), UpDownScaling())

@testset "ComplexityMeasures: same ordinal pattern probabilities" begin
  for x in _cm_series(Xoshiro(1)), m in 2:4, d in 1:3
    p_cm = collect(first(allprobabilities_and_outcomes(OrdinalPatterns{m}(d), x)))
    p_sop = stat_op(x; chart_choice=DistanceToWhiteNoise(), m=m, d=d)[2]
    @test p_cm ≈ p_sop
  end
end

@testset "ComplexityMeasures: classification of the chart types" begin
  # An information measure must be a functional of any probability vector. None of the
  # chart types of this package is one; `Shannon` and `ShannonExtropy`, which the
  # package uses, are information measures of ComplexityMeasures.
  charts = [UpDownBalance(), Persistence(), RotationalAsymmetry(), UpDownScaling(),
    DistanceToWhiteNoise(), TauHat(), KappaHat(), TauTilde(), KappaTilde(),
    D_Chart(), KappaN(), KappaO(), KappaN1(), KappaN2(), KappaO1(), KappaO2()]
  for c in charts
    @test c isa ComplexityEstimator
    @test !(c isa InformationMeasure)
  end
  @test Shannon() isa InformationMeasure
  @test ShannonExtropy() isa InformationMeasure
  # no exported type of the package claims to be an information measure
  for name in names(StatsOrdinalPatterns)
    T = getfield(StatsOrdinalPatterns, name)
    T isa DataType && parentmodule(T) === StatsOrdinalPatterns && @test !(T <: InformationMeasure)
  end
end

@testset "ComplexityMeasures: complexity equals stat_op for the Bandt statistics" begin
  for x in _cm_series(Xoshiro(2)), c in _BANDT, d in 1:4
    p = probabilities(OrdinalPatterns{3}(d), x)
    @test complexity(c, p) ≈ stat_op(x; chart_choice=c, m=3, d=d)[1]
  end
  for x in _cm_series(Xoshiro(3)), c in _BANDT
    @test complexity(c, x) ≈ stat_op(x; chart_choice=c, m=3, d=1)[1]
  end
  # UpDownBalance is also defined for patterns of length 2
  for x in _cm_series(Xoshiro(4)), d in 1:3
    p = probabilities(OrdinalPatterns{2}(d), x)
    @test complexity(UpDownBalance(), p) ≈ stat_op(x; chart_choice=UpDownBalance(), m=2, d=d)[1]
  end
end

@testset "ComplexityMeasures: information and complexity equal stat_op for H, H_ex and Δ" begin
  for x in _cm_series(Xoshiro(5)), m in 2:4, d in 1:3, b in (exp(1), 2, 10)
    for C in (Shannon, ShannonExtropy)
      @test information(C(base=b), OrdinalPatterns{m}(d), x) ≈
            stat_op(x; chart_choice=C(base=b), m=m, d=d)[1]
    end
  end
  for x in _cm_series(Xoshiro(6)), m in 2:4, d in 1:3
    expected = stat_op(x; chart_choice=DistanceToWhiteNoise(), m=m, d=d)[1]
    @test complexity(DistanceToWhiteNoise(), probabilities(OrdinalPatterns{m}(d), x)) ≈ expected
  end
  for x in _cm_series(Xoshiro(9))
    @test complexity(DistanceToWhiteNoise(), x) ≈ stat_op(x; chart_choice=DistanceToWhiteNoise())[1]
  end
end

@testset "ComplexityMeasures: patterns that do not occur" begin
  # An increasing series only shows the pattern [1, 2, 3]; all other probabilities are
  # zero and must be filled in, which matters for the Δ statistic.
  x = collect(1.0:50.0)
  @test complexity(Persistence(), x) ≈ 2 / 3
  @test complexity(UpDownBalance(), x) ≈ 1
  @test complexity(DistanceToWhiteNoise(), x) ≈ stat_op(x; chart_choice=DistanceToWhiteNoise())[1]
end

@testset "ComplexityMeasures: complexity equals stat_sop for the SOP statistics" begin
  rng = Xoshiro(7)
  # white noise and a field with dependence along rows and columns
  fields = [rand(rng, 20, 25), cumsum(cumsum(randn(rng, 20, 25); dims=1); dims=2)]
  for X in fields, c in (TauHat(), KappaHat(), TauTilde(), KappaTilde())
    @test complexity(c, X) ≈ stat_sop(X, 1, 1; chart_choice=c)[1]
    @test complexity(c, view(X, 1:15, 1:20)) ≈ stat_sop(X[1:15, 1:20], 1, 1; chart_choice=c)[1]
  end
end

@testset "ComplexityMeasures: invalid input raises an error" begin
  x = randn(Xoshiro(8), 100)
  @test_throws ArgumentError complexity(Persistence(), probabilities(OrdinalPatterns{4}(1), x))
  @test_throws ArgumentError complexity(UpDownScaling(), probabilities(OrdinalPatterns{2}(1), x))
  @test_throws ArgumentError complexity(Persistence(), Probabilities([0.5, 0.5]))
  @test_throws ArgumentError complexity(DistanceToWhiteNoise(), Probabilities([0.5, 0.5]))
end
