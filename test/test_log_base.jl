# The logarithm base of `Shannon` and `ShannonExtropy`.
#
# The asymptotic theory is derived for the natural logarithm. Raw entropies (statistic,
# critical value, control limit) are reported in the base of the chart, standardized
# statistics (Box-Pierce, rescaled SOP statistic) do not depend on the base, and test
# decisions and p-values never depend on it. Bases outside (1, ∞) must raise an error,
# because a base in (0, 1) would silently flip the direction of the lower sided tests.

using ComplexityMeasures: information, Probabilities

const _BASES = (2, 10, 3.7)

@testset "log base: chart statistic matches ComplexityMeasures" begin
  p = [0.30, 0.20, 0.15, 0.15, 0.10, 0.10]
  for C in (Shannon, ShannonExtropy), b in (exp(1), _BASES...)
    @test chart_stat_op(p, C(base=b)) ≈ information(C(base=b), Probabilities(p))
    @test chart_stat_sop(p, C(base=b)) ≈ information(C(base=b), Probabilities(p))
  end
end

@testset "log base: raw statistics scale with 1 / log(base)" begin
  x = randn(Xoshiro(1), 200)
  img = rand(Xoshiro(2), 15, 15)
  for C in (Shannon, ShannonExtropy), b in _BASES
    @test stat_op(x; chart_choice=C(base=b))[1] ≈
          stat_op(x; chart_choice=C(base=exp(1)))[1] / log(b)
    @test stat_op(x, 0.1; chart_choice=C(base=b))[1] ≈
          stat_op(x, 0.1; chart_choice=C(base=exp(1)))[1] ./ log(b)
    @test stat_sop(img, 1, 1; chart_choice=C(base=b))[1] ≈
          stat_sop(img, 1, 1; chart_choice=C(base=exp(1)))[1] / log(b)
  end
end

@testset "log base: test decisions and p-values do not depend on the base" begin
  # A strongly dependent series (random walk), so that the tests reject and the
  # comparison is not trivial.
  x = cumsum(randn(Xoshiro(3), 300))
  img = rand(Xoshiro(4), 15, 15)

  for C in (Shannon, ShannonExtropy)
    ref = test_op(x; chart_choice=C(base=exp(1)))
    ref_bp = test_op_bp(x, 2; chart_choice=C(base=exp(1)))
    ref_sop = test_sop(img, 1, 1; chart_choice=C(base=exp(1)))
    for b in _BASES
      r = test_op(x; chart_choice=C(base=b))
      @test r.stat ≈ ref.stat / log(b)
      @test r.asymp_crit ≈ ref.asymp_crit / log(b)
      @test r.asymp_pval ≈ ref.asymp_pval
      @test r.asymp_reject == ref.asymp_reject

      # Standardized statistics are identical across bases.
      r_bp = test_op_bp(x, 2; chart_choice=C(base=b))
      @test r_bp.stat ≈ ref_bp.stat
      @test r_bp.asymp_crit == ref_bp.asymp_crit
      @test r_bp.asymp_reject == ref_bp.asymp_reject

      r_sop = test_sop(img, 1, 1; chart_choice=C(base=b))
      @test r_sop.stat ≈ ref_sop.stat
      @test r_sop.asymp_pval ≈ ref_sop.asymp_pval
      @test r_sop.asymp_reject == ref_sop.asymp_reject
    end
  end
  @test test_op(x; chart_choice=Shannon(base=exp(1))).asymp_reject

  # Bootstrap: the same resamples give a bootstrap distribution on the scale of the base.
  Random.seed!(5)
  ref = test_op_bootstrap(x, 200; chart_choice=Shannon(base=exp(1)))
  Random.seed!(5)
  r = test_op_bootstrap(x, 200; chart_choice=Shannon(base=2))
  @test r.boot_crit ≈ ref.boot_crit / log(2)
  @test r.boot_pval == ref.boot_pval
  @test r.boot_reject == ref.boot_reject
end

@testset "log base: bases outside (1, ∞) raise an error" begin
  x = randn(Xoshiro(6), 100)
  img = rand(Xoshiro(7), 10, 10)
  for C in (Shannon, ShannonExtropy), b in (1, 0.5, 0, -2)
    chart = C(base=b)
    @test_throws ArgumentError stat_op(x; chart_choice=chart)
    @test_throws ArgumentError test_op(x; chart_choice=chart)
    @test_throws ArgumentError crit_val_op(chart, 3, 98)
    @test_throws ArgumentError test_op_bp(x, 2; chart_choice=chart)
    @test_throws ArgumentError stat_sop(img, 1, 1; chart_choice=chart)
    @test_throws ArgumentError test_sop(img, 1, 1; chart_choice=chart)
  end
end
