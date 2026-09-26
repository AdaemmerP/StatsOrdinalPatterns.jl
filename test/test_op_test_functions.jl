using Distributions

@testset "qup3_value" begin
  @test StatsOrdinalPatterns.qup3_op_value(0.01) ≈ 2.2672 atol = 1e-4
  @test StatsOrdinalPatterns.qup3_op_value(0.05) ≈ 1.4842 atol = 1e-4
  @test StatsOrdinalPatterns.qup3_op_value(0.10) ≈ 1.1626 atol = 1e-4
end

# ── Regression: the m = 2 Δ-chart scaling ────────────────────────────────────
# With m = 2 the only estimated quantity is the relative frequency p̂ of up-steps, and
# Δ = 2(p̂ − 1/2)². Consecutive comparisons share an observation, so under H₀
# Var(p̂) = 1/(12·n) rather than 1/(4·n), which gives 6·n·Δ ~ Chisq(1).
#
# The factor 6 was previously missing from crit_val_op / _asymp_pval for m = 2, making the
# critical value six times too large; since Δ is a squared quantity the test then never
# rejected at all (measured size 0.0000 over 4000 replications at a nominal 5%).
@testset "crit_val_op — m=2 Δ-chart uses the correct 1/(12n) variance" begin
  n = 1000
  q = quantile(Chisq(1), 0.95)

  @test StatsOrdinalPatterns.crit_val_op(DistanceToWhiteNoise(), 2, n; alpha=0.05) ≈ q / (6 * n)

  # Deterministic cross-check tying the two m = 2 charts together: a Taylor expansion
  # gives log(2) − H ≈ Δ for m = 2, so the Shannon chart's distance from its maximum must
  # equal the Δ-chart's critical value. This only holds when both carry the factor 6.
  @test log(2) - StatsOrdinalPatterns.crit_val_op(Shannon(base=exp(1)), 2, n; alpha=0.05) ≈
        StatsOrdinalPatterns.crit_val_op(DistanceToWhiteNoise(), 2, n; alpha=0.05)

  # The m = 3 case is unaffected by the fix.
  @test StatsOrdinalPatterns.crit_val_op(DistanceToWhiteNoise(), 3, n; alpha=0.05) ≈
        StatsOrdinalPatterns.qup3_op_value(0.05) / n
end

@testset "_asymp_pval — p-value equals alpha at the critical value" begin
  # Pins the scaling used in _asymp_pval to the one used in crit_val_op. Deterministic.
  n = 1000
  for alpha in (0.01, 0.05, 0.10)
    cv = StatsOrdinalPatterns.crit_val_op(DistanceToWhiteNoise(), 2, n; alpha=alpha)
    @test StatsOrdinalPatterns._asymp_pval(DistanceToWhiteNoise(), cv, n, 2) ≈ alpha atol = 1e-8
  end

  # Same invariant for the charts that were already correct, as a guard against
  # a similar scaling slip elsewhere.
  for (chart, m) in ((Shannon(), 2), (Shannon(), 3), (DistanceToWhiteNoise(), 3),
                     (UpDownBalance(), 3), (Persistence(), 3), (UpDownScaling(), 3))
    cv = StatsOrdinalPatterns.crit_val_op(chart, m, n; alpha=0.05)
    @test StatsOrdinalPatterns._asymp_pval(chart, cv, n, m) ≈ 0.05 atol = 1e-4
  end
end

@testset "test_op — m=2 charts are equivalent (holds for every draw)" begin
  # At m = 2 every chart is a function of p̂ alone. For Δ and UpDownBalance the two
  # rejection regions are algebraically identical:
  #   Δ > q/(6n)             ⟺ |p̂ − 1/2| > sqrt(q/(12n))
  #   |β| > z·sqrt(1/(3n))   ⟺ |p̂ − 1/2| > z/sqrt(12n),   and z² = q.
  # This is the sharpest regression test for the bug: before the fix Δ never rejected
  # while UpDownBalance did, so the identity was violated on any rejecting data set.
  agree_beta = rejection_rate(s -> begin
      x = isodd(s) ? tar1_series(300, s) : randn(MersenneTwister(s), 300)
      test_op(x; chart_choice=DistanceToWhiteNoise(), m=2).asymp_reject ==
      test_op(x; chart_choice=UpDownBalance(), m=2).asymp_reject
    end, 300)
  @test agree_beta == 1.0

  # Shannon is only asymptotically equivalent, so allow a small discrepancy rate.
  agree_shannon = rejection_rate(s -> begin
      x = isodd(s) ? tar1_series(300, s) : randn(MersenneTwister(s), 300)
      test_op(x; chart_choice=DistanceToWhiteNoise(), m=2).asymp_reject ==
      test_op(x; chart_choice=Shannon(), m=2).asymp_reject
    end, 300)
  @test agree_shannon > 0.98
end

@testset "test_op — empirical size at m=2 is close to nominal alpha" begin
  # Behavioural consequence of the fix: before it, the Δ-chart rejected in 0 of 4000
  # replications. Aggregate check, so it does not depend on any single draw.
  for chart in (Shannon(), UpDownBalance(), DistanceToWhiteNoise())
    size_hat = rejection_rate(
      s -> test_op(randn(MersenneTwister(s), 1000); chart_choice=chart, m=2).asymp_reject,
      SIZE_REPS
    )
    @test SIZE_LOWER < size_hat < SIZE_UPPER
  end
end

@testset "test_op — m=2 charts detect up/down imbalance, not general dependence" begin
  # m = 2 patterns only see the up-step frequency p̂. A Gaussian AR(1) is time-reversible,
  # so p̂ stays at 1/2 and no m = 2 chart has power against it — the rejection rate stays
  # at the nominal level. The asymmetric threshold AR(1) does shift p̂ and is detected.
  power_ar = rejection_rate(
    s -> test_op(ar1_series(500, 0.9, s); chart_choice=DistanceToWhiteNoise(), m=2).asymp_reject,
    200
  )
  @test power_ar < 0.15

  # Power against the threshold AR(1) verified to be 1.0 over several hundred seeds.
  res = test_op(tar1_series(2000, 2026); chart_choice=DistanceToWhiteNoise(), m=2)
  @test res.asymp_reject
  @test res.asymp_pval < 0.05

  power_tar = rejection_rate(
    s -> test_op(tar1_series(2000, s); chart_choice=DistanceToWhiteNoise(), m=2).asymp_reject,
    100
  )
  @test power_tar > 0.95
end
