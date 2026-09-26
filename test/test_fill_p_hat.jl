
# Tests for fill_p_hat! dispatch via stat_sop.
#
# The 3×3 data matrix and d1=d2=1 give four SOPs whose frequencies land
# at positions [2, 10, 15, 19] in the 24-element sop_freq vector
# (cross-checked in test_frequencies.jl):
#
#   Group s_1 = {1,3,8,11,14,17,22,24}  →  0 hits
#   Group s_2 = {2,5,7,9,16,18,20,23}   →  1 hit  (position 2)
#   Group s_3 = {4,6,10,12,13,15,19,21} →  3 hits (positions 10, 15, 19)
#
# m = n = 2  →  m·n = 4

const _DATA = Float64[9 1 2;
                      6 7 6;
                      9 4 9]

@testset "fill_p_hat! dispatch — TauTilde" begin

    stat, p_hat = stat_sop(_DATA, 1, 1; chart_choice=TauTilde())

    # TauTilde fills only p_hat[3] (from s_3)
    @test p_hat[1] ≈ 0.0
    @test p_hat[2] ≈ 0.0
    @test p_hat[3] ≈ 3 / 4

    # chart_stat_sop for TauTilde: p_hat[3] - 1/3
    @test stat ≈ 3 / 4 - 1 / 3

end

@testset "fill_p_hat! dispatch — Shannon" begin

    stat, p_hat = stat_sop(_DATA, 1, 1; chart_choice=Shannon(base=exp(1)))

    # Shannon fills all three groups
    @test p_hat[1] ≈ 0.0
    @test p_hat[2] ≈ 1 / 4
    @test p_hat[3] ≈ 3 / 4

    # Shannon entropy: −∑ p·log(p)  (zero terms skipped)
    expected_stat = -(1 / 4 * log(1 / 4) + 3 / 4 * log(3 / 4))
    @test stat ≈ expected_stat

end

@testset "fill_p_hat! dispatch — TauTilde vs Shannon differ" begin

    stat_tt, _ = stat_sop(_DATA, 1, 1; chart_choice=TauTilde())
    stat_sh, _ = stat_sop(_DATA, 1, 1; chart_choice=Shannon())

    @test stat_tt ≉ stat_sh

end
