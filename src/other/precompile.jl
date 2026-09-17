# Precompile the user-facing entry points of every family (OP, ACF, GOP, SOP,
# SACF, kappa). Inputs are deliberately tiny: only the compiled signatures are
# cached, the results are discarded.
#
# Names here must not collide with anything StatsOrdinalPatterns imports (`counts`, `sample`,
# `params`, ... from StatsBase/Distributions). PrecompileTools < 1.3 expands the
# workload at module scope, where assigning to an imported binding is an error on
# Julia 1.10 — so a colliding name breaks precompilation on the oldest supported
# release only.

PrecompileTools.@setup_workload begin

  lam = 0.1
  # The ARL functions split `reps` into `4 * nthreads()` chunks and require more
  # repetitions than chunks.
  reps = 8 * Threads.nthreads()
  n_boot = 10
  rl_max = 20
  L0 = 20

  ts = randn(120)
  ts2 = randn(120)
  count_ts = Float64.(rand(Poisson(5), 120))
  img = randn(11, 11)
  imgs = randn(11, 11, 3)

  # In-control and out-of-control DGPs
  ic_cont = ContinuousDGPIC(Normal(0, 1))
  ic_disc = DiscreteDGPIC(Poisson(5), false)
  ic_spat = ICSTS(11, 11, Normal(0, 1))
  ar1 = AR1(0.5, Normal(0, 1))
  inar1 = INAR1(0.5, Poisson(2), false)
  sar11 = SAR11((0.4, 0.3, 0.2), 11, 11, Normal(0, 1), nothing, 20)

  op_charts = (Shannon(), ShannonExtropy(), DistanceToWhiteNoise(), UpDownBalance())
  sop_charts = (TauTilde(), KappaTilde(), TauHat(), KappaHat())
  kappa_charts = (KappaN1(), KappaN2(), KappaO1(), KappaO2())

  PrecompileTools.@compile_workload begin

    # --- Ordinal patterns (OP) ---
    for cc in op_charts
      stat_op(ts; chart_choice=cc)
      stat_op(ts, lam; chart_choice=cc)
      test_op(ts; chart_choice=cc)
      stat_op_bp(ts, 3; chart_choice=cc)
      test_op_bp(ts, 3; chart_choice=cc)
      arl_op_ic(ic_cont, lam, 1.5, reps; chart_choice=cc, rl_max=rl_max)
      arl_op_oc(ar1, lam, 1.5, reps; chart_choice=cc, rl_max=rl_max)
      monitor_op(ts, lam, 1.5; chart_choice=cc)
    end
    test_op_bootstrap(ts, n_boot; chart_choice=Shannon())
    test_op_bp_bootstrap(ts, n_boot, 3; chart_choice=Shannon())
    cl_op(ic_cont, lam, L0, 1.57; chart_choice=Shannon(),
      reps_final=reps, reps_bracket=reps, bracket_step=0.1)
    dependence_op(ts, ts2)
    changepoint_op(ts, ts2)

    # --- Autocorrelation (ACF) ---
    stat_acf(ts, 1)
    test_acf(ts, 1)
    test_acf_bootstrap(ts, n_boot, 1)
    arl_acf_ic(lam, 0.3, ic_cont, reps, 1; rl_max=rl_max)
    arl_acf_oc(lam, 0.3, ar1, Normal(0, 1), reps, 1; rl_max=rl_max)
    cl_acf(ic_cont, lam, L0, 0.253;
      reps_final=reps, reps_bracket=reps, bracket_step=0.15)

    # --- Generalized ordinal patterns (GOP) ---
    for cc in (D_Chart(), Persistence())
      stat_gop(count_ts, Poisson(5), cc)
      stat_gop(count_ts, Poisson(5), lam, cc)
      arl_gop_ic(ic_disc, lam, 0.1, reps; chart_choice=cc, rl_max=rl_max)
      arl_gop_oc(inar1, Poisson(5), lam, 0.1, reps; chart_choice=cc, rl_max=rl_max)
    end
    test_gop_bootstrap(count_ts, n_boot, Poisson(5), lam; chart_choice=D_Chart())
    cl_gop(ic_disc, lam, L0, 0.057; chart_choice=D_Chart(),
      reps_final=reps, reps_bracket=reps, bracket_step=0.03)

    # --- Spatial ordinal patterns (SOP) ---
    for cc in sop_charts
      stat_sop(img, 1, 1; chart_choice=cc)
      stat_sop(imgs, lam, 1, 1; chart_choice=cc)
      stat_sop_bp(img, 3; chart_choice=cc)
      stat_sop_bp(imgs, lam, 3; chart_choice=cc)
      test_sop(img, 1, 1; chart_choice=cc)
      monitor_sop(imgs, lam, 0.02, 1, 1; chart_choice=cc)
      arl_sop_ic(ic_spat, lam, 0.02, 1, 1, reps; chart_choice=cc, rl_max=rl_max)
      arl_sop_oc(sar11, lam, 0.02, 1, 1, reps; chart_choice=cc, rl_max=rl_max)
    end
    # Refined SOP classification
    for rf in (RotationType(), DirectionType(), DiagonalType())
      stat_sop(img, 1, 1; chart_choice=Shannon(), refinement=rf)
      test_sop(img, 1, 1; chart_choice=Shannon(), refinement=rf)
      arl_sop_ic(ic_spat, lam, 0.02, 1, 1, reps;
        chart_choice=TauTilde(), refinement=rf, rl_max=rl_max)
    end
    # Box-Pierce ARLs
    arl_sop_bp_ic(ic_spat, lam, 0.001, 3, reps; rl_max=rl_max)
    arl_sop_bp_oc(sar11, lam, 0.001, 3, reps; rl_max=rl_max)
    # Tests and control limits
    test_sop_bootstrap(img, n_boot, 1, 1)
    test_sop_bp_bootstrap(img, n_boot, 3)
    cl_sop(ic_spat, lam, L0, 0.018, 1, 1;
      reps_final=reps, reps_bracket=reps, bracket_step=0.008)
    cl_sop_bp(ic_spat, lam, L0, 0.0007, 3;
      reps_final=reps, reps_bracket=reps, bracket_step=0.0004)
    # Bootstrap ARL path (p_mat / p_array are the user-supplied inputs)
    p_mat = compute_p_array(imgs, 1, 1)
    arl_sop_bootstrap(p_mat, lam, 0.02, reps; rl_max=rl_max)
    p_array = compute_p_array_bp(imgs, 3; chart_choice=TauTilde(), refinement=OrdinaryType())
    arl_sop_bp_bootstrap(p_array, lam, 0.001, 3, reps; rl_max=rl_max)

    # --- Spatial autocorrelation (SACF) ---
    stat_sacf(img, 1, 1)
    stat_sacf(imgs, lam, 1, 1)
    stat_sacf_bp(img, 3)
    stat_sacf_bp(imgs, lam, 3)
    test_sacf(img, 1, 1)
    test_sacf_bp(img, 3)
    test_sacf_bootstrap(img, n_boot, 1, 1)
    test_sacf_bp_bootstrap(img, n_boot, 3)
    arl_sacf_ic(ic_spat, lam, 0.03, 1, 1, reps; rl_max=rl_max)
    arl_sacf_oc(sar11, lam, 0.03, 1, 1, reps; rl_max=rl_max)
    arl_sacf_bp_ic(ic_spat, lam, 0.018, 3, reps; rl_max=rl_max)
    arl_sacf_bp_oc(sar11, lam, 0.018, 3, reps; rl_max=rl_max)
    cl_sacf(ic_spat, lam, L0, 0.028, 1, 1;
      reps_final=reps, reps_bracket=reps, bracket_step=0.016)
    cl_sacf_bp(ic_spat, lam, L0, 0.0175, 3;
      reps_final=reps, reps_bracket=reps, bracket_step=0.01)

    # --- Kappa charts for qualitative processes ---
    for cc in kappa_charts
      stat_kappa(count_ts, lam, Poisson(5), cc)
      arl_kappa_ic(ic_disc, lam, 0.1, reps; chart_choice=cc, rl_max=rl_max)
      arl_kappa_oc(inar1, Poisson(5), lam, 0.1, reps; chart_choice=cc, rl_max=rl_max)
    end
    cl_kappa(ic_disc, lam, L0, 0.106; chart_choice=KappaN1(),
      reps_final=reps, reps_bracket=reps, bracket_step=0.09)

  end
end
