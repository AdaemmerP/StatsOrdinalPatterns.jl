module StatsOrdinalPatterns

# Packages to use
using Combinatorics
using ComplexityMeasures: InformationMeasure, ComplexityEstimator, Entropy, Shannon, ShannonExtropy
using Distributions
using LinearAlgebra
using Random
using Reexport
using Roots
using Statistics
using StatsBase
import PrecompileTools

# Reexport
# Distributions is re-exported in full — deliberately, not by oversight. The DGP types
# (`AR1`, `INAR1`, `SAR11`, …) take a distribution object as a constructor argument, so
# essentially no non-trivial call can be written without one; requiring a separate
# `using Distributions` would only add a line to every example. From ComplexityMeasures
# only the two chart types actually used as `chart_choice` values are re-exported.
@reexport using Distributions
@reexport using ComplexityMeasures: Shannon, ShannonExtropy

# ---------------------------------------------#
# OP related functions and structs  to export  #
# ---------------------------------------------#

# op/op_dgp_structs.jl
export AAR1,
  AR1,
  BAR1,
  ContinuousDGPIC,
  DAR1,
  DiscreteDGPIC,
  INAR1,
  MA1,
  MA2,
  QAR1,
  SINAR1,
  TEAR1,
  TINAR1,
  WDAR1

# op/op_information_measures.jl
export chart_stat_op,
  DistanceToWhiteNoise,
  Persistence,
  RotationalAsymmetry,
  UpDownBalance,
  UpDownScaling

# op/op_stat_bp_functions.jl
export crit_val_op_bp,
  stat_op_bp

# op/op_test_bp_functions.jl
export test_op_bp,
  OPBPTestResult

# op/op_bp_bootstrap_functions.jl
export bootstrap_op_bp, test_op_bp_bootstrap, OPBPTestResultBoot

# op/op_arl_ic_functions.jl, op_arl_oc_functions.jl, op_cl_functions.jl,
# op_dependence.jl, op_stat_functions.jl, op_test_functions.jl
export arl_op_ic,
  arl_op_oc,
  changepoint_op,
  cl_op,
  count_mv_op,
  count_uv_op,
  crit_val_op,
  dependence_op,
  rl_op_ic,
  rl_op_oc,
  stat_op,
  test_op,
  OPTestResult

# op/op_bootstrap_functions.jl
export bootstrap_op, test_op_bootstrap, OPTestResultBoot

# op/op_surrogate_functions.jl (implementation in ext/StatsOrdinalPatternsTimeseriesSurrogatesExt.jl)
export test_op_surrogate, OPTestResultSurrogate

# ---------------------------------------------#
# ACF related functions and structs to export  #
# ---------------------------------------------#

# acf/acf_arl_ic_functions.jl
export arl_acf_ic,
  rl_acf_ic

# acf/acf_arl_oc_functions.jl
export arl_acf_oc

# acf/acf_cl_functions.jl
export cl_acf

# acf/acf_stat_functions.jl
export stat_acf

# acf/acf_test_functions.jl
export crit_val_acf,
  test_acf,
  ACFTestResult

# acf/acf_bootstrap_functions.jl
export bootstrap_acf, test_acf_bootstrap, ACFTestResultBoot
# ----------------------------------------------#
#  GOP related functions and structs to export  #
# ----------------------------------------------#

# gop/gop_information_measures.jl
export chart_stat_gop,
  D_Chart

# gop/gop_arl_ic_functions.jl
export arl_gop_ic,
  rl_gop_ic

# gop/gop_arl_oc_functions.jl
export arl_gop_oc,
  rl_gop_oc

# gop/gop_cl_functions.jl
export cl_gop

# gop/gop_help_functions.jl
# Required as a positional argument of the exported `rl_gop_ic` / `rl_gop_oc`.
export compute_lookup_array_gop

# gop/gop_stat_functions.jl
export stat_gop

# gop/gop_bootstrap_functions.jl
export test_gop_bootstrap

# ---------------------------------------------#
# SOP related functions and structs to export  #
# ---------------------------------------------#

# sop/sop_information_measures.jl
export chart_stat_sop,
  DiagonalType,
  DirectionType,
  KappaHat,
  KappaTilde,
  OrdinaryType,
  RefinedType,
  RotationType,
  SOPClassification,
  TauHat,
  TauTilde

# sop/sop_dgp_structs.jl
export BSQMA11,
  ICSTS,
  SAR1,
  SAR11,
  SAR22,
  SINAR11,
  SQINMA11,
  SQMA11,
  SQMA22

# sop/sop_distributions.jl
export BinNorm,
  BinomialC,
  BinomialCvec,
  PoiBin,
  ZIP

# sop/sop_help_functions.jl
# Required to build the `p_mat` / `p_array` arguments of the exported
# `arl_sop_bootstrap` / `arl_sop_bp_bootstrap`.
export compute_p_array,
  compute_p_array_bp

# sop/sop_arl_ic_functions.jl
export arl_sop_ic

# sop/sop_arl_oc_functions.jl
export arl_sop_oc

# sop/sop_arl_bootstrap_functions.jl
export arl_sop_bootstrap

# sop/sop_bp_arl_ic_functions.jl
export arl_sop_bp_ic

# sop/sop_bp_arl_oc_functions.jl
export arl_sop_bp_oc

# sop/sop_bp_arl_bootstrap_functions.jl
export arl_sop_bp_bootstrap

# sop/sop_cl_functions.jl
export cl_sop

# sop/sop_cl_bootstrap_functions.jl
export cl_sop_bootstrap

# sop/sop_bp_cl_functions.jl
export cl_sop_bp

# sop/sop_bp_cl_bootstrap_functions.jl
export cl_sop_bp_bootstrap

# sop/sop_stat_functions.jl
export stat_sop

# sop/sop_stat_bp_functions.jl
export stat_sop_bp

# sop/sop_bootstrap_functions.jl
export bootstrap_sop, test_sop_bootstrap, SOPTestResultBoot

# sop/sop_bp_bootstrap_functions.jl
export bootstrap_sop_bp, test_sop_bp_bootstrap, SOPBPTestResultBoot

# sop/sop_test_functions.jl
export crit_val_sop,
  test_sop,
  SOPTestResult

# ---------------------------------------------#
# SACF related functions and structs to export #
# ---------------------------------------------#

# sacf/sacf_arl_ic_functions.jl
export arl_sacf_ic

# sacf/sacf_arl_oc_functions.jl
export arl_sacf_oc

# sacf/sacf_bp_arl_ic_functions.jl
export arl_sacf_bp_ic

# sacf/sacf_bp_arl_oc_functions.jl
export arl_sacf_bp_oc

# sacf/sacf_cl_functions.jl
export cl_sacf

# sacf/sacf_cl_bp_functions.jl
export cl_sacf_bp

# sacf/sacf_stat_functions.jl
export crit_val_sacf,
  sacf,
  stat_sacf

# sacf/sacf_stat_bp_functions.jl
export stat_sacf_bp

# sacf/sacf_bootstrap_functions.jl
export bootstrap_sacf, bootstrap_sacf_bp,
  test_sacf_bootstrap, test_sacf_bp_bootstrap,
  SACFTestResultBoot, SACFBPTestResultBoot

# sacf/sacf_test_functions.jl
export crit_val_sacf_bp,
  test_sacf,
  test_sacf_bp,
  SACFTestResult,
  SACFBPTestResult

# ---------------------------------------------#
# Kappa related functions and structs to export#
# ---------------------------------------------#

# kappa_procs/kappa_information_measures.jl
export chart_stat_qual,
  KappaN,
  KappaN1,
  KappaN2,
  KappaO,
  KappaO1,
  KappaO2

# kappa_procs/kappa_arl_ic_functions.jl
export arl_kappa_ic,
  rl_kappa_ic

# kappa_procs/kappa_arl_oc_functions.jl
export arl_kappa_oc,
  rl_kappa_oc

# kappa_procs/kappa_cl_functions.jl
export cl_kappa

# kappa_procs/kappa_stat_functions.jl
export stat_kappa


# -----------------------------------------------#
#  Vendored third-party code (not part of the API)#
# -----------------------------------------------#
# Verbatim copy of GeneralizedChisqDistribution.jl @ revise-computation
# (commit 665424e920927a4513ac6760807e16738713eb61); see src/vendor/README.md.
# Defines the submodule `GeneralizedChisqDistribution`. Nothing from it is
# exported: it is reached only through the internal alias below, which keeps the
# name `GeneralizedChisq` from ever colliding with a package StatsOrdinalPatterns loads.
include("vendor/GeneralizedChisqDistribution.jl")

const _GChisqDist = GeneralizedChisqDistribution.GeneralizedChisq

# -----------------------------------------------#
#  General helper functions and structs to include#
# -----------------------------------------------#
include("algorithms_and_types/lehmer_function.jl")

# ---------------------------------------------#
#  OP related functions and structs to include #
# ---------------------------------------------#
include("op/op_dgp_structs.jl")
include("op/op_dgp_functions.jl")
include("op/op_information_measures.jl")
include("op/op_arl_ic_functions.jl")
include("op/op_arl_oc_functions.jl")
include("op/op_cl_functions.jl")
include("op/op_dependence.jl")
include("op/op_stat_functions.jl")
include("op/op_stat_bp_functions.jl")
include("op/op_test_functions.jl")
# after op_test_functions.jl: reuses the `_gc_op` null distribution defined there
include("op/op_test_bp_functions.jl")
include("op/op_help_functions.jl")
include("op/op_bootstrap_functions.jl")
include("op/op_bp_bootstrap_functions.jl")
include("op/op_surrogate_functions.jl")

# ACF files
include("acf/acf_arl_ic_functions.jl")
include("acf/acf_arl_oc_functions.jl")
include("acf/acf_cl_functions.jl")
include("acf/acf_stat_functions.jl")
include("acf/acf_test_functions.jl")
include("acf/acf_bootstrap_functions.jl")

# ----------------------------------------------#
#  GOP related functions and structs to include #
# ----------------------------------------------#
include("gop/gop_information_measures.jl")
include("gop/gop_arl_ic_functions.jl")
include("gop/gop_arl_oc_functions.jl")
include("gop/gop_cl_functions.jl")
include("gop/gop_help_functions.jl")
include("gop/gop_stat_functions.jl")
include("gop/gop_bootstrap_functions.jl")

# ---------------------------------------------#
# SOP related functions and structs to include #
# ---------------------------------------------#
include("sop/sop_information_measures.jl")
include("sop/sop_dgp_structs.jl")
include("sop/sop_dgp_functions.jl")
include("sop/sop_distributions.jl")
include("sop/sop_arl_ic_functions.jl")
include("sop/sop_arl_oc_functions.jl")
include("sop/sop_arl_bootstrap_functions.jl")
include("sop/sop_help_functions.jl")
# ---
include("sop/sop_bp_arl_ic_functions.jl")
include("sop/sop_bp_arl_oc_functions.jl")
include("sop/sop_bp_arl_bootstrap_functions.jl")
# ---
include("sop/sop_cl_functions.jl")
include("sop/sop_cl_bootstrap_functions.jl")
include("sop/sop_bp_cl_functions.jl")
include("sop/sop_bp_cl_bootstrap_functions.jl")
# ---
include("sop/sop_stat_functions.jl")
include("sop/sop_stat_bp_functions.jl")
include("sop/sop_bootstrap_functions.jl")
include("sop/sop_bp_bootstrap_functions.jl")
include("sop/sop_test_functions.jl")

# SACFs
include("sacf/sacf_arl_ic_functions.jl")
include("sacf/sacf_arl_oc_functions.jl")
include("sacf/sacf_bp_arl_ic_functions.jl")
include("sacf/sacf_bp_arl_oc_functions.jl")
include("sacf/sacf_cl_functions.jl")
include("sacf/sacf_cl_bp_functions.jl")
include("sacf/sacf_stat_functions.jl")
include("sacf/sacf_stat_bp_functions.jl")
include("sacf/sacf_bootstrap_functions.jl")
include("sacf/sacf_test_functions.jl")

# Kappa scripts
include("kappa_procs/kappa_information_measures.jl")
include("kappa_procs/kappa_arl_ic_functions.jl")
include("kappa_procs/kappa_arl_oc_functions.jl")
include("kappa_procs/kappa_cl_functions.jl")
include("kappa_procs/kappa_stat_functions.jl")


# Precompile
include("other/precompile.jl")

end
