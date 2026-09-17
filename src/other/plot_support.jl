# Facts about the result types that the Makie extension needs and that belong to the
# statistics, not to the plotting: on which side a result rejects, which vector holds its
# resampled null distribution, and how the family is called. Keeping them here means the
# extension never re-derives a rule that `src` already encodes.
#
# Nothing in this file is exported.

# --- Side of the rejection region ---------------------------------------------------
# `:lower`: reject when stat < crit. `:upper`: stat > crit. `:two_sided`: abs(stat) > crit.
# The direction depends on the family as well as on the chart: `Shannon()` is lower tailed
# for OP but upper tailed for SOP, where the entropy statistics are rescaled first.
rejection_tail(r::OPTestResultBoot) = _tail_op(r.chart)
rejection_tail(r::OPTestResultSurrogate) = _tail_op(r.chart)
rejection_tail(::OPBPTestResultBoot) = :upper
rejection_tail(r::SOPTestResultBoot) =
  r.chart isa Union{TauHat,KappaHat,TauTilde,KappaTilde} ? :two_sided : :upper
rejection_tail(::SOPBPTestResultBoot) = :upper
rejection_tail(::ACFTestResultBoot) = :two_sided
rejection_tail(::SACFTestResultBoot) = :two_sided
rejection_tail(::SACFBPTestResultBoot) = :upper
rejection_tail(r::ControlChartResult) = r.tail

# --- Resampled null distribution and the values that go with it ----------------------
const _BootResult = Union{
  OPTestResultBoot, OPBPTestResultBoot, SOPTestResultBoot, SOPBPTestResultBoot,
  ACFTestResultBoot, SACFTestResultBoot, SACFBPTestResultBoot
}

null_sample(r::_BootResult) = r.boot_dist
null_sample(r::OPTestResultSurrogate) = r.surr_dist
null_crit(r::_BootResult) = r.boot_crit
null_crit(r::OPTestResultSurrogate) = r.surr_crit
null_pval(r::_BootResult) = r.boot_pval
null_pval(r::OPTestResultSurrogate) = r.surr_pval
null_reject(r::_BootResult) = r.boot_reject
null_reject(r::OPTestResultSurrogate) = r.surr_reject

# --- Labels ---------------------------------------------------------------------------
# One line naming the test, for plot titles.
null_label(r::OPTestResultBoot) = "Bootstrap null, OP $(r.chart), n_boot = $(r.n_boot)"
null_label(r::OPBPTestResultBoot) = "Bootstrap null, OP Box-Pierce $(r.chart), n_boot = $(r.n_boot)"
null_label(r::OPTestResultSurrogate) =
  "Surrogate null ($(r.method)), OP $(r.chart), n_surrogates = $(r.n_surrogates)"
null_label(r::SOPTestResultBoot) = "Bootstrap null, SOP $(r.chart), n_boot = $(r.n_boot)"
null_label(r::SOPBPTestResultBoot) = "Bootstrap null, SOP Box-Pierce $(r.chart), n_boot = $(r.n_boot)"
null_label(r::ACFTestResultBoot) = "Bootstrap null, ACF, n_boot = $(r.n_boot)"
null_label(r::SACFTestResultBoot) = "Bootstrap null, SACF, n_boot = $(r.n_boot)"
null_label(r::SACFBPTestResultBoot) = "Bootstrap null, SACF Box-Pierce, n_boot = $(r.n_boot)"
