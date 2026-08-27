#' steinsampling: Stein tests and Stein sampling tools
#'
#' Score-based goodness-of-fit tests and Stein sampling tools: KSD and FSSD
#' tests, Stein thinning, Stein Points, SP-MCMC, SVGD, and reusable Stein
#' kernels.
#'
#' @details
#' Most functions use the target score
#' \deqn{s_p(x) = \nabla_x \log p(x).}
#' Because this is the log-density gradient, many tests and samplers do not
#' require the normalizing constant of `p(x)`.
#'
#' The public API is layered. Complete methods occupy the top layer:
#' [ksd_u_test()], [ksd_v_test()], [fssd_test()], [stein_thinning()],
#' [stein_points()], [sp_mcmc()], and [svgd()]. The next layer exposes their
#' reusable numerical pieces:
#' [stein_kernel_matrix()], [ksd_uq_matrix()], [ksd_u_statistic()],
#' [ksd_u_bootstrap()], [ksd_vq_matrix()], [ksd_v_statistic()],
#' [ksd_v_bootstrap()], [compute_tau()], [fssd_statistic()], and
#' [fssd_null_pvalue()]. They expose intermediate matrices and reusable
#' calibration components.
#' All public GOF statistic primitives return the scale used for null
#' comparison: `n U_n`, `n V_n`, or `n FSSDhat^2`. Their bootstrap or simulated
#' null draws use exactly the matching scale; unscaled U-statistics are private
#' implementation details used only where an optimization objective requires
#' them.
#'
#' For testing, [ksd_u_test()] implements the Liu, Lee, and Jordan
#' independent-sample KSD test; its lower-level path is [ksd_uq_matrix()], the
#' off-diagonal [ksd_u_statistic()], and centered-multinomial
#' [ksd_u_bootstrap()]. [ksd_v_test()] keeps the same sample, score, and kernel
#' layer but includes the diagonal and uses Rademacher or Markov signs through
#' [ksd_vq_matrix()], [ksd_v_statistic()], and [ksd_v_bootstrap()].
#' [fssd_test()] instead replaces the full pairwise matrix with [compute_tau()],
#' finite Stein features evaluated at test locations.
#'
#' The sampling and compression methods reuse this kernel layer differently.
#' [stein_thinning()] returns row indices that compress an existing sample,
#' usually MCMC output. [stein_points()] constructs points by searching
#' continuous candidates with [fmin_grid()], [fmin_mc()], or [fmin_nm()], while
#' [sp_mcmc()] uses short MCMC paths as finite candidate sets. [svgd()] instead
#' moves every particle by deterministic transport.
#'
#' [stein_kernel()] and [custom_stein_kernel()] create kernel objects.
#' [eval_kernel()], [grad_x_kernel()], [trace_mixed_kernel()],
#' [cross_kernel()], and [grad_theta_v_kernel()] expose the pieces needed to
#' assemble scalar Stein kernels `k0` and finite-location FSSD features. Most
#' users need only the top-level algorithms, but custom kernels and diagnostics
#' use the same public interface as built-ins.
#'
#' Finally, the example-model layer provides [gmm()] construction, [rgmm()]
#' simulation, [densitygmm()] evaluation, and [get_score_evaluator()] to create
#' the `function(X)` score callback expected by Stein routines.
"_PACKAGE"
