#' steinsampling: Stein tests and Stein sampling tools
#'
#' Score-based goodness-of-fit tests and Stein sampling tools: KSD and FSSD
#' tests, Stein thinning, Stein Points, SP-MCMC, SVGD, and reusable Stein
#' kernels.
#'
#' @details
#' Many targets are known only through an unnormalized density. Stein methods
#' use the target score
#' \deqn{s_p(x) = \nabla_x \log p(x)}
#' to test whether data match the target and to transport, construct, or select
#' representative points. The normalizing constant of `p(x)` is not required.
#'
#' All public GOF statistic primitives return the scale used for null
#' comparison: `n U_n`, `n V_n`, or `n FSSDhat^2`. Their bootstrap or simulated
#' null draws use exactly the matching scale. Unscaled U-statistics are internal
#' intermediate values.
#'
#' [ksd_u_test()] tests independent observations with an off-diagonal
#' U-statistic. [ksd_v_test()] includes the diagonal and uses Rademacher or
#' Markov signs. [fssd_test()] uses finite Stein features at test locations.
#' For KSD-U, [ksd_uq_matrix()] builds the matrix, [ksd_u_statistic()] computes
#' the statistic, and [ksd_u_bootstrap()] calibrates it. The KSD-V counterparts
#' are [ksd_vq_matrix()], [ksd_v_statistic()], and [ksd_v_bootstrap()]. For
#' FSSD, [compute_tau()] constructs the features, [fssd_statistic()] computes
#' the statistic, and [fssd_null_pvalue()] simulates the null distribution.
#'
#' [svgd()] transports an initial particle set. [stein_points()] constructs a
#' point set by continuous search with [fmin_grid()], [fmin_mc()], or
#' [fmin_nm()]. [sp_mcmc()] constructs points from states visited by short
#' Markov chains. [stein_thinning()] selects rows from an existing sample.
#'
#' [stein_kernel()] creates Gaussian RBF and IMQ kernels;
#' [custom_stein_kernel()] creates one from callbacks. [eval_kernel()],
#' [grad_x_kernel()], [trace_mixed_kernel()], [cross_kernel()], and
#' [grad_theta_v_kernel()] provide the corresponding kernel calculations.
#'
#' [gmm()] constructs a Gaussian mixture, [rgmm()] samples it,
#' [densitygmm()] evaluates its density, and [get_score_evaluator()] returns its
#' `function(X)` score callback.
"_PACKAGE"
