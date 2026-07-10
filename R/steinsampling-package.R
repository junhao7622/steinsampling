#' steinsampling: Stein tests and Stein sampling tools
#'
#' Tools for score-based goodness-of-fit testing and Stein-based sampling in R.
#' The package provides KSD and FSSD tests, Stein thinning, Stein Points,
#' SP-MCMC, SVGD, and reusable Stein kernel objects.
#'
#' @details
#' Most functions in this package work with a score function. For a target
#' density `p(x)`, the score is the gradient of the log density,
#' \deqn{s_p(x) = \nabla_x \log p(x).}
#' This means many tests and samplers can be used without knowing the
#' normalizing constant of `p(x)`.
#'
#' The main testing functions are [ksd_u_test()], [ksd_v_test()], and
#' [fssd_test()]. The main sampling, compression, and point-construction
#' functions are [stein_thinning()], [stein_points()], [sp_mcmc()], and
#' [svgd()]. The helper [stein_kernel()] creates the built-in kernels used by
#' these methods, and the Gaussian mixture helpers such as [gmm()] and
#' [scorefunctiongmm()] are included for small simulations and examples.
#'
#' The public API is organized in layers. The top layer runs complete methods:
#' [ksd_u_test()], [ksd_v_test()], [fssd_test()], [stein_thinning()],
#' [stein_points()], [sp_mcmc()], and [svgd()]. The next layer exposes the
#' reusable numerical pieces used by those methods, such as
#' [stein_kernel_matrix()], [ksd_uq_matrix()], [ksd_u_statistic()],
#' [ksd_u_bootstrap()], [ksd_vq_matrix()], [ksd_v_statistic()],
#' [ksd_v_bootstrap()], [compute_tau()], [compute_fssd_unbiased_stat()], and
#' [compute_fssd_null_pvalue()]. These lower-level functions are exported so a
#' user can inspect intermediate matrices, reuse the same bootstrap weights, or
#' reproduce a paper calculation one step at a time.
#'
#' The main testing path is top-down. Start with [ksd_u_test()] for the Liu,
#' Lee, and Jordan independent-sample KSD test, [ksd_v_test()] for the
#' Chwialkowski et al. V-statistic and wild-bootstrap version, or [fssd_test()]
#' for a finite-feature Stein test. Moving down from [ksd_u_test()] gives
#' [ksd_uq_matrix()] for the Stein-kernel matrix, [ksd_u_statistic()] for the
#' off-diagonal U-statistic, and [ksd_u_bootstrap()] for the centered
#' multinomial bootstrap. Moving sideways to [ksd_v_test()] keeps the same
#' sample, score function, and kernel layer, but changes the matrix compression
#' to include the diagonal and changes the bootstrap weights to Rademacher or
#' Markov signs. Moving sideways to [fssd_test()] keeps the same score-function
#' idea but replaces the full pairwise matrix by [compute_tau()], a finite set
#' of Stein features evaluated at test locations.
#'
#' The sampling, compression, and point-construction path uses the same
#' Stein-kernel layer in different ways. [stein_thinning()] is a compression
#' method: it starts from an existing sample matrix, usually an MCMC output, and
#' returns row indices for a smaller empirical measure. [stein_points()] is a
#' point-construction method: it searches over continuous candidates using an
#' optimizer such as [fmin_grid()], [fmin_mc()], or [fmin_nm()]. [sp_mcmc()] is a
#' Stein Points variant that uses short MCMC paths as finite candidate sets for
#' constructing new support points. [svgd()] and [update_svgd()] move all
#' particles by a deterministic transport update rather than compressing an
#' existing sample or selecting a point sequence.
#'
#' Kernel functions form a separate layer. [stein_kernel()] and
#' [custom_stein_kernel()] create kernel objects. The generic functions
#' [eval_kernel()], [grad_x_kernel()], [trace_mixed_kernel()],
#' [cross_kernel()], and [grad_theta_v_kernel()] expose the pieces needed to
#' assemble the scalar Stein kernel `k0` or the finite-location FSSD features.
#' Most users call the top-level algorithms and never call these generics
#' directly; they are public because custom kernels and diagnostic checks need
#' the same interface as the built-in kernels.
#'
#' The Gaussian mixture helpers form an example-model layer. [gmm()] creates a
#' model, [rgmm()] simulates from it, [likelihoodgmm()],
#' [posteriorgmm()], and [scorefunctiongmm()] evaluate density-related
#' quantities, and [get_score_evaluator()] turns a fixed mixture into the
#' `function(X)` score callback expected by the Stein routines.
#'
#' @import stats graphics
"_PACKAGE"
