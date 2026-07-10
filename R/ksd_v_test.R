#' KSD goodness-of-fit test using the Chwialkowski et al. V-statistic
#'
#' Runs the Kernel Stein Discrepancy goodness-of-fit test in the V-statistic
#' form used by Chwialkowski et al. (2016). It compares samples with a target
#' distribution through the target score and uses a wild bootstrap to approximate
#' the null distribution.
#'
#' @details
#' Let `p` be the target density and let `k0(x, y)` be the Stein kernel built
#' from the selected base kernel and the supplied target score. The score is
#' \eqn{s_p(x) = \nabla_x \log p(x)}; because only this score is used, the
#' normalizing constant of `p` can be unknown. See [stein_kernel_matrix()] for
#' the four terms that make up `k0`.
#'
#' The sample is represented by rows of `X`, and `score_function(X)` must return
#' one score vector for each row. If `scaling = NULL`, the built-in kernel scale
#' is selected by [find_median_distance()]. This default is a squared scale:
#' for Gaussian RBF kernels it is `h^2`, while for IMQ kernels it is `c^2`.
#'
#' For observations \eqn{x_1,\ldots,x_n}, this test uses
#' \deqn{n V_n = \frac{1}{n} \sum_{i=1}^n \sum_{j=1}^n k0(x_i, x_j).}
#' Some papers write the unscaled V-statistic as
#' \eqn{V_n = n^{-2} \sum_{i,j} k0(x_i, x_j)}. This package reports and
#' bootstraps the scaled version `n V_n`, which is the form compared with the
#' wild-bootstrap distribution. Unlike the U-statistic in [ksd_u_test()], the
#' diagonal terms `k0(x_i, x_i)` are included.
#'
#' The implemented algorithm is: (1) build the full Stein-kernel matrix,
#' including diagonal entries, (2) compute the scaled V-statistic, (3) generate
#' wild-bootstrap sign weights, and (4) recompute the same quadratic form with
#' those signs. Independent Rademacher signs are appropriate for independent
#' samples. The dependent-sample result in Chwialkowski et al. is narrower than
#' merely requiring ordered rows: it assumes a stationary tau-mixing process,
#' the summability condition
#' \deqn{\sum_{t=1}^{\infty} t^2 \sqrt{\tau(t)} < \infty,}
#' a Lipschitz Stein kernel `k0`, and the moment condition
#' \eqn{E[k0(Z,Z)^2] < \infty}. Geometrically ergodic Markov chains satisfy the
#' required mixing property only with additional moment assumptions.
#'
#' Markov signs use a two-state sign chain with probability `change_prob` of
#' switching sign between neighboring observations. In the paper's asymptotic
#' construction this probability is a sequence \eqn{p_n = 1 / w_n} with
#' \eqn{w_n \to \infty} and \eqn{w_n = o(n)}; equivalently,
#' \eqn{p_n \to 0} while \eqn{n p_n \to \infty}. Thus `change_prob` must be
#' chosen as a function of sample size and dependence strength. The value `0.5`
#' produces independent signs and is the paper's i.i.d. setting, not a default
#' calibration for dependent samples; using it for dependent rows can
#' miscalibrate the test. Large positive statistic values are evidence that the
#' sample does not match the target score well.
#'
#' The reported p-value is the package's finite-bootstrap convention
#' \deqn{\frac{1 + \#\{b:T_b \ge T_{obs}\}}{B+1}.}
#' Chwialkowski et al. instead state the decision rule through the empirical
#' bootstrap quantile. The bootstrap statistics and observed statistic here
#' follow the paper, but this corrected p-value is an additional finite-
#' simulation convention and is not a separate finite-sample validity guarantee
#' for dependent data.
#'
#' This function is parallel to [ksd_u_test()]. Both start from `X`,
#' `score_function`, and a Stein kernel. The U version targets Liu et al.'s
#' off-diagonal unbiased statistic with centered multinomial bootstrap weights.
#' This V version keeps diagonal self-interactions, reports the scaled statistic
#' `n V_n`, and uses wild-bootstrap signs. Moving down one layer gives
#' [ksd_vq_matrix()], [ksd_v_statistic()], and [ksd_v_bootstrap()].
#'
#' @param X Numeric vector or matrix of samples.
#' @param score_function Function returning the target score for each row of
#'   `X`.
#' @param boot_method One of `"rademacher"` or `"markov"`.
#' @param scaling Positive kernel scale. `NULL` uses the median squared
#'   distance between sample pairs.
#' @param nboot Number of bootstrap samples.
#' @param change_prob Markov sign-flip probability used when
#'   `boot_method = "markov"`. It must then be supplied explicitly and should
#'   be tuned with sample size; see Details. Use `boot_method = "rademacher"`
#'   for independent signs.
#' @param kernel Kernel choice: `"gaussian_rbf"`, `"imq"`, or a `SteinKernel`
#'   object, including objects created by [custom_stein_kernel()].
#' @param return_raw_boot Logical; if `TRUE`, append raw bootstrap samples to
#'   the output.
#' @param median_max_samples,median_use_sampling,median_seed Options used by
#'   the median-scale rule when `scaling = NULL`.
#' @param block_size,block_threshold Options for computing large kernel
#'   matrices in blocks.
#' @param imq_beta IMQ kernel exponent.
#' @return
#' An object of class `htest`, implemented as a list with:
#' * `statistic`: one number labeled `ksd_v`; this is the observed scaled
#'   statistic `n V_n`.
#' * `p.value`: right-tail wild-bootstrap p-value using the package's
#'   `(1 + exceedances) / (nboot + 1)` finite-bootstrap convention.
#' * `method`: text naming the bootstrap method and kernel.
#' * `data.name`: expression used for `X`.
#' * `parameter`: numeric vector containing `nboot`, the squared kernel
#'   `scaling` used, and `change_prob` when `boot_method = "markov"`.
#' * `bootstrap_samples`: present only when `return_raw_boot = TRUE`; numeric
#'   vector of wild-bootstrap statistics on the same `n V_n` scale.
#'
#' The `htest` shape is used so the result behaves like a standard R test
#' object. The `parameter` field records the bootstrap and kernel choices
#' because changing either one changes the null distribution being simulated.
#' @examples
#' X <- matrix(rnorm(20), ncol = 1)
#' score_function <- function(x) -as.matrix(x)
#' ksd_v_test(X, score_function, nboot = 10)
#' @export
ksd_v_test <- function(X, score_function,
                       boot_method = c("rademacher", "markov"),
                       scaling = NULL,
                       nboot = 1000,
                       change_prob = NULL,
                       kernel = c("gaussian_rbf", "imq"),
                       return_raw_boot = FALSE,
                       median_max_samples = 2000, median_use_sampling = TRUE,
                       median_seed = 123, block_size = NULL, block_threshold = 5000, imq_beta = -0.5) {
  data_name <- deparse(substitute(X))
  boot_method <- match.arg(boot_method)
  prep <- prepare_ksd_v_inputs(
    X = X,
    score_function = score_function,
    scaling = scaling,
    kernel = kernel,
    median_max_samples = median_max_samples,
    median_use_sampling = median_use_sampling,
    median_seed = median_seed,
    imq_beta = imq_beta
  )
  n <- nrow(prep$X)

  if (!is.numeric(nboot) || length(nboot) != 1 || !is.finite(nboot) || nboot <= 0) {
    stop("nboot must be a positive scalar")
  }
  nboot <- as.integer(nboot)

  cp_info <- resolve_change_probability_for_bootstrap(boot_method, change_prob)
  # Wild-bootstrap columns are sign sequences; Markov signs retain sample-order dependence.
  W_mat <- generate_bootstrap_weights(n, nboot, boot_method = boot_method,
                                      change_prob = cp_info$change_prob)
  block_settings <- resolve_block_settings(n, block_size, block_threshold)

  engine_res <- internal_v_compute_engine(
    X = prep$X,
    grads = prep$grads,
    kernel_obj = prep$kernel_obj,
    W_mat = W_mat,
    use_block = block_settings$use_block_mode,
    block_size = block_settings$block_size
  )

  if (any(is.finite(engine_res$max_offdiag) & engine_res$max_offdiag < 1e-12)) {
    warning("Vanishing off-diagonal kernel detected: bandwidth may be too small.", call. = FALSE)
  }

  res <- list(
    statistic = c(ksd_v = engine_res$global_stat),
    p.value = engine_res$global_pval,
    method = sprintf(
      "Kernel Stein Discrepancy (V-statistics) - %s bootstrap, %s kernel",
      boot_method, prep$kernel_name
    ),
    data.name = data_name,
    parameter = c(
      nboot = nboot,
      scaling = if (is.null(prep$scaling)) NA_real_ else prep$scaling,
      change_prob = if (boot_method == "markov") cp_info$change_prob else NA_real_
    )
  )

  if (return_raw_boot) res$bootstrap_samples <- engine_res$raw_boots
  class(res) <- "htest"
  res
}

#' Build the Stein-kernel matrix for the KSD V-statistic
#'
#' Computes the pairwise Stein kernel values used by the V-statistic version of
#' the KSD test.
#'
#' @details
#' Entry `(i, j)` is `k0(x_i, x_j)`, including diagonal entries. Use
#' [ksd_v_statistic()] to turn the matrix into the scaled statistic `n V_n`,
#' [ksd_v_bootstrap()] to generate wild-bootstrap draws from the fixed matrix,
#' or [ksd_v_test()] to run the full test. See [stein_kernel_matrix()] for the
#' formula used to assemble `k0`.
#'
#' This is the V-statistic analogue of [ksd_uq_matrix()]. It is useful when the
#' sample comes from an ordered or dependent process and the user wants to keep
#' matrix construction separate from the wild-bootstrap step. The diagonal is
#' retained because the V-statistic includes self-interactions.
#'
#' To move sideways from this function to [ksd_uq_matrix()], keep the same
#' `X`, `score_function`, and kernel settings. The matrix entries are still
#' `k0(x_i, x_j)`. The difference is the statistic and bootstrap that should be
#' used afterward: U-statistic helpers drop the diagonal and use centered
#' multinomial weights; V-statistic helpers keep the diagonal and use
#' wild-bootstrap signs.
#' The `q` in `ksd_vq_matrix()` follows the literature's Stein-kernel notation;
#' it does not introduce a second user-supplied distribution. The target is the
#' density whose score is returned by `score_function`, and the returned object
#' is the package's `K0` matrix with entries `k0(x_i, x_j)`.
#'
#' @param X Numeric vector or matrix of samples.
#' @param score_function Function returning the target score for each row of
#'   `X`.
#' @param scaling Positive kernel scale. `NULL` uses the median squared
#'   distance between sample pairs.
#' @param kernel Kernel choice: `"gaussian_rbf"`, `"imq"`, or a `SteinKernel`
#'   object, including objects created by [custom_stein_kernel()].
#' @param median_max_samples,median_use_sampling,median_seed Options used by
#'   the median-scale rule when `scaling = NULL`.
#' @param imq_beta IMQ kernel exponent.
#' @return
#' Numeric `n x n` matrix with entry `k0(x_i, x_j)`, including the diagonal.
#' Pass this matrix to [ksd_v_statistic()] to get `n V_n`, or to
#' [ksd_v_bootstrap()] to generate wild-bootstrap replicates.
#' @examples
#' X <- matrix(rnorm(5), ncol = 1)
#' score_function <- function(x) -as.matrix(x)
#' ksd_vq_matrix(X, score_function)
#' @export
ksd_vq_matrix <- function(X, score_function,
                          scaling = NULL,
                          kernel = c("gaussian_rbf", "imq"),
                          median_max_samples = 2000, median_use_sampling = TRUE,
                          median_seed = 123, imq_beta = -0.5) {
  prep <- prepare_ksd_v_inputs(
    X = X,
    score_function = score_function,
    scaling = scaling,
    kernel = kernel,
    median_max_samples = median_max_samples,
    median_use_sampling = median_use_sampling,
    median_seed = median_seed,
    imq_beta = imq_beta
  )

  ksd_vq_matrix_from_scores(prep$X, prep$grads, prep$kernel_obj)
}

#' Compute the KSD V-statistic from its Stein-kernel matrix
#'
#' The returned statistic is `n * mean(U_mat)`, which is the same as summing all
#' entries of the matrix and dividing by `n`.
#'
#' @details
#' If `U_mat[i, j] = k0(x_i, x_j)`, the statistic is
#' \deqn{\frac{1}{n} \sum_{i=1}^n \sum_{j=1}^n U_{ij}.}
#' This is the scaled statistic `n V_n` when
#' \eqn{V_n = n^{-2} \sum_{i,j} U_{ij}}.
#' The helper is separated out so users can reuse a precomputed Stein-kernel
#' matrix or compare dense and blocked matrix calculations. It is the
#' matrix-compression layer between [ksd_vq_matrix()], which evaluates scores
#' and kernels, and [ksd_v_bootstrap()], which keeps the matrix fixed and
#' changes only the wild-bootstrap sign sequences.
#'
#' The diagonal is included here because the V-statistic treats every ordered
#' pair of rows, including `(i, i)`, as part of the empirical average. This is
#' the main algebraic difference from [ksd_u_statistic()]. Keeping the two
#' compression functions separate prevents accidental mixing of U-statistic and
#' V-statistic conventions when users inspect lower-level matrices.
#'
#' This function performs no bootstrap and does not evaluate the score
#' function. It is the deterministic compression from the full pairwise matrix
#' to the one-number V-statistic used by [ksd_v_test()].
#'
#' Because the function only needs `U_mat`, it is useful for checking whether
#' two ways of building the same Stein-kernel matrix, for example dense and
#' blocked computation, lead to the same final statistic before any bootstrap
#' randomness is introduced.
#'
#' @param U_mat Numeric square matrix of pairwise Stein kernel values.
#' @return
#' One numeric value: the scaled V-statistic `n V_n`. This is the same scale as
#' the values returned by [ksd_v_bootstrap()] and as the `statistic` component
#' of [ksd_v_test()]. It is returned as a plain number because the kernel choice
#' and wild-bootstrap settings are handled by the higher-level test object.
#' @examples
#' U <- matrix(c(1, 0.2, 0.3, 0.2, 1, 0.4, 0.3, 0.4, 1), 3, 3)
#' ksd_v_statistic(U)
#' @export
ksd_v_statistic <- function(U_mat) {
  U_mat <- validate_ksd_v_matrix(U_mat)
  nrow(U_mat) * mean(U_mat)
}

#' Wild-bootstrap the KSD V-statistic
#'
#' Applies independent or Markov sign weights to a fixed KSD V-statistic matrix.
#' This is the bootstrap step of the Chwialkowski et al. test separated from the
#' construction of the Stein-kernel matrix.
#'
#' @details
#' For each bootstrap weight vector `w`, the bootstrap statistic is
#' \deqn{\frac{1}{n} \sum_{i,j} w_i U_{ij} w_j.}
#' This is on the same scaled `n V_n` scale as [ksd_v_statistic()]. Equivalently,
#' Chwialkowski et al. define a bootstrap V-statistic with a `1 / n^2` factor and
#' compare `n` times that value; the formula above is that scaled value. The
#' weights are either independent signs or Markov signs, depending on
#' `boot_method`. For the dependent-sample theory, the Markov flip probability
#' is a sample-size-dependent sequence \eqn{p_n \to 0} satisfying
#' \eqn{n p_n \to \infty}, together with the mixing, Lipschitz, and moment
#' assumptions documented in [ksd_v_test()]. A fixed value of `0.5` gives
#' independent signs and does not reproduce that dependent wild bootstrap.
#' Supplying `W_mat` is useful when the same bootstrap sign sequences should be
#' reused across kernels or repeated experiments.
#'
#' The function is separated from [ksd_vq_matrix()] for the same reason as in
#' the U-statistic API: matrix construction is deterministic once `X`, the
#' score, and the kernel are fixed, while the bootstrap is random unless the
#' user supplies `W_mat`. This split lets simulation studies control those two
#' sources separately.
#'
#' The parallel U-statistic helper is [ksd_u_bootstrap()]. Use this V helper
#' when `U_mat` will be summarized by [ksd_v_statistic()] and when bootstrap
#' weights should be sign sequences. Use the U helper when `U_mat` will be
#' summarized by [ksd_u_statistic()] and the Liu et al. centered multinomial
#' bootstrap is desired.
#'
#' @param U_mat Numeric square matrix of pairwise Stein kernel values.
#' @param W_mat Optional `n x B` wild-bootstrap sign matrix, with entries
#'   usually equal to `-1` or `1`.
#' @param nboot Number of bootstrap samples.
#' @param boot_method One of `"rademacher"` or `"markov"` when `W_mat` is NULL.
#' @param change_prob Markov sign-flip probability when
#'   `boot_method = "markov"`; it must then be supplied explicitly.
#' @return
#' Numeric vector of bootstrap statistics on the scaled `n V_n` scale. Its
#' length is `nboot` when signs are generated internally, or `ncol(W_mat)` when
#' a sign matrix is supplied.
#' @examples
#' U <- matrix(c(1, 0.2, 0.3, 0.2, 1, 0.4, 0.3, 0.4, 1), 3, 3)
#' ksd_v_bootstrap(U, nboot = 5)
#' @export
ksd_v_bootstrap <- function(U_mat, W_mat = NULL, nboot = 1000,
                            boot_method = c("rademacher", "markov"),
                            change_prob = NULL) {
  boot_method <- match.arg(boot_method)
  U_mat <- validate_ksd_v_matrix(U_mat)
  n <- nrow(U_mat)

  if (is.null(W_mat)) {
    if (!is.numeric(nboot) || length(nboot) != 1 || !is.finite(nboot) || nboot <= 0) {
      stop("nboot must be a positive scalar")
    }
    nboot <- as.integer(nboot)
    cp_info <- resolve_change_probability_for_bootstrap(boot_method, change_prob)
    W_mat <- generate_bootstrap_weights(n, nboot, boot_method = boot_method,
                                        change_prob = cp_info$change_prob)
  } else {
    W_mat <- as.matrix(W_mat)
    if (!is.numeric(W_mat) || nrow(W_mat) != n || ncol(W_mat) < 1) {
      stop("W_mat must be a numeric matrix with nrow(U_mat) rows and at least one column")
    }
  }

  as.numeric(colSums((U_mat %*% W_mat) * W_mat) / n)
}


#' Prepare samples, scores, and kernel for KSD V-statistic routines
#'
#' Checks the sample and score shapes, builds the requested Stein kernel, and
#' chooses a default scale when needed.
#'
#' @details
#' A vector `X` is treated as one-dimensional data. If `scaling` is `NULL`, the
#' median squared distance between sample rows is used as the squared kernel
#' scale. Built-in Gaussian RBF kernels use `h^2 = scaling`; built-in IMQ
#' kernels use `c^2 = scaling`.
#'
#' @param X Numeric vector or matrix of samples.
#' @param score_function Function returning the target score for each row of
#'   `X`.
#' @param scaling Positive kernel scale. `NULL` uses the median squared
#'   distance between sample pairs.
#' @param kernel Kernel choice: `"gaussian_rbf"`, `"imq"`, or a `SteinKernel`
#'   object.
#' @param median_max_samples,median_use_sampling,median_seed Options used by
#'   the median-scale rule when `scaling = NULL`.
#' @param imq_beta IMQ kernel exponent.
#'
#' @return A list with checked samples, scores, kernel object, kernel name, and
#'   selected scale.
#' @noRd
prepare_ksd_v_inputs <- function(X, score_function, scaling = NULL,
                                 kernel = c("gaussian_rbf", "imq"),
                                 median_max_samples = 2000,
                                 median_use_sampling = TRUE,
                                 median_seed = 123, imq_beta = -0.5) {
  x_mat <- validate_samples(X, min_rows = 2)
  grads <- validate_scores(score_function, x_mat)

  if (inherits(kernel, "SteinKernel")) {
    kernel_obj <- kernel
    kernel_name <- kernel_type_name(kernel_obj)
    kernel_scale <- kernel_scaling_value(kernel_obj)
    if (inherits(kernel_obj, "SteinKernel_gaussian_rbf") && !is.finite(kernel_scale)) {
      if (is.null(scaling)) {
        scaling <- find_median_distance(x_mat, median_max_samples, median_use_sampling, median_seed)
      }
      kernel_obj <- instantiate_kernel("gaussian_rbf", scaling, imq_beta)
    } else {
      scaling <- kernel_scale
    }
  } else {
    kernel_name <- match.arg(kernel, c("gaussian_rbf", "imq"))
    if (is.null(scaling)) {
      scaling <- find_median_distance(x_mat, median_max_samples, median_use_sampling, median_seed)
    }
    kernel_obj <- instantiate_kernel(kernel_name, scaling, imq_beta)
  }

  list(
    X = x_mat,
    grads = grads,
    kernel_obj = kernel_obj,
    kernel_name = kernel_name,
    scaling = scaling
  )
}

#' Check a KSD V-statistic matrix
#'
#' Converts the input to a matrix and checks that it is square with at least two
#' rows.
#'
#' @param U_mat Candidate KSD matrix.
#'
#' @return A numeric square matrix.
#' @noRd
validate_ksd_v_matrix <- function(U_mat) {
  if (is.null(dim(U_mat))) {
    stop("U_mat must be a square numeric matrix")
  }
  U_mat <- as.matrix(U_mat)
  if (!is.numeric(U_mat) || nrow(U_mat) != ncol(U_mat) || nrow(U_mat) < 2) {
    stop("U_mat must be a square numeric matrix with at least two rows")
  }
  U_mat
}

#' Build a KSD V matrix from precomputed scores
#'
#' This is the lower-level version of [ksd_vq_matrix()] used after inputs have
#' already been checked.
#'
#' @param X Numeric sample matrix.
#' @param grads Score matrix for `X`.
#' @param kernel_obj Stein kernel object.
#'
#' @return Numeric matrix of pairwise Stein kernel values.
#' @noRd
ksd_vq_matrix_from_scores <- function(X, grads, kernel_obj) {
  stein_kernel_matrix(kernel_obj, X, grads)
}

#' Compute the KSD V statistic and bootstrap values
#'
#' Runs the dense or blocked calculation path after inputs and bootstrap weights
#' have been prepared.
#'
#' @param X Numeric sample matrix.
#' @param grads Score matrix for `X`.
#' @param kernel_obj Stein kernel object.
#' @param W_mat Bootstrap weight matrix.
#' @param use_block Logical; whether to use blockwise matrix calculations.
#' @param block_size Number of rows and columns in each block.
#'
#' @return A list with the observed statistic, p-value, largest off-diagonal
#'   base-kernel value, and raw bootstrap statistics.
#' @noRd
internal_v_compute_engine <- function(X, grads, kernel_obj, W_mat, use_block, block_size) {
  if (use_block) {
    return(compute_v_blocked_pure(X, grads, kernel_obj, W_mat, block_size))
  }

  k_mat <- eval_kernel(kernel_obj, X)
  U_mat <- ksd_vq_matrix_from_scores(X, grads, kernel_obj)
  stat <- ksd_v_statistic(U_mat)
  boot_stats <- ksd_v_bootstrap(U_mat, W_mat = W_mat)

  list(
    global_stat = unname(stat),
    global_pval = bootstrap_pvalue_right_tail(boot_stats, stat),
    max_offdiag = max_offdiag_from_kernel_matrix(k_mat),
    raw_boots = as.numeric(boot_stats)
  )
}

#' Compute the KSD V statistic in matrix blocks
#'
#' Splits the pairwise Stein kernel matrix into blocks so that the full `n x n`
#' matrix does not need to be stored at once.
#'
#' @param X Numeric sample matrix.
#' @param grads Score matrix for `X`.
#' @param kernel_obj Stein kernel object.
#' @param W_mat Bootstrap weight matrix.
#' @param block_size Number of rows and columns in each block.
#'
#' @return A list with the observed statistic, p-value, largest off-diagonal
#'   base-kernel value, and raw bootstrap statistics.
#' @noRd
compute_v_blocked_pure <- function(X, grads, kernel_obj, W_mat, block_size) {
  n <- nrow(X)
  nboot <- ncol(W_mat)
  i_starts <- seq(1, n, by = block_size)
  j_starts <- seq(1, n, by = block_size)

  stat_sum <- 0
  boot_stats <- numeric(nboot)
  max_offdiag <- -Inf

  for (i_start in i_starts) {
    i_end <- min(i_start + block_size - 1L, n)
    ii <- i_start:i_end
    X_i <- X[ii, , drop = FALSE]
    g_i <- grads[ii, , drop = FALSE]
    W_i <- W_mat[ii, , drop = FALSE]

    for (j_start in j_starts) {
      j_end <- min(j_start + block_size - 1L, n)
      jj <- j_start:j_end
      X_j <- X[jj, , drop = FALSE]
      g_j <- grads[jj, , drop = FALSE]
      W_j <- W_mat[jj, , drop = FALSE]

      block <- compute_v_block_pure(X_i, X_j, g_i, g_j, kernel_obj)
      stat_sum <- stat_sum + sum(block$m_block)
      # ksd_v_statistic() reports n V_n, so bootstrap blocks are accumulated on the same scale.
      boot_stats <- boot_stats + colSums((block$m_block %*% W_j) * W_i) / n

      if (i_start == j_start) {
        if (nrow(block$k_block) > 1) {
          max_offdiag <- max(
            max_offdiag,
            max(block$k_block[lower.tri(block$k_block, diag = FALSE)])
          )
        }
      } else {
        max_offdiag <- max(max_offdiag, max(block$k_block))
      }
    }
  }

  stat <- stat_sum / n
  list(
    global_stat = stat,
    global_pval = bootstrap_pvalue_right_tail(boot_stats, stat),
    max_offdiag = max_offdiag,
    raw_boots = as.numeric(boot_stats)
  )
}

#' Compute one KSD V matrix block
#'
#' Evaluates both the base kernel and the Stein kernel between two blocks of
#' rows. The base-kernel block is used only for the small-bandwidth warning.
#'
#' @param X_i,X_j Sample blocks.
#' @param g_i,g_j Score blocks matching `X_i` and `X_j`.
#' @param kernel_obj Stein kernel object.
#'
#' @return A list with `m_block`, the Stein-kernel block, and `k_block`, the
#'   base-kernel block.
#' @noRd
compute_v_block_pure <- function(X_i, X_j, g_i, g_j, kernel_obj) {
  k_block <- eval_kernel(kernel_obj, X_i, X_j)
  m_block <- stein_kernel_matrix(kernel_obj, X_i, g_i, X_j, g_j)
  list(m_block = m_block, k_block = k_block)
}
