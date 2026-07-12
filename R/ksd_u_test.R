#' KSD goodness-of-fit test using the Liu et al. U-statistic
#'
#' Runs the Kernel Stein Discrepancy goodness-of-fit test in the U-statistic
#' form of Liu, Lee, and Jordan (2016). The input sample is compared with a
#' target distribution through the target score, so the normalizing constant of
#' the target density is not needed.
#'
#' @details
#' Let `p` be the target density, \eqn{s_p(x)=\nabla_x\log p(x)} its score, and
#' `k0(x,y)` the Stein kernel built from that score and the selected base kernel;
#' the normalizing constant of `p` is unnecessary. Liu et al. call this kernel
#' `u_q` because their target is `q`; the package consistently uses `k0`. See
#' [stein_kernel_matrix()] for its four terms.
#'
#' `X` is interpreted row-wise; vectors become `n x 1` matrices, and
#' `score_function` is evaluated once and must return an `n x d` matrix. With
#' `scaling = NULL`, [find_median_distance()] uses all rows to return the squared
#' median pairwise distance: an RBF squared bandwidth or IMQ squared length
#' scale. This exact calculation has quadratic time and memory cost, so large
#' samples should supply `scaling`. For a preconditioned RBF object without a
#' fixed bandwidth, the rule uses preconditioned distances and retains that
#' preconditioner.
#'
#' For observations \eqn{x_1,\ldots,x_n}, the paper's estimator is
#' \deqn{U = \frac{1}{n(n - 1)} \sum_{i \ne j} k0(x_i, x_j).}
#' It omits `k0(x_i,x_i)` and is degenerate under the target, so its null is not
#' a plain normal approximation.
#'
#' Following Liu et al., the test (1) builds the `n x n` Stein-kernel matrix,
#' (2) zeros its diagonal for the U-statistic, (3) generates centered
#' multinomial weights `w_i=N_i/n-1/n`, and (4) compares `U` with `w^T K0 w`.
#' Observed and bootstrap values remain on the unmultiplied U-statistic scale;
#' multiplying both by `n` leaves the p-value unchanged. Large positive values
#' indicate disagreement with the target score in kernel Stein discrepancy.
#'
#' The statistic and bootstrap reproduce Liu et al.; the scalar p-value follows
#' the package convention
#' \deqn{p = \frac{1 + \#\{b:T_b \ge T_{obs}\}}{B+1}.}
#' Algorithm 1 instead rejects using the uncorrected strict proportion
#' \eqn{B^{-1}\#\{b:T_b>T_{obs}\}}. The `+1` correction prevents zero Monte
#' Carlo p-values but is neither a verbatim algorithm step nor a new
#' finite-sample KSD-bootstrap guarantee.
#'
#' This is Liu et al.'s independent-sample test; [ksd_v_test()] supplies the
#' Chwialkowski et al. V-statistic with independent or Markov wild-bootstrap
#' signs for ordered or dependent samples. Its lower-level pieces are
#' [ksd_uq_matrix()] for the Stein-kernel matrix, [ksd_u_statistic()] for the
#' off-diagonal average, and [ksd_u_bootstrap()] for centered
#' multinomial weights. The V-statistic counterparts are [ksd_vq_matrix()],
#' [ksd_v_statistic()], and [ksd_v_bootstrap()].
#'
#' @param X Numeric vector or matrix of samples. A vector is treated as `n x 1`.
#' @param score_function Function returning the target score for each row of
#'   `X`.
#' @param boot_method Bootstrap method. Currently only
#'   `"multinomial_centered"` is supported.
#' @param scaling Positive kernel scale. `NULL` uses the square of the exact
#'   all-pair median distance.
#' @param nboot Number of bootstrap samples.
#' @param kernel Kernel choice: `"gaussian_rbf"`, `"imq"`, or a `SteinKernel`
#'   object, including objects created by [custom_stein_kernel()].
#' @param return_raw_boot Logical; if `TRUE`, return bootstrap samples.
#' @param block_size,block_threshold Options for computing large kernel
#'   matrices in blocks.
#' @param imq_beta IMQ kernel exponent.
#' @return
#' An object of class `htest`, implemented as a list with:
#' * `statistic`: one number labeled `ksd_u`; this is the observed
#'   off-diagonal U-statistic.
#' * `p.value`: right-tail bootstrap p-value using the package's
#'   `(1 + exceedances) / (nboot + 1)` finite-bootstrap convention.
#' * `method`: text naming the KSD U-statistic test and kernel.
#' * `data.name`: expression used for `X`.
#' * `parameter`: numeric vector containing `nboot`, the squared kernel
#'   `scaling` used, and `imq_beta` when the IMQ kernel is used.
#' * `bootstrap_samples`: present only when `return_raw_boot = TRUE`; numeric
#'   vector of bootstrap U-statistics on the same scale as `statistic`.
#'
#' The `htest` shape prints like other R tests; `parameter` and optional
#' `bootstrap_samples` retain settings needed to reproduce or diagnose the
#' p-value.
#' @examples
#' X <- matrix(rnorm(10), ncol = 1)
#' score_function <- function(x) -as.matrix(x)
#' ksd_u_test(X, score_function, nboot = 10)
#' @export
ksd_u_test <- function(X, score_function,
                       boot_method = "multinomial_centered",
                       scaling = NULL,
                       nboot = 1000,
                       kernel = c("gaussian_rbf", "imq"),
                       return_raw_boot = FALSE,
                       block_size = NULL, block_threshold = 5000,
                       imq_beta = -0.5) {

  data_name <- deparse(substitute(X))
  boot_method <- resolve_ksd_u_boot_method(boot_method)
  prep <- prepare_ksd_u_inputs(
    X = X,
    score_function = score_function,
    scaling = scaling,
    kernel = kernel,
    imq_beta = imq_beta
  )
  n <- nrow(prep$X)

  nboot <- validate_positive_integer(nboot, "nboot")

  # Liu et al. Eq. 16 uses centered multinomial proportions N_i / n - 1 / n.
  W_mat <- generate_bootstrap_weights(n, nboot, boot_method = boot_method)
  block_settings <- resolve_block_settings(n, block_size, block_threshold)

  engine_res <- internal_u_compute_engine(
    X = prep$X,
    grads = prep$grads,
    kernel_obj = prep$kernel_obj,
    W_mat = W_mat,
    boot_method = boot_method,
    use_block = block_settings$use_block_mode,
    block_size = block_settings$block_size
  )

  res <- list(
    statistic = c(ksd_u = engine_res$stat),
    p.value = engine_res$pval,
    method = paste("Kernel Stein Discrepancy (U-statistics) -", prep$kernel_name),
    data.name = data_name,
    parameter = c(
      nboot = nboot,
      scaling = if (is.null(prep$scaling)) NA_real_ else prep$scaling,
      imq_beta = if (prep$kernel_name == "imq" && !is.null(prep$kernel_obj$beta)) prep$kernel_obj$beta else NA_real_
    )
  )

  if (isTRUE(return_raw_boot)) res$bootstrap_samples <- engine_res$raw_boots
  class(res) <- "htest"
  res
}

#' Build the Stein-kernel matrix for the KSD U-statistic
#'
#' Computes the pairwise Stein kernel values that are later averaged by the
#' U-statistic. This is the matrix-level part of the Liu et al. KSD test.
#'
#' @details
#' Entry `(i, j)` of the returned matrix is `k0(x_i, x_j)`. The statistic
#' itself is not computed here. Use [ksd_u_statistic()] to take the paper's
#' off-diagonal average, [ksd_u_bootstrap()] to make bootstrap draws from this
#' fixed matrix, or [ksd_u_test()] to run the full test in one call. See
#' [stein_kernel_matrix()] for the formula used to assemble `k0`.
#'
#' This function is the first lower-level step under [ksd_u_test()]. It is
#' exported because the Stein-kernel matrix is often the object one wants to
#' inspect when diagnosing a KSD test: large diagonal terms, nearly zero
#' off-diagonal terms, or a poor kernel scale are visible here before the
#' statistic and bootstrap compress the matrix to one number.
#'
#' The parallel V-statistic builder is [ksd_vq_matrix()]. Both builders evaluate
#' the same Stein-kernel formula through [stein_kernel_matrix()] and require the
#' same `X`, `score_function`, and kernel settings. The difference appears in
#' the next step: [ksd_u_statistic()] discards the diagonal, while
#' [ksd_v_statistic()] keeps it and reports the scaled `n V_n` statistic.
#' The `q` in `ksd_uq_matrix()` is a historical name inherited from the KSD
#' literature's `u_q` notation; the function still uses the target score
#' supplied by `score_function`, following the package-wide target-`p`
#' convention in this manual. In other words, the returned object is the
#' package's `K0` matrix with entries `k0(x_i, x_j)`, not an estimate involving
#' a second distribution named `q`.
#'
#' @param X Numeric vector or matrix of samples. A vector is treated as `n x 1`.
#' @param score_function Function returning the target score for each row of
#'   `X`.
#' @param scaling Positive kernel scale. `NULL` uses the square of the exact
#'   all-pair median distance.
#' @param kernel Kernel choice: `"gaussian_rbf"`, `"imq"`, or a `SteinKernel`
#'   object, including objects created by [custom_stein_kernel()].
#' @param imq_beta IMQ kernel exponent.
#' @return
#' Numeric `n x n` matrix. Row `i`, column `j` is `k0(x_i, x_j)` for the checked
#' sample matrix. The diagonal is included in the matrix because it is useful
#' for diagnostics, but [ksd_u_statistic()] ignores it when computing the
#' U-statistic. The matrix can be passed directly to [ksd_u_statistic()] and
#' [ksd_u_bootstrap()].
#' @examples
#' X <- matrix(rnorm(5), ncol = 1)
#' score_function <- function(x) -as.matrix(x)
#' ksd_uq_matrix(X, score_function)
#' @export
ksd_uq_matrix <- function(X, score_function,
                          scaling = NULL,
                          kernel = c("gaussian_rbf", "imq"),
                          imq_beta = -0.5) {
  prep <- prepare_ksd_u_inputs(
    X = X,
    score_function = score_function,
    scaling = scaling,
    kernel = kernel,
    imq_beta = imq_beta
  )

  ksd_uq_matrix_from_scores(prep$X, prep$grads, prep$kernel_obj)
}

#' Compute the KSD U-statistic from its Stein-kernel matrix
#'
#' Takes the pairwise Stein-kernel matrix and returns the off-diagonal average
#' used as the observed KSD statistic in the Liu et al. test.
#'
#' @details
#' If `U_mat[i, j] = k0(x_i, x_j)`, the diagonal is ignored and the returned
#' value is
#' \deqn{\frac{1}{n(n - 1)} \sum_{i \ne j} U_{ij}.}
#' This helper is useful when the matrix has already been built once and the
#' user wants to inspect or reuse the observed statistic separately from the
#' bootstrap. It is the matrix-compression layer between [ksd_uq_matrix()],
#' which knows about samples, scores, and kernels, and [ksd_u_bootstrap()],
#' which knows only about a fixed Stein-kernel matrix and bootstrap weights.
#'
#' In the U-statistic form the diagonal is deliberately removed. Those diagonal
#' entries are self-interactions `k0(x_i, x_i)`; they are present in the matrix
#' returned by [ksd_uq_matrix()] because they are useful diagnostics, but they
#' are not part of Liu et al.'s unbiased KSD estimator. This helper makes that
#' convention explicit, so a user who has built a matrix by hand can apply the
#' same statistic used by [ksd_u_test()].
#'
#' The function is deliberately small: it only validates that the input is a
#' square matrix, removes the diagonal contribution, and returns the average.
#' Keeping this step separate from [ksd_uq_matrix()] and [ksd_u_bootstrap()]
#' makes it possible to compare kernels or bootstrap choices while holding the
#' observed matrix fixed.
#'
#' @param U_mat Numeric square matrix of pairwise Stein kernel values.
#' @return
#' One numeric value: the off-diagonal average of `U_mat`. This is the same
#' number stored as `statistic` by [ksd_u_test()], before any bootstrap p-value
#' is computed. It is returned as a plain number because all reproducibility
#' information, such as the kernel scale and bootstrap settings, belongs to the
#' higher-level [ksd_u_test()] object rather than to this matrix-compression
#' step.
#' @examples
#' U <- matrix(c(1, 0.2, 0.3, 0.2, 1, 0.4, 0.3, 0.4, 1), 3, 3)
#' ksd_u_statistic(U)
#' @export
ksd_u_statistic <- function(U_mat) {
  U_mat <- validate_ksd_u_matrix(U_mat)
  n <- nrow(U_mat)
  diag(U_mat) <- 0
  sum(U_mat) / (n * (n - 1))
}

#' Bootstrap the KSD U-statistic from a Stein-kernel matrix
#'
#' Generates or accepts centered multinomial weights and applies them to the
#' U-statistic matrix. This is the bootstrap part of the Liu et al. KSD test
#' separated from matrix construction.
#'
#' @details
#' For each bootstrap weight vector `w`, the bootstrap statistic is computed as
#' \deqn{\sum_{i,j} w_i U_{ij} w_j,}
#' with the diagonal of `U_mat` set to zero. These values are on the same scale
#' as [ksd_u_statistic()]. The default weights are centered multinomial
#' proportions, `N_i / n - 1 / n`, matching Eq. 16 of Liu, Lee, and Jordan
#' (2016). The paper compares `n` times the observed and bootstrap statistics;
#' this package leaves both on the unmultiplied U-statistic scale, which gives
#' the same p-value. Supplying `W_mat` lets users reuse exactly the same
#' bootstrap weights across kernels or experiments.
#'
#' This function sits below [ksd_u_test()] and above the raw weight generator.
#' It exists so users can separate randomness in the bootstrap weights from the
#' deterministic matrix calculation. Supplying `W_mat` is the design point:
#' the same matrix of weights can be reused across several `U_mat` objects,
#' which is useful for paired comparisons of kernels or bandwidths.
#'
#' The parallel bootstrap helper is [ksd_v_bootstrap()]. The input matrix plays
#' the same role, but the weights have different meanings: `ksd_u_bootstrap()`
#' uses centered multinomial proportions for the degenerate U-statistic in Liu
#' et al., whereas [ksd_v_bootstrap()] uses wild-bootstrap sign sequences for
#' the V-statistic. To move from this function to the V version, keep a
#' Stein-kernel matrix but supply sign weights or choose a sign-generation
#' method instead of centered multinomial weights.
#'
#' @param U_mat Numeric square matrix of pairwise Stein kernel values.
#' @param nboot Number of bootstrap samples.
#' @param W_mat Optional `n x B` centered multinomial bootstrap weight matrix,
#'   with one row per observation and one column per bootstrap draw.
#' @param boot_method Bootstrap method. Currently only
#'   `"multinomial_centered"` is supported.
#' @return
#' Numeric vector of length `nboot` when `W_mat` is not supplied, or length
#' `ncol(W_mat)` when weights are supplied. Each entry is one bootstrap
#' U-statistic on the same scale as [ksd_u_statistic()], so it can be compared
#' directly with the observed statistic.
#' @examples
#' U <- matrix(c(1, 0.2, 0.3, 0.2, 1, 0.4, 0.3, 0.4, 1), 3, 3)
#' ksd_u_bootstrap(U, nboot = 5)
#' @export
ksd_u_bootstrap <- function(U_mat, nboot = 1000, W_mat = NULL,
                            boot_method = "multinomial_centered") {
  boot_method <- resolve_ksd_u_boot_method(boot_method)
  U_mat <- validate_ksd_u_matrix(U_mat)
  n <- nrow(U_mat)
  diag(U_mat) <- 0

  if (is.null(W_mat)) {
    nboot <- validate_positive_integer(nboot, "nboot")
    W_mat <- generate_bootstrap_weights(n, nboot, boot_method = boot_method)
  } else {
    W_mat <- as.matrix(W_mat)
    if (!is.numeric(W_mat) || nrow(W_mat) != n || ncol(W_mat) < 1) {
      stop("W_mat must be a numeric matrix with nrow(U_mat) rows and at least one column")
    }
  }

  # Liu, Lee & Jordan (2016), Eq. 16.
  as.numeric(colSums((U_mat %*% W_mat) * W_mat))
}

#' Prepare samples, scores, and kernel for KSD U-statistic routines
#'
#' Checks the sample and score shapes, builds or accepts the requested Stein
#' kernel, and chooses a default scale when needed.
#'
#' @details
#' A vector `X` is treated as one-dimensional data. If `kernel` is one of the
#' built-in names and `scaling` is `NULL`, the square of the median distance
#' between all sample-row pairs is used as the squared kernel scale. Built-in Gaussian RBF
#' kernels use `h^2 = scaling`; built-in IMQ kernels use `c^2 = scaling`.
#' For a Gaussian RBF object with a preconditioner and no fixed bandwidth, the
#' median is computed in its preconditioned distance and the object is updated
#' without discarding its other settings.
#'
#' `kernel` may also be a ready-made `SteinKernel` object. This includes custom
#' kernels created by [custom_stein_kernel()], as long as they provide the base
#' kernel, the gradient with respect to the first input, and the mixed
#' derivative trace needed to assemble the Stein kernel matrix. In that case
#' the object is used directly. The `scaling` argument is not inserted into a
#' custom kernel, so any bandwidth or other tuning constant should be captured
#' inside the custom kernel functions.
#'
#' @param X Numeric vector or matrix of samples. A vector is treated as `n x 1`.
#' @param score_function Function returning the target score for each row of
#'   `X`.
#' @param scaling Positive kernel scale. `NULL` uses the square of the exact
#'   all-pair median distance.
#' @param kernel Kernel choice: `"gaussian_rbf"`, `"imq"`, or a `SteinKernel`
#'   object, including objects created by [custom_stein_kernel()].
#' @param imq_beta IMQ kernel exponent.
#'
#' @return A list with checked samples, scores, kernel object, kernel name, and
#'   selected scale.
#' @noRd
prepare_ksd_u_inputs <- function(X, score_function, scaling = NULL,
                                 kernel = c("gaussian_rbf", "imq"),
                                 imq_beta = -0.5) {
  x_mat <- validate_samples(X, min_rows = 2)
  grads <- validate_scores(score_function, x_mat)

  if (inherits(kernel, "SteinKernel")) {
    kernel_obj <- kernel
    kernel_name <- kernel_type_name(kernel_obj)
    kernel_scale <- kernel_scaling_value(kernel_obj)
    if (inherits(kernel_obj, "SteinKernel_gaussian_rbf") && !is.finite(kernel_scale)) {
      if (is.null(scaling)) {
        scaling <- resolve_gaussian_rbf_h2(kernel_obj, x_mat)
      }
      kernel_obj$h2 <- validate_kernel_squared_scale(scaling)
    } else {
      scaling <- kernel_scale
    }
  } else {
    kernel_name <- match.arg(kernel, c("gaussian_rbf", "imq"))
    if (is.null(scaling)) {
      scaling <- find_median_distance(x_mat)
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

#' Resolve the KSD U bootstrap method
#'
#' The current U-statistic implementation supports centered multinomial
#' bootstrap weights.
#'
#' @param boot_method Bootstrap method name.
#'
#' @return The checked method name.
#' @noRd
resolve_ksd_u_boot_method <- function(boot_method) {
  if (!identical(boot_method, "multinomial_centered")) {
    stop("boot_method must be 'multinomial_centered'")
  }
  boot_method
}

#' Check a KSD U-statistic matrix
#'
#' Converts the input to a matrix and checks that it is square with at least two
#' rows.
#'
#' @param U_mat Candidate KSD matrix.
#'
#' @return A numeric square matrix.
#' @noRd
validate_ksd_u_matrix <- function(U_mat) {
  if (is.null(dim(U_mat))) {
    stop("U_mat must be a square numeric matrix")
  }
  U_mat <- as.matrix(U_mat)
  if (!is.numeric(U_mat) || nrow(U_mat) != ncol(U_mat) || nrow(U_mat) < 2) {
    stop("U_mat must be a square numeric matrix with at least two rows")
  }
  U_mat
}

#' Build a KSD U matrix from precomputed scores
#'
#' This is the lower-level version of [ksd_uq_matrix()] used after inputs have
#' already been checked.
#'
#' @param X Numeric sample matrix.
#' @param grads Score matrix for `X`.
#' @param kernel_obj Stein kernel object.
#'
#' @return Numeric matrix of pairwise Stein kernel values.
#' @noRd
ksd_uq_matrix_from_scores <- function(X, grads, kernel_obj) {
  stein_kernel_matrix(kernel_obj, X, grads)
}


#' Compute the KSD U statistic and bootstrap values
#'
#' Runs the dense or blocked calculation path after inputs and bootstrap weights
#' have been prepared.
#'
#' @param X Numeric sample matrix.
#' @param grads Score matrix for `X`.
#' @param kernel_obj Stein kernel object.
#' @param W_mat Bootstrap weight matrix.
#' @param boot_method Bootstrap method name.
#' @param use_block Logical; whether to use blockwise matrix calculations.
#' @param block_size Number of rows and columns in each block.
#'
#' @return A list with the observed statistic, p-value, and raw bootstrap
#'   statistics.
#' @noRd
internal_u_compute_engine <- function(X, grads, kernel_obj, W_mat, boot_method, use_block, block_size) {
  if (use_block) {
    return(compute_u_blocked_pure(X, grads, kernel_obj, W_mat, block_size))
  }

  U_mat <- ksd_uq_matrix_from_scores(X, grads, kernel_obj)
  stat <- ksd_u_statistic(U_mat)
  boot_stats <- ksd_u_bootstrap(U_mat, W_mat = W_mat, boot_method = boot_method)

  list(
    stat = unname(stat),
    pval = bootstrap_pvalue_right_tail(boot_stats, stat),
    raw_boots = as.numeric(boot_stats)
  )
}

#' Compute the KSD U statistic in matrix blocks
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
#' @return A list with the observed statistic, p-value, and raw bootstrap
#'   statistics.
#' @noRd
compute_u_blocked_pure <- function(X, grads, kernel_obj, W_mat, block_size) {
  n <- nrow(X)
  nboot <- ncol(W_mat)
  i_starts <- seq(1, n, by = block_size)
  j_starts <- seq(1, n, by = block_size)

  stat_sum <- 0
  boot_stats <- numeric(nboot)

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

      m_block <- compute_u_block_pure(X_i, X_j, g_i, g_j, kernel_obj)

      if (i_start == j_start) {
        # The U-statistic is off-diagonal, so k0(x_i, x_i) never contributes.
        diag(m_block) <- 0
      }

      stat_sum <- stat_sum + sum(m_block)
      boot_stats <- boot_stats + colSums((m_block %*% W_j) * W_i)
    }
  }

  stat <- stat_sum / (n * (n - 1))
  list(
    stat = stat,
    pval = bootstrap_pvalue_right_tail(boot_stats, stat),
    raw_boots = as.numeric(boot_stats)
  )
}

#' Compute one KSD U matrix block
#'
#' Evaluates the Stein kernel between two blocks of rows.
#'
#' @param X_i,X_j Sample blocks.
#' @param g_i,g_j Score blocks matching `X_i` and `X_j`.
#' @param kernel_obj Stein kernel object.
#'
#' @return Numeric matrix with `nrow(X_i)` rows and `nrow(X_j)` columns.
#' @noRd
compute_u_block_pure <- function(X_i, X_j, g_i, g_j, kernel_obj) {
  stein_kernel_matrix(kernel_obj, X_i, g_i, X_j, g_j)
}
