#' Kernel Stein Discrepancy GOF Test (V-statistics)
#'
#' @param X Numeric vector or matrix of samples.
#' @param score_function Function returning score values.
#' @param boot_method One of `"rademacher"` or `"markov"`.
#' @param scaling Positive kernel scaling parameter. NULL uses median heuristic.
#' @param nboot Number of bootstrap samples.
#' @param change_prob Markov sign-flip probability. Use `0.5` for the iid
#'   special case. For a dependent MCMC chain, Chwialkowski et al. (2016)
#'   recommend thinning outside this function until lag-one autocorrelation is
#'   below `0.5`, then passing the chosen `a_n` here, e.g. `0.1` for their
#'   Figure 3 setup.
#' @param kernel Kernel choice: `"gaussian_rbf"`, `"imq"`, or instantiated SteinKernel.
#' @param return_raw_boot Logical; if TRUE, appends raw bootstrap samples to output.
#' @param median_max_samples,median_use_sampling,median_seed Controls for the
#'   median-distance heuristic when `scaling = NULL`.
#' @param block_size,block_threshold Controls for blocked matrix computation.
#' @param imq_beta IMQ kernel exponent.
#' @return An object of class `htest`.
#' @examples
#' X <- matrix(rnorm(20), ncol = 1)
#' score_function <- function(x) -as.matrix(x)
#' ksd_v_test(X, score_function, nboot = 10)
#' @export
ksd_v_test <- function(X, score_function,
                       boot_method = c("rademacher", "markov"),
                       scaling = NULL,
                       nboot = 1000,
                       change_prob = 0.5,
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

#' KSD V-statistic kernel matrix (Theorem 2.1)
#'
#' @param X Numeric vector or matrix of samples.
#' @param score_function Function returning score values.
#' @param scaling Positive kernel scaling parameter. NULL uses median heuristic.
#' @param kernel Kernel choice: `"gaussian_rbf"`, `"imq"`, or instantiated SteinKernel.
#' @param median_max_samples,median_use_sampling,median_seed Controls for the
#'   median-distance heuristic when `scaling = NULL`.
#' @param imq_beta IMQ kernel exponent.
#' @return Numeric matrix with entries `u_q(x_i, x_j)`.
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

#' KSD V-statistic
#'
#' @param U_mat Numeric square matrix with entries `u_q(x_i, x_j)`.
#' @return Numeric scalar `n * mean(U_mat)`.
#' @examples
#' U <- matrix(c(1, 0.2, 0.3, 0.2, 1, 0.4, 0.3, 0.4, 1), 3, 3)
#' ksd_v_statistic(U)
#' @export
ksd_v_statistic <- function(U_mat) {
  U_mat <- validate_ksd_v_matrix(U_mat)
  nrow(U_mat) * mean(U_mat)
}

#' KSD V-statistic bootstrap samples
#'
#' @param U_mat Numeric square matrix with entries `u_q(x_i, x_j)`.
#' @param W_mat Optional wild bootstrap weight matrix.
#' @param nboot Number of bootstrap samples.
#' @param boot_method One of `"rademacher"` or `"markov"` when `W_mat` is NULL.
#' @param change_prob Markov sign-flip probability when `boot_method = "markov"`.
#' @return Numeric vector of bootstrap samples on the V-statistic scale.
#' @examples
#' U <- matrix(c(1, 0.2, 0.3, 0.2, 1, 0.4, 0.3, 0.4, 1), 3, 3)
#' ksd_v_bootstrap(U, nboot = 5)
#' @export
ksd_v_bootstrap <- function(U_mat, W_mat = NULL, nboot = 1000,
                            boot_method = c("rademacher", "markov"),
                            change_prob = 0.5) {
  boot_method <- match.arg(boot_method)
  U_mat <- validate_ksd_v_matrix(U_mat)
  n <- nrow(U_mat)

  if (is.null(W_mat)) {
    if (!is.numeric(nboot) || length(nboot) != 1 || !is.finite(nboot) || nboot <= 0) {
      stop("nboot must be a positive scalar")
    }
    nboot <- as.integer(nboot)
    W_mat <- generate_bootstrap_weights(n, nboot, boot_method = boot_method,
                                        change_prob = change_prob)
  } else {
    W_mat <- as.matrix(W_mat)
    if (!is.numeric(W_mat) || nrow(W_mat) != n || ncol(W_mat) < 1) {
      stop("W_mat must be a numeric matrix with nrow(U_mat) rows and at least one column")
    }
  }

  as.numeric(colSums((U_mat %*% W_mat) * W_mat) / n)
}


#' @keywords internal
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

#' @keywords internal
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

#' @keywords internal
ksd_vq_matrix_from_scores <- function(X, grads, kernel_obj) {
  stein_kernel_matrix(kernel_obj, X, grads)
}

#' @keywords internal
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

#' @keywords internal
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

#' @keywords internal
compute_v_block_pure <- function(X_i, X_j, g_i, g_j, kernel_obj) {
  k_block <- eval_kernel(kernel_obj, X_i, X_j)
  m_block <- stein_kernel_matrix(kernel_obj, X_i, g_i, X_j, g_j)
  list(m_block = m_block, k_block = k_block)
}
