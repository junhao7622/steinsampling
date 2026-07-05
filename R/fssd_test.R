# ---- Public API ---------------------------------------------------------

#' Finite Set Stein Discrepancy (FSSD) Goodness-of-Fit Test
#'
#' Dispatcher between FSSD-rand and FSSD-opt.
#'
#' @param X Numeric vector or matrix of samples (`n x d`).
#' @param score_function Function returning `nabla_x log p(x)` with shape
#'   `n x d` (or length-`n` when `d = 1`).
#' @param variant `"opt"` learns test locations and kernel scale on a training
#'   split; `"rand"` draws test locations from a Gaussian fitted to the data.
#' @param J Number of test locations.
#' @param nboot Null-simulation draws for the chi-square mixture.
#' @param kernel `"gaussian_rbf"`, `"imq"`, or a `SteinKernel` object.
#' @param scaling Optional positive squared kernel scale. `NULL` uses the
#'   median squared-distance heuristic.
#' @param train_ratio FSSD-opt training fraction.
#' @param eval_on FSSD-opt evaluation sample. `"test"` is the methodologically
#'   valid held-out test described in the paper. `"all"` evaluates on the full
#'   sample after tuning and is diagnostic only.
#' @param gamma Fixed FSSD-opt regularizer added outside the square root:
#'   `FSSD^2 / (sqrt(var_H1) + gamma)`.
#' @param maxit FSSD-opt L-BFGS-B iterations.
#' @param locs_bounds_frac Test-location box expansion in standard deviations.
#' @param scale_lower,scale_upper Absolute bounds for the optimized squared
#'   kernel scale.
#' @param seed Optional RNG seed for data split and random locations.
#' @param median_max_samples,median_use_sampling,median_seed Forwarded to
#'   `find_median_distance`.
#' @param imq_beta IMQ exponent.
#'
#' @return An object of class `htest`.
#' @examples
#' X <- matrix(rnorm(8), ncol = 1)
#' score_function <- function(x) -as.matrix(x)
#' fssd_test(X, score_function, variant = "rand", J = 1, nboot = 10)
#' @export
fssd_test <- function(X, score_function,
                      variant = c("opt", "rand"),
                      J = 5,
                      nboot = 2000,
                      kernel = c("gaussian_rbf", "imq"),
                      scaling = NULL,
                      train_ratio = 0.2,
                      eval_on = c("test", "all"),
                      gamma = 1e-4,
                      maxit = 100,
                      locs_bounds_frac = 10,
                      scale_lower = 1e-1,
                      scale_upper = 1e4,
                      seed = NULL,
                      median_max_samples = 2000,
                      median_use_sampling = TRUE,
                      median_seed = 123,
                      imq_beta = -0.5) {
  variant <- match.arg(variant)
  eval_on <- match.arg(eval_on)

  if (variant == "opt") {
    fssd_opt_test(
      X = X, score_function = score_function, J = J, nboot = nboot,
      kernel = kernel, scaling = scaling,
      train_ratio = train_ratio, eval_on = eval_on,
      gamma = gamma, maxit = maxit,
      locs_bounds_frac = locs_bounds_frac,
      scale_lower = scale_lower, scale_upper = scale_upper,
      seed = seed,
      median_max_samples = median_max_samples,
      median_use_sampling = median_use_sampling,
      median_seed = median_seed,
      imq_beta = imq_beta
    )
  } else {
    fssd_rand_test(
      X = X, score_function = score_function, J = J, nboot = nboot,
      kernel = kernel, scaling = scaling, seed = seed,
      median_max_samples = median_max_samples,
      median_use_sampling = median_use_sampling,
      median_seed = median_seed,
      imq_beta = imq_beta
    )
  }
}


#' FSSD-rand: random test locations and median-heuristic kernel scale.
#' @inheritParams fssd_test
#' @return An object of class `htest`.
#' @examples
#' X <- matrix(rnorm(8), ncol = 1)
#' score_function <- function(x) -as.matrix(x)
#' fssd_rand_test(X, score_function, J = 1, nboot = 10)
#' @export
fssd_rand_test <- function(X, score_function,
                           J = 5,
                           nboot = 2000,
                           kernel = c("gaussian_rbf", "imq"),
                           scaling = NULL,
                           seed = NULL,
                           median_max_samples = 2000,
                           median_use_sampling = TRUE,
                           median_seed = 123,
                           imq_beta = -0.5) {
  data_name <- deparse(substitute(X))

  run_rand <- function() {
    prep <- prepare_fssd_inputs(
      X = X,
      score_function = score_function,
      min_rows = 2,
      kernel = kernel,
      scaling = scaling,
      median_max_samples = median_max_samples,
      median_use_sampling = median_use_sampling,
      median_seed = median_seed,
      imq_beta = imq_beta
    )
    J <- validate_positive_integer(J, "J")
    nboot <- validate_positive_integer(nboot, "nboot")

    V <- fssd_init_locations_from_fit(prep$X, J)
    engine_res <- internal_fssd_compute_engine(
      X = prep$X,
      grads = prep$grads,
      V = V,
      kernel_obj = prep$kernel_obj,
      n_simulations = nboot
    )

    structure(list(
      statistic = c(fssd = engine_res$fssd_stat),
      p.value = engine_res$p_value,
      method = sprintf("Finite Set Stein Discrepancy (rand) - %s", prep$kernel_name),
      data.name = data_name,
      parameter = c(
        nboot = nboot,
        J = J,
        scaling = if (is.null(prep$scaling)) NA_real_ else prep$scaling
      ),
      info = list(variant = "rand", V = V, kernel = prep$kernel_name)
    ), class = "htest")
  }

  if (is.null(seed)) run_rand() else with_local_seed(seed, run_rand())
}


#' FSSD-opt: train/test split plus bounded optimization.
#' @inheritParams fssd_test
#' @return An object of class `htest`.
#' @examples
#' X <- matrix(rnorm(12), ncol = 1)
#' score_function <- function(x) -as.matrix(x)
#' fssd_opt_test(X, score_function, J = 1, nboot = 10, maxit = 1,
#'               train_ratio = 0.5)
#' @export
fssd_opt_test <- function(X, score_function,
                          J = 5,
                          nboot = 2000,
                          kernel = c("gaussian_rbf", "imq"),
                          scaling = NULL,
                          train_ratio = 0.2,
                          eval_on = c("test", "all"),
                          gamma = 1e-4,
                          maxit = 100,
                          locs_bounds_frac = 10,
                          scale_lower = 1e-1,
                          scale_upper = 1e4,
                          seed = NULL,
                          median_max_samples = 2000,
                          median_use_sampling = TRUE,
                          median_seed = 123,
                          imq_beta = -0.5) {
  data_name <- deparse(substitute(X))

  run_opt <- function() {
    x_mat <- validate_samples(X, min_rows = 4)
    grads <- validate_scores(score_function, x_mat)
    n <- nrow(x_mat)

    J <- validate_positive_integer(J, "J")
    nboot <- validate_positive_integer(nboot, "nboot")
    eval_on <- match.arg(eval_on, c("test", "all"))
    maxit <- validate_positive_integer(maxit, "maxit")
    gamma <- validate_fssd_positive_scalar(gamma, "gamma")
    locs_bounds_frac <- validate_fssd_positive_scalar(locs_bounds_frac, "locs_bounds_frac")
    scale_lower <- validate_fssd_positive_scalar(scale_lower, "scale_lower")
    scale_upper <- validate_fssd_positive_scalar(scale_upper, "scale_upper")
    if (scale_lower >= scale_upper) stop("scale_lower must be smaller than scale_upper")
    train_ratio <- validate_fssd_train_ratio(train_ratio)

    n_train <- max(2L, min(floor(train_ratio * n), n - 2L))
    idx_train <- sample.int(n, size = n_train, replace = FALSE)
    idx_test <- setdiff(seq_len(n), idx_train)

    x_train <- x_mat[idx_train, , drop = FALSE]
    x_test <- x_mat[idx_test, , drop = FALSE]
    grads_train <- grads[idx_train, , drop = FALSE]
    grads_test <- grads[idx_test, , drop = FALSE]

    prep <- prepare_fssd_kernel(
      X = x_train,
      kernel = kernel,
      scaling = scaling,
      median_max_samples = median_max_samples,
      median_use_sampling = median_use_sampling,
      median_seed = median_seed,
      imq_beta = imq_beta
    )

    opt <- internal_fssd_optimize_engine(
      X_train = x_train,
      grads_train = grads_train,
      J = J,
      kernel_obj = prep$kernel_obj,
      gamma = gamma,
      maxit = maxit,
      locs_bounds_frac = locs_bounds_frac,
      scale_lower = scale_lower,
      scale_upper = scale_upper,
      median_max_samples = median_max_samples,
      median_use_sampling = median_use_sampling,
      median_seed = median_seed
    )

    x_eval <- if (eval_on == "all") x_mat else x_test
    grads_eval <- if (eval_on == "all") grads else grads_test
    engine_res <- internal_fssd_compute_engine(
      X = x_eval,
      grads = grads_eval,
      V = opt$V_opt,
      kernel_obj = opt$kernel_obj_opt,
      n_simulations = nboot
    )

    scaling_opt <- if (is.finite(opt$sigma2_opt)) opt$sigma2_opt
                   else kernel_scaling_value(opt$kernel_obj_opt)

    structure(list(
      statistic = c(fssd = engine_res$fssd_stat),
      p.value = engine_res$p_value,
      method = sprintf("Finite Set Stein Discrepancy (opt) - %s", prep$kernel_name),
      data.name = data_name,
      parameter = c(
        nboot = nboot,
        J = J,
        train_ratio = train_ratio,
        eval_all = if (eval_on == "all") 1 else 0,
        gamma = gamma,
        scaling = if (is.null(scaling_opt)) NA_real_ else scaling_opt
      ),
      info = list(
        variant = "opt",
        eval_on = eval_on,
        V = opt$V_opt,
        kernel = prep$kernel_name,
        sigma2_opt = opt$sigma2_opt,
        objective_opt = opt$objective_opt,
        iterations = opt$iterations,
        convergence = opt$convergence,
        scale_grid = opt$scale_grid,
        scale_grid_objectives = opt$scale_grid_objectives
      )
    ), class = "htest")
  }

  if (is.null(seed)) run_opt() else with_local_seed(seed, run_opt())
}


# ---- Public Computation Primitives --------------------------------------

#' Compute the FSSD feature matrix tau (n x dJ).
#'
#' Row i is `tau(x_i) = vec(Xi(x_i))` with
#' `[Xi(x_i)]_{l,j} = xi_{p,l}(x_i, v_j) / sqrt(dJ)`, where
#' `xi_{p,l}(x, v) = score_l(x) * k(x, v) + d/dx_l k(x, v)`.
#'
#' @param X Numeric sample matrix.
#' @param grads Score matrix for `X`.
#' @param V Matrix of FSSD test locations.
#' @param kernel_obj Stein kernel object.
#'
#' @return Numeric FSSD feature matrix.
#' @examples
#' X <- matrix(c(-1, 0, 1), ncol = 1)
#' grads <- -X
#' V <- matrix(0, ncol = 1)
#' kernel <- stein_kernel(type = "gaussian_rbf", h = 1)
#' compute_tau(X, grads, V, kernel)
#' @export
compute_tau <- function(X, grads, V, kernel_obj) {
  x_mat <- as.matrix(X)
  grads_mat <- as.matrix(grads)
  v_mat <- as.matrix(V)
  if (!is.numeric(x_mat) || !is.numeric(grads_mat) || !is.numeric(v_mat)) {
    stop("X, grads, and V must be numeric matrices")
  }
  if (nrow(x_mat) != nrow(grads_mat) || ncol(x_mat) != ncol(grads_mat)) {
    stop("X and grads must have the same dimensions")
  }
  if (ncol(v_mat) != ncol(x_mat) || nrow(v_mat) < 1) {
    stop("V must be a numeric matrix with ncol(V) == ncol(X)")
  }
  if (!inherits(kernel_obj, "SteinKernel")) {
    stop("kernel_obj must be a SteinKernel object")
  }

  n <- nrow(x_mat)
  d <- ncol(x_mat)
  J <- nrow(v_mat)
  scale_fac <- 1 / sqrt(d * J)
  tau <- matrix(0, nrow = n, ncol = d * J)

  for (j in seq_len(J)) {
    v_one <- matrix(v_mat[j, ], nrow = 1, ncol = d)
    k_xv <- eval_kernel(kernel_obj, x_mat, v_one)
    grad_arr <- grad_x_kernel(kernel_obj, x_mat, v_one)

    grad_k <- matrix(0, nrow = n, ncol = d)
    for (l in seq_len(d)) grad_k[, l] <- grad_arr[, 1, l]

    xi <- (grads_mat * as.numeric(k_xv[, 1]) + grad_k) * scale_fac
    tau[, ((j - 1L) * d + 1L):(j * d)] <- xi
  }

  tau
}


#' Unbiased estimator of FSSD^2 (paper Eq. 2).
#'
#' @param tau_matrix Numeric FSSD feature matrix.
#'
#' @return Numeric scalar.
#' @examples
#' tau <- matrix(c(1, 2, 3), ncol = 1)
#' compute_fssd_unbiased_stat(tau)
#' @export
compute_fssd_unbiased_stat <- function(tau_matrix) {
  tau <- as.matrix(tau_matrix)
  n <- nrow(tau)
  if (n < 2) stop("Need at least two rows to compute the unbiased FSSD statistic")

  sum_tau <- colSums(tau)
  as.numeric((sum(sum_tau * sum_tau) - sum(rowSums(tau * tau))) / (n * (n - 1)))
}


#' Simulate the asymptotic null distribution from Theorem 3.
#'
#' @param tau_matrix Numeric FSSD feature matrix.
#' @param fssd_stat Observed FSSD statistic.
#' @param n_simulations Number of null simulations.
#'
#' @return P-value for a right-tail FSSD test.
#' @examples
#' tau <- matrix(c(1, 2, 3), ncol = 1)
#' compute_fssd_null_pvalue(tau, fssd_stat = 0.1, n_simulations = 10)
#' @export
compute_fssd_null_pvalue <- function(tau_matrix, fssd_stat, n_simulations = 2000) {
  tau <- as.matrix(tau_matrix)
  if (!is.numeric(tau) || nrow(tau) < 2 || ncol(tau) < 1) {
    stop("tau_matrix must be a numeric matrix with at least two rows")
  }
  if (!is.numeric(fssd_stat) || length(fssd_stat) != 1 || !is.finite(fssd_stat)) {
    stop("fssd_stat must be a finite scalar")
  }
  n_simulations <- validate_positive_integer(n_simulations, "n_simulations")

  sigma_q <- fssd_tau_covariance(tau)
  eigvals <- eigen(sigma_q, symmetric = TRUE, only.values = TRUE)$values
  eigvals[eigvals < 0] <- 0

  z <- matrix(stats::rnorm(n_simulations * length(eigvals)),
              nrow = n_simulations, ncol = length(eigvals))
  chi_terms <- z^2 - 1
  s_null <- as.numeric(chi_terms %*% eigvals)
  test_score <- nrow(tau) * fssd_stat

  list(
    p_value = mean(s_null > test_score),
    test_score = test_score,
    eigenvalues = eigvals
  )
}


# ---- Computation Engines -------------------------------------------------

#' @keywords internal
internal_fssd_compute_engine <- function(X, grads, V, kernel_obj, n_simulations) {
  tau <- compute_tau(X = X, grads = grads, V = V, kernel_obj = kernel_obj)
  fssd_stat <- compute_fssd_unbiased_stat(tau)
  null_res <- compute_fssd_null_pvalue(
    tau_matrix = tau,
    fssd_stat = fssd_stat,
    n_simulations = n_simulations
  )

  list(fssd_stat = fssd_stat, p_value = null_res$p_value)
}


#' Maximize the paper FSSD power criterion.
#'
#' This follows the paper's Section 3.2 optimization structure: initialize
#' locations from a Gaussian fit, grid-search the squared kernel scale, then run
#' bounded ascent on a root-regularized power criterion. This changes only the
#' optimizer objective; the final test statistic and null distribution are
#' unchanged.
#'
#' @keywords internal
internal_fssd_optimize_engine <- function(X_train, grads_train, J, kernel_obj,
                                          gamma = 1e-4,
                                          maxit = 100,
                                          locs_bounds_frac = 10,
                                          scale_lower = 1e-1,
                                          scale_upper = 1e4,
                                          median_max_samples = 2000,
                                          median_use_sampling = TRUE,
                                          median_seed = 123) {
  x_train <- validate_samples(X_train, min_rows = 4)
  grads <- as.matrix(grads_train)
  if (!is.numeric(grads) || nrow(grads) != nrow(x_train) || ncol(grads) != ncol(x_train)) {
    stop("grads_train must be a numeric matrix with the same shape as X_train")
  }
  if (!inherits(kernel_obj, "SteinKernel")) stop("kernel_obj must be a SteinKernel object")

  J <- validate_positive_integer(J, "J")
  maxit <- validate_positive_integer(maxit, "maxit")
  gamma <- validate_fssd_positive_scalar(gamma, "gamma")
  locs_bounds_frac <- validate_fssd_positive_scalar(locs_bounds_frac, "locs_bounds_frac")
  scale_lower <- validate_fssd_positive_scalar(scale_lower, "scale_lower")
  scale_upper <- validate_fssd_positive_scalar(scale_upper, "scale_upper")
  if (scale_lower >= scale_upper) stop("scale_lower must be smaller than scale_upper")

  n <- nrow(x_train)
  d <- ncol(x_train)
  has_kernel_param <- inherits(kernel_obj, "SteinKernel_gaussian_rbf") ||
    inherits(kernel_obj, "SteinKernel_imq")
  n_params <- d * J + if (has_kernel_param) 1L else 0L
  if (n < n_params) {
    warning(sprintf(
      "Number of optimization parameters (%s) exceeds the training sample size; FSSD-opt is ill-posed.",
      if (has_kernel_param) "d*J+1" else "d*J"
    ), call. = FALSE)
  }

  v_init <- fssd_init_locations_from_fit(x_train, J, cov_reg = 1e-6)
  scale2_init <- if (has_kernel_param) kernel_scaling_value(kernel_obj) else NA_real_
  scale_grid <- scale_grid_objectives <- NULL

  if (has_kernel_param) {
    scale_grid <- fssd_scale_grid(
      X = x_train,
      max_samples = min(as.integer(median_max_samples), 1000L),
      use_sampling = median_use_sampling,
      seed = median_seed
    )
    grid_res <- fssd_grid_search_scale(
      X = x_train,
      grads = grads,
      V = v_init,
      kernel_obj_base = kernel_obj,
      scale_grid = scale_grid,
      gamma = gamma
    )
    scale2_init <- grid_res$scale
    scale_grid_objectives <- grid_res$objectives
  }

  bounds <- fssd_optimization_bounds(
    X = x_train,
    J = J,
    has_kernel_param = has_kernel_param,
    locs_bounds_frac = locs_bounds_frac,
    scale_lower = scale_lower,
    scale_upper = scale_upper
  )

  opt_bounds <- bounds
  if (has_kernel_param) {
    scale_idx <- J * d + 1L
    opt_bounds$lower[scale_idx] <- sqrt(opt_bounds$lower[scale_idx])
    opt_bounds$upper[scale_idx] <- sqrt(opt_bounds$upper[scale_idx])
  }

  # L-BFGS-B sees sqrt(scale2) for better conditioning. The kernel itself
  # still uses the squared scale.
  par0 <- if (has_kernel_param) c(as.numeric(t(v_init)), sqrt(scale2_init))
          else as.numeric(t(v_init))
  par0 <- fssd_project_parameters(par0, opt_bounds$lower, opt_bounds$upper)

  cache <- new.env(parent = emptyenv())
  objective_at <- function(par) {
    if (!is.null(cache$par) && length(cache$par) == length(par) && all(cache$par == par)) {
      return(cache$og)
    }
    par_obj <- par
    if (has_kernel_param) par_obj[scale_idx] <- par[scale_idx]^2
    og <- fssd_objective_and_grad(
      par = par_obj,
      X = x_train,
      grads = grads,
      J = J,
      kernel_obj_base = kernel_obj,
      has_kernel_param = has_kernel_param,
      gamma = gamma
    )
    if (has_kernel_param) og$grad[scale_idx] <- og$grad[scale_idx] * (2 * par[scale_idx])
    cache$par <- par
    cache$og <- og
    og
  }

  opt_res <- stats::optim(
    par = par0,
    fn = function(par) -objective_at(par)$value,
    gr = function(par) -objective_at(par)$grad,
    method = "L-BFGS-B",
    lower = opt_bounds$lower,
    upper = opt_bounds$upper,
    control = list(
      maxit = maxit,
      pgtol = 1e-7,
      factr = 1e-8 / .Machine$double.eps
    )
  )

  best_par <- opt_res$par
  best_og <- objective_at(best_par)
  best_value <- best_og$value

  if (has_kernel_param) {
    V_opt <- matrix(best_par[seq_len(J * d)], nrow = J, ncol = d, byrow = TRUE)
    sigma2_opt <- best_par[J * d + 1L]^2
  } else {
    sigma2_opt <- NA_real_
    V_opt <- matrix(best_par, nrow = J, ncol = d, byrow = TRUE)
  }

  list(
    V_opt = V_opt,
    sigma2_opt = sigma2_opt,
    kernel_obj_opt = fssd_kernel_with_scale(kernel_obj, sigma2_opt, has_kernel_param),
    objective_opt = best_value,
    iterations = unname(opt_res$counts[["function"]]),
    convergence = opt_res$convergence,
    scale_grid = scale_grid,
    scale_grid_objectives = scale_grid_objectives
  )
}


#' Objective and analytic gradient for the FSSD-opt training criterion.
#'
#' `par` packs `vec(V)` and, for built-in kernels, the squared kernel scale
#' directly: `h^2` for Gaussian RBF and `c^2` for IMQ. The optimizer uses
#' `FSSD^2 / (sqrt(var_H1) + gamma)`, with fixed `gamma`.
#'
#' @keywords internal
fssd_objective_and_grad <- function(par, X, grads, J, kernel_obj_base,
                                    has_kernel_param, gamma) {
  gamma <- validate_fssd_positive_scalar(gamma, "gamma")
  d <- ncol(X)
  n_local <- nrow(X)

  V <- matrix(par[seq_len(J * d)], nrow = J, ncol = d, byrow = TRUE)
  scale2_k <- if (has_kernel_param) par[J * d + 1L] else NA_real_

  kernel_obj_current <- fssd_kernel_with_scale(kernel_obj_base, scale2_k, has_kernel_param)
  tau <- compute_tau(X, grads, V, kernel_obj_current)
  p_local <- ncol(tau)

  mu_tau <- colMeans(tau)
  sum_tau <- colSums(tau)
  fssd2 <- compute_fssd_unbiased_stat(tau)
  grad_f <- (2 / (n_local * (n_local - 1))) *
    (matrix(sum_tau, n_local, p_local, byrow = TRUE) - tau)

  centered_tau <- sweep(tau, 2, mu_tau, "-")
  sigma_tau <- crossprod(centered_tau) / n_local
  sigma_h1_sq <- as.numeric(4 * t(mu_tau) %*% sigma_tau %*% mu_tau)
  sigma_h1 <- sqrt(max(sigma_h1_sq, 0))
  denom <- sigma_h1 + gamma

  sigma_mu <- as.numeric(sigma_tau %*% mu_tau)
  c_mu <- as.numeric(centered_tau %*% mu_tau)
  grad_sigma_sq <-
    (8 / n_local) * matrix(sigma_mu, n_local, p_local, byrow = TRUE) +
    (8 / n_local) * (c_mu %o% mu_tau)

  d_denom <- if (sigma_h1 > 1e-12) grad_sigma_sq / (2 * sigma_h1)
             else matrix(0, n_local, p_local)
  grad_obj_tau <- grad_f / denom - (fssd2 / (denom^2)) * d_denom

  scale_fac <- 1 / sqrt(d * J)
  grad_v <- matrix(0, nrow = J, ncol = d)
  grad_param <- 0

  for (j in seq_len(J)) {
    g_block <- grad_obj_tau[, ((j - 1L) * d + 1L):(j * d), drop = FALSE]
    fssd_grads <- grad_theta_v_kernel(
      obj = kernel_obj_current,
      X = X,
      vj = V[j, ],
      grads_X = grads,
      g_block = g_block * scale_fac
    )
    grad_v[j, ] <- fssd_grads$grad_vj
    grad_param <- grad_param + fssd_grads$grad_param
  }

  grad_par <- as.numeric(t(grad_v))
  if (has_kernel_param) grad_par <- c(grad_par, grad_param)

  list(value = fssd2 / denom, grad = grad_par)
}


# ---- Helpers -------------------------------------------------------------

#' @keywords internal
prepare_fssd_inputs <- function(X, score_function, min_rows,
                                kernel = c("gaussian_rbf", "imq"),
                                scaling = NULL,
                                median_max_samples = 2000,
                                median_use_sampling = TRUE,
                                median_seed = 123,
                                imq_beta = -0.5) {
  x_mat <- validate_samples(X, min_rows = min_rows)
  grads <- validate_scores(score_function, x_mat)
  kernel_prep <- prepare_fssd_kernel(
    X = x_mat,
    kernel = kernel,
    scaling = scaling,
    median_max_samples = median_max_samples,
    median_use_sampling = median_use_sampling,
    median_seed = median_seed,
    imq_beta = imq_beta
  )

  c(list(X = x_mat, grads = grads), kernel_prep)
}


#' @keywords internal
prepare_fssd_kernel <- function(X,
                                kernel = c("gaussian_rbf", "imq"),
                                scaling = NULL,
                                median_max_samples = 2000,
                                median_use_sampling = TRUE,
                                median_seed = 123,
                                imq_beta = -0.5) {
  x_mat <- validate_samples(X, min_rows = 2)

  if (inherits(kernel, "SteinKernel")) {
    kernel_obj <- kernel
    kernel_name <- kernel_type_name(kernel_obj)
    kernel_scale <- kernel_scaling_value(kernel_obj)

    if (inherits(kernel_obj, "SteinKernel_gaussian_rbf") && !is.finite(kernel_scale)) {
      if (is.null(scaling)) {
        scaling <- find_median_distance(x_mat, median_max_samples, median_use_sampling, median_seed)
      }
      kernel_obj <- instantiate_kernel("gaussian_rbf", scaling, imq_beta)
    } else if (inherits(kernel_obj, "SteinKernel_imq") && !is.finite(kernel_scale)) {
      if (is.null(scaling)) {
        scaling <- find_median_distance(x_mat, median_max_samples, median_use_sampling, median_seed)
      }
      kernel_obj <- instantiate_kernel("imq", max(scaling, 1e-8), imq_beta)
    } else {
      scaling <- kernel_scale
    }
  } else {
    kernel_name <- match.arg(kernel, c("gaussian_rbf", "imq"))
    if (is.null(scaling)) {
      scaling <- find_median_distance(x_mat, median_max_samples, median_use_sampling, median_seed)
    }
    if (kernel_name == "imq") scaling <- max(scaling, 1e-8)
    kernel_obj <- instantiate_kernel(kernel_name, scaling, imq_beta)
  }

  list(kernel_obj = kernel_obj, kernel_name = kernel_name, scaling = scaling)
}


#' Bandwidth grid: median squared distance times `2^seq(-3, 3)`.
#' @keywords internal
fssd_scale_grid <- function(X, max_samples = 1000, use_sampling = TRUE, seed = NULL) {
  med2 <- find_median_distance(
    X,
    max_samples = max_samples,
    use_sampling = use_sampling,
    seed = seed
  )
  med2 * 2^seq(-3, 3, length.out = 5)
}


#' Grid-search the squared kernel scale with locations fixed.
#' @keywords internal
fssd_grid_search_scale <- function(X, grads, V, kernel_obj_base, scale_grid, gamma) {
  scale_grid <- as.numeric(scale_grid)
  scale_grid <- scale_grid[is.finite(scale_grid) & scale_grid > 0]
  if (length(scale_grid) < 1) stop("scale_grid must contain at least one positive finite value")

  J <- nrow(V)
  par_v <- as.numeric(t(V))
  objectives <- vapply(scale_grid, function(scale2) {
    fssd_objective_and_grad(
      par = c(par_v, scale2),
      X = X,
      grads = grads,
      J = J,
      kernel_obj_base = kernel_obj_base,
      has_kernel_param = TRUE,
      gamma = gamma
    )$value
  }, numeric(1))

  finite_objectives <- objectives
  finite_objectives[!is.finite(finite_objectives)] <- -Inf
  if (all(is.infinite(finite_objectives))) {
    return(list(scale = scale_grid[1L], objectives = objectives, best_index = 1L))
  }

  best_index <- which.max(finite_objectives)
  list(scale = scale_grid[best_index], objectives = objectives, best_index = best_index)
}


#' V ~ N(mu_hat, Sigma_hat + cov_reg * I) fitted to X.
#' @keywords internal
fssd_init_locations_from_fit <- function(X, J, cov_reg = 0) {
  X <- as.matrix(X)
  cov_reg <- if (is.null(cov_reg)) 0 else as.numeric(cov_reg)
  if (length(cov_reg) != 1 || !is.finite(cov_reg) || cov_reg < 0) {
    stop("cov_reg must be a non-negative scalar")
  }
  sigma <- safe_covariance(X)
  if (cov_reg > 0) diag(sigma) <- diag(sigma) + cov_reg
  sample_gaussian_rows(J, colMeans(X), sigma)
}


#' Re-instantiate a kernel with a new squared scale.
#' @keywords internal
fssd_kernel_with_scale <- function(base_kernel, scale2_k, has_kernel_param) {
  if (!has_kernel_param) return(base_kernel)
  scale2_k <- validate_fssd_positive_scalar(scale2_k, "scale2_k")

  if (inherits(base_kernel, "SteinKernel_gaussian_rbf")) {
    return(stein_kernel(type = "gaussian_rbf", h = sqrt(scale2_k)))
  }
  if (inherits(base_kernel, "SteinKernel_imq")) {
    out <- base_kernel
    out$c <- sqrt(scale2_k)
    return(out)
  }

  stop("Kernel parameter optimization is not implemented for this kernel class")
}


#' @keywords internal
fssd_optimization_bounds <- function(X, J, has_kernel_param, locs_bounds_frac,
                                     scale_lower, scale_upper) {
  x_mat <- as.matrix(X)
  x_std <- apply(x_mat, 2, stats::sd)
  x_std[!is.finite(x_std) | x_std <= 0] <- 1e-8

  v_lb <- rep(apply(x_mat, 2, min) - locs_bounds_frac * x_std, times = J)
  v_ub <- rep(apply(x_mat, 2, max) + locs_bounds_frac * x_std, times = J)

  if (!has_kernel_param) {
    return(list(lower = v_lb, upper = v_ub))
  }

  list(
    lower = c(v_lb, scale_lower),
    upper = c(v_ub, scale_upper)
  )
}


#' @keywords internal
fssd_project_parameters <- function(par, lower, upper) {
  pmin(pmax(par, lower), upper)
}


#' Paper plug-in covariance for FSSD null simulation.
#' @keywords internal
fssd_tau_covariance <- function(tau) {
  x_mat <- as.matrix(tau)
  if (!is.numeric(x_mat) || nrow(x_mat) < 1 || ncol(x_mat) < 1) {
    stop("tau must be a non-empty numeric matrix")
  }
  centered <- sweep(x_mat, 2, colMeans(x_mat), "-")
  cov_hat <- crossprod(centered) / nrow(x_mat)
  cov_hat[!is.finite(cov_hat)] <- 0
  (cov_hat + t(cov_hat)) / 2
}


#' @keywords internal
validate_fssd_positive_scalar <- function(x, arg_name) {
  if (!is.numeric(x) || length(x) != 1 || !is.finite(x) || x <= 0) {
    stop(sprintf("%s must be a positive scalar", arg_name))
  }
  as.numeric(x)
}


#' @keywords internal
validate_fssd_train_ratio <- function(train_ratio) {
  if (!is.numeric(train_ratio) || length(train_ratio) != 1 ||
      !is.finite(train_ratio) || train_ratio <= 0 || train_ratio >= 1) {
    stop("train_ratio must be a scalar in (0, 1)")
  }
  as.numeric(train_ratio)
}


#' @keywords internal
safe_covariance <- function(x) {
  x_mat <- as.matrix(x)
  if (ncol(x_mat) == 1) {
    v <- stats::var(x_mat[, 1])
    if (!is.finite(v) || v < 0) v <- 0
    return(matrix(v, 1, 1))
  }

  s <- stats::cov(x_mat)
  s[!is.finite(s)] <- 0
  (s + t(s)) / 2
}


#' Draw rows from a Gaussian with a stabilized covariance root.
#' @keywords internal
sample_gaussian_rows <- function(n, mu, sigma) {
  d <- length(mu)
  sigma <- as.matrix(sigma)
  if (!all(dim(sigma) == c(d, d))) stop("sigma must be d x d")

  sigma <- (sigma + t(sigma)) / 2
  eig <- eigen(sigma, symmetric = TRUE)
  vals <- pmax(eig$values, 1e-10)
  root <- eig$vectors %*% diag(sqrt(vals), nrow = d)
  z <- matrix(stats::rnorm(n * d), nrow = n, ncol = d)
  sweep(z %*% t(root), 2, mu, "+")
}
