# Gaussian-mixture targets used by the package examples and tests.

# ---- Public API ----------------------------------------------------------

#' Create a Gaussian mixture model
#'
#' Constructs a Gaussian mixture distribution with `nComp` components.
#'
#' @details
#' The model density is
#' \deqn{p(x) = \sum_{k=1}^K w_k \phi(x; \mu_k, \Sigma_k),}
#' where \eqn{w_k}, \eqn{\mu_k}, and \eqn{\Sigma_k} are the weight, mean, and
#' covariance of component \eqn{k}. Weights are normalized to sum to one.
#'
#' By default, `nComp = 5` and `d = 1`. If `mu` is omitted, centered random
#' means are generated. If `sigma` is omitted, every component uses the
#' identity covariance matrix.
#'
#' @param nComp Number of mixture components. If `NULL`, five components are
#'   generated.
#' @param mu Component means, stored as a `d x nComp` matrix. A vector is also
#'   accepted for a one-dimensional mixture or a single component.
#' @param sigma Component covariance matrices, stored as a `d x d x nComp`
#'   array. Each matrix must be symmetric positive definite. A scalar or vector
#'   is treated as one-dimensional component variances; a `d x d` matrix is
#'   reused for every component.
#' @param weights Optional nonnegative mixture weights, with at least one
#'   positive value. Values are normalized to sum to one.
#' @param d Dimension of each sample. If omitted, it is inferred from `mu` or
#'   `sigma`, and otherwise defaults to one.
#'
#' @return
#' An object of class `"gmm"` containing `nComp`; the `d x nComp` mean matrix
#' `mu`; the `d x d x nComp` covariance array `sigma`; normalized `weights`;
#' and dimension `d`.
#'
#' @export
#'
#' @examples
#' model <- gmm()
#'
#' mu <- matrix(c(1, 2, 3, 2, 3, 4, 5, 6, 7), ncol = 3)
#' sigma <- array(diag(3), c(3, 3, 3))
#' model <- gmm(nComp = 3, mu = mu, sigma = sigma,
#'              weights = c(0.2, 0.4, 0.4), d = 3)
gmm <- function(nComp = NULL, mu = NULL, sigma = NULL, weights = NULL, d = NULL) {
  # Do not reset the RNG here; callers control reproducibility with set.seed().
  k <- if (is.null(nComp)) 5L else validate_integer(nComp, "nComp")

  if (!is.null(d)) {
    d <- validate_integer(d, "d")
  } else if (!is.null(mu) && !is.null(dim(mu))) {
    if (length(dim(mu)) != 2L) stop("mu must be a vector or matrix", call. = FALSE)
    d <- nrow(mu)
  } else if (!is.null(sigma) && !is.null(dim(sigma))) {
    sigma_dim <- dim(sigma)
    if (!length(sigma_dim) %in% c(2L, 3L)) {
      stop("sigma must be a vector, matrix, or three-dimensional array", call. = FALSE)
    }
    d <- sigma_dim[1L]
  } else {
    d <- 1L
  }
  d <- validate_integer(d, "d")

  if (is.null(mu)) {
    mu <- matrix(stats::runif(d * k, 0, 10), nrow = d, ncol = k)
    mu <- sweep(mu, 1L, rowMeans(mu), "-")
  } else {
    if (!is.numeric(mu) || any(!is.finite(mu))) {
      stop("mu must contain only finite numeric values", call. = FALSE)
    }
    if (is.null(dim(mu))) {
      if (d == 1L && length(mu) == k) {
        mu <- matrix(mu, nrow = 1L)
      } else if (k == 1L && length(mu) == d) {
        mu <- matrix(mu, ncol = 1L)
      } else {
        stop("vector mu must contain one mean per component in one dimension", call. = FALSE)
      }
    } else {
      mu <- as.matrix(mu)
    }
    if (!identical(dim(mu), c(d, k))) {
      stop("mu must have dimensions d x nComp", call. = FALSE)
    }
  }

  if (is.null(sigma)) {
    sigma <- array(diag(d), dim = c(d, d, k))
  } else {
    if (!is.numeric(sigma) || any(!is.finite(sigma))) {
      stop("sigma must contain only finite numeric values", call. = FALSE)
    }
    sigma_dim <- dim(sigma)
    if (is.null(sigma_dim)) {
      if (d != 1L || !length(sigma) %in% c(1L, k)) {
        stop("one-dimensional sigma must contain one variance or nComp variances", call. = FALSE)
      }
      sigma <- array(rep(as.numeric(sigma), length.out = k), dim = c(1L, 1L, k))
    } else if (length(sigma_dim) == 2L) {
      sigma <- as.matrix(sigma)
      if (!identical(dim(sigma), c(d, d))) {
        stop("matrix sigma must have dimensions d x d", call. = FALSE)
      }
      sigma <- array(rep(as.numeric(sigma), k), dim = c(d, d, k))
    } else if (length(sigma_dim) != 3L ||
      !identical(as.integer(sigma_dim), c(d, d, k))) {
      stop("sigma must have dimensions d x d x nComp", call. = FALSE)
    }
  }

  for (component_idx in seq_len(k)) {
    sigma_k <- matrix(sigma[, , component_idx], nrow = d, ncol = d)
    sym_tol <- sqrt(.Machine$double.eps) * max(1, max(abs(sigma_k)))
    if (max(abs(sigma_k - t(sigma_k))) > sym_tol) {
      stop("each component covariance must be symmetric positive definite", call. = FALSE)
    }
    sigma_k <- (sigma_k + t(sigma_k)) / 2
    tryCatch(chol(sigma_k), error = function(e) {
      stop("each component covariance must be symmetric positive definite", call. = FALSE)
    })
    sigma[, , component_idx] <- sigma_k
  }

  if (is.null(weights)) {
    weights <- rep(1 / k, k)
  } else {
    if (!is.numeric(weights) || length(weights) != k ||
      any(!is.finite(weights)) || any(weights < 0) || sum(weights) <= 0) {
      stop("weights must be finite, nonnegative, and contain one value per component with a positive sum", call. = FALSE)
    }
    weights <- as.numeric(weights) / sum(weights)
  }

  structure(
    list(nComp = k, mu = mu, sigma = sigma, weights = weights, d = d),
    class = "gmm"
  )
}

#' @rdname gmm
#' @param x A `"gmm"` object.
#' @param ... Ignored.
#' @export
print.gmm <- function(x, ...) {
  cat(sprintf("Gaussian mixture  K=%d  d=%d\n", x$nComp, x$d))
  cat("  weights:", format(x$weights, digits = 3), "\n")
  invisible(x)
}

#' Sample from a Gaussian mixture model
#'
#' Draws independent observations from a [gmm()] object. For each observation,
#' a component is sampled according to `model$weights`, followed by a Gaussian
#' draw with that component's mean and covariance.
#'
#' @param model Gaussian mixture model returned by [gmm()].
#' @param n Number of samples to draw.
#'
#' @return
#' For `model$d == 1`, a numeric vector of length `n`. For `model$d > 1`, an
#' `n x d` numeric matrix with one observation per row.
#' @export
#'
#' @examples
#' model <- gmm()
#' X <- rgmm(model)
rgmm <- function(model = NULL, n = 100) {
  if (is.null(model)) {
    stop("Supply GMM Model", call. = FALSE)
  }
  n <- validate_integer(n, "n")
  k <- model$nComp
  mu <- model$mu
  sigma <- model$sigma
  weights <- model$weights
  d <- model$d

  components <- sample.int(k, prob = weights, size = n, replace = TRUE)

  # One-dimensional mixtures can be sampled directly with `rnorm()`.
  if (d == 1) {
    stdev <- vapply(
      seq_len(k), function(i) sqrt(sigma[, , i]), numeric(1L)
    )
    data <- stats::rnorm(n = n, mean = mu[components], sd = stdev[components])
  } else {
    # Multivariate mixtures require one Gaussian draw per sampled component.
    if (!requireNamespace("mvtnorm", quietly = TRUE)) {
      stop(
        "mvtnorm is required for multivariate Gaussian mixture sampling.",
        call. = FALSE
      )
    }
    # Preserve the order of the sampled component labels.
    data <- matrix(0, nrow = n, ncol = d)
    for (i in seq_len(k)) {
      idx <- which(components == i)
      if (length(idx) > 0L) {
        data[idx, ] <- mvtnorm::rmvnorm(
          length(idx), mean = mu[, i], sigma = sigma[, , i]
        )
      }
    }
  }

  data
}

#' Evaluate a Gaussian mixture density
#'
#' Returns the density of a [gmm()] object at each supplied observation.
#'
#' @details
#' For each observation \eqn{x_i},
#' \deqn{p(x_i) = \sum_{k=1}^K w_k \phi(x_i; \mu_k, \Sigma_k).}
#' The function returns ordinary density values, not log densities.
#'
#' @param model Gaussian mixture model returned by [gmm()].
#' @param X Finite numeric vector or matrix. For a one-dimensional model, a
#'   vector represents multiple observations. For a multivariate model, a
#'   length-`d` vector represents one observation. Matrix rows are observations.
#'
#' @return
#' A numeric vector containing one ordinary density value per interpreted
#' observation in `X`.
#'
#' @export
#'
#' @examples
#' model <- gmm()
#' X <- rgmm(model)
#' p <- densitygmm(model = model, X = X)
densitygmm <- function(model = NULL, X = NULL) {
  if (is.null(model) || is.null(X)) {
    stop("supply `model` and `X`", call. = FALSE)
  }
  exp(.row_logsumexp(.gmm_log_joint_densities(model, X)))
}

#' Create a score function for a fixed Gaussian mixture model
#'
#' Returns a `function(X)` that evaluates the score of a [gmm()] object.
#'
#' @details
#' Define the component responsibility
#' \deqn{r_k(x)=
#' \frac{w_k\phi(x;\mu_k,\Sigma_k)}
#' {\sum_{\ell=1}^K w_\ell\phi(x;\mu_\ell,\Sigma_\ell)}.}
#' The returned function evaluates
#' \deqn{s_p(x)=\nabla_x\log p(x)
#' =\sum_{k=1}^K r_k(x)\Sigma_k^{-1}(\mu_k-x).}
#'
#' @param model Gaussian mixture model returned by [gmm()].
#'
#' @return
#' A function with signature `function(X)`. For an `n x d` matrix, it returns
#' the corresponding `n x d` score matrix. For vector input, it returns a
#' vector: multiple one-dimensional scores when `model$d == 1`, or one
#' length-`d` score when `model$d > 1`.
#'
#' @export
#'
#' @examples
#' model <- gmm()
#' grad_log_prob <- get_score_evaluator(model)
#' X <- rgmm(model)
#' G <- grad_log_prob(X)
get_score_evaluator <- function(model) {
  precision_cache <- .build_precision_cache(model)

  function(X) {
    score <- .gmm_score_core(model, X, precision_cache)

    if (is.null(dim(X))) {
      return(as.vector(score))
    }

    score
  }
}


# ---- Internal: shared coercion and numerics ------------------------------

# Stable row-wise log-sum-exp used by Gaussian-mixture density calculations.
.row_logsumexp <- function(log_mat) {
  log_mat <- as.matrix(log_mat)
  if (!is.numeric(log_mat) || nrow(log_mat) < 1L || ncol(log_mat) < 1L || anyNA(log_mat)) {
    stop("log_mat must be a non-empty numeric matrix without missing values", call. = FALSE)
  }
  row_max <- apply(log_mat, 1, max)
  out <- row_max
  finite <- is.finite(row_max)
  if (any(finite)) {
    shifted <- log_mat[finite, , drop = FALSE] - row_max[finite]
    out[finite] <- row_max[finite] + log(rowSums(exp(shifted)))
  }
  out
}

.get_component_mean <- function(model, component_idx) {
  if (model$d == 1) {
    if (is.null(dim(model$mu))) {
      return(as.numeric(model$mu[component_idx]))
    }
    return(as.numeric(model$mu[1, component_idx]))
  }
  as.numeric(model$mu[, component_idx])
}

.as_gmm_sample_matrix <- function(model, X) {
  if (!is.numeric(X)) stop("X must contain only finite numeric values", call. = FALSE)
  if (is.null(dim(X))) {
    X <- if (model$d == 1L) matrix(X, ncol = 1L) else matrix(X, nrow = 1L)
  } else {
    X <- as.matrix(X)
  }
  if (nrow(X) < 1L || ncol(X) != model$d || any(!is.finite(X))) {
    stop("X must be a finite numeric matrix with model$d columns", call. = FALSE)
  }
  X
}


# ---- Internal: mixture densities and responsibilities --------------------

.gmm_log_component_densities <- function(model, X) {
  d <- model$d
  k <- model$nComp
  mu <- model$mu
  sigma <- model$sigma

  x_mat <- .as_gmm_sample_matrix(model, X)
  n <- nrow(x_mat)

  log_comp <- matrix(NA_real_, nrow = n, ncol = k)

  if (d == 1) {
    x_vec <- as.numeric(x_mat[, 1])
    for (i in seq_len(k)) {
      stdev <- sqrt(sigma[, , i])
      log_comp[, i] <- stats::dnorm(x_vec, mean = mu[i], sd = stdev, log = TRUE)
    }
  } else {
    if (!requireNamespace("mvtnorm", quietly = TRUE)) {
      stop("mvtnorm is required for multivariate Gaussian mixture evaluation.",
           call. = FALSE)
    }
    for (i in seq_len(k)) {
      log_comp[, i] <- mvtnorm::dmvnorm(x_mat, mean = mu[, i], sigma = sigma[, , i], log = TRUE)
    }
  }

  log_comp
}

.gmm_log_joint_densities <- function(model, X) {
  sweep(
    .gmm_log_component_densities(model, X),
    2L,
    log(as.numeric(model$weights)),
    "+"
  )
}

.gmm_responsibilities <- function(model, X) {
  log_joint <- .gmm_log_joint_densities(model, X)
  row_max <- apply(log_joint, 1L, max, na.rm = TRUE)
  invalid <- is.infinite(row_max) & row_max < 0
  if (any(invalid)) {
    stop("all component log densities are -Inf for at least one sample row", call. = FALSE)
  }
  shifted <- log_joint - row_max
  exp(shifted) / rowSums(exp(shifted))
}


# ---- Internal: score and precision cache ---------------------------------

# Return a Cholesky-based precision matrix after positive-definiteness checks.
.safe_precision_matrix <- function(sigma) {
  if (is.null(dim(sigma))) {
    val <- as.numeric(sigma)
    if (length(val) != 1L || !is.finite(val) || val <= 0) {
      stop("variance must be a positive finite scalar", call. = FALSE)
    }
    return(matrix(1 / val, nrow = 1, ncol = 1))
  }

  sigma <- as.matrix(sigma)
  d <- nrow(sigma)
  if (ncol(sigma) != d) {
    stop("Covariance matrix must be square", call. = FALSE)
  }

  sigma <- (sigma + t(sigma)) / 2
  chol_sigma <- tryCatch(chol(sigma), error = function(e) NULL)
  if (is.null(chol_sigma)) stop("covariance matrix must be positive definite", call. = FALSE)
  chol2inv(chol_sigma)
}

.build_precision_cache <- function(model) {
  lapply(seq_len(model$nComp), function(component_idx) {
    .safe_precision_matrix(model$sigma[, , component_idx])
  })
}

.gmm_score_core <- function(model, X, precision_cache) {
  x_mat <- .as_gmm_sample_matrix(model, X)
  responsibilities <- .gmm_responsibilities(model, x_mat)
  score <- matrix(0, nrow = nrow(x_mat), ncol = model$d)

  for (component_idx in seq_len(model$nComp)) {
    mean_k <- .get_component_mean(model, component_idx)
    centered <- sweep(x_mat, 2L, mean_k, "-")
    component_score <- -centered %*% precision_cache[[component_idx]]
    score <- score + component_score * responsibilities[, component_idx]
  }

  score
}
