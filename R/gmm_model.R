# gmm_model.R

#' Create a Gaussian mixture model
#'
#' Builds a Gaussian mixture model object for simulations and examples.
#'
#' @details
#' A Gaussian mixture draws each observation in two steps. First it chooses a
#' component `k` with probability `weights[k]`. Then it draws the observation
#' from the Gaussian distribution with mean `mu[, k]` and covariance
#' `sigma[, , k]`. Its density is
#' \deqn{p(x) = \sum_{k=1}^K w_k \phi(x; \mu_k, \Sigma_k),}
#' where `w_k` is the component weight and `phi` is the Gaussian density.
#'
#' If `nComp` is `NULL`, the function creates a five-component mixture in
#' dimension `d`, with `d = 1` by default. If means or covariances are omitted,
#' random means and identity covariance matrices are filled in.
#'
#' The returned object is deliberately a simple list because the package uses it
#' as an example target distribution rather than as a fitted statistical model.
#' The fields are exactly the quantities needed by [rgmm()],
#' [likelihoodgmm()], [posteriorgmm()], [scorefunctiongmm()], and
#' [get_score_evaluator()]: weights for component probabilities, means and
#' covariances for Gaussian densities, and dimension `d` for input checking.
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
#' A list with components:
#' `nComp`, the number of mixture components;
#' `mu`, a `d x nComp` matrix whose column `k` is component mean `mu_k`;
#' `sigma`, a `d x d x nComp` array whose slice `sigma[, , k]` is covariance
#' matrix `Sigma_k`;
#' `weights`, normalized component probabilities; and
#' `d`, the dimension of each observation. The list is the common input object
#' for the other Gaussian mixture helpers.
#'
#'
#' @export
#'
#' @examples
#' # Default one-dimensional Gaussian mixture model
#' model <- gmm()
#'
#' # One-dimensional Gaussian mixture model with three components
#' model <- gmm(nComp = 3)
#'
#' # Three-dimensional mixture with specified means, covariances, and weights
#' mu <- matrix(c(1, 2, 3, 2, 3, 4, 5, 6, 7), ncol = 3)
#' sigma <- array(diag(3), c(3, 3, 3))
#' model <- gmm(nComp = 3, mu = mu, sigma = sigma, weights = c(0.2, 0.4, 0.4), d = 3)
gmm <- function(nComp = NULL, mu = NULL, sigma = NULL, weights = NULL, d = NULL) {
  # Do not reset the RNG here; callers control reproducibility with set.seed().
  k <- if (is.null(nComp)) 5L else validate_positive_integer(nComp, "nComp")

  if (!is.null(d)) {
    d <- validate_positive_integer(d, "d")
  } else if (!is.null(mu) && !is.null(dim(mu))) {
    if (length(dim(mu)) != 2L) stop("mu must be a vector or matrix")
    d <- nrow(mu)
  } else if (!is.null(sigma) && !is.null(dim(sigma))) {
    sigma_dim <- dim(sigma)
    if (!length(sigma_dim) %in% c(2L, 3L)) {
      stop("sigma must be a vector, matrix, or three-dimensional array")
    }
    d <- sigma_dim[1L]
  } else {
    d <- 1L
  }
  d <- validate_positive_integer(d, "d")

  if (is.null(mu)) {
    mu <- matrix(stats::runif(d * k, 0, 10), nrow = d, ncol = k)
    mu <- sweep(mu, 1L, rowMeans(mu), "-")
  } else {
    if (!is.numeric(mu) || any(!is.finite(mu))) {
      stop("mu must contain only finite numeric values")
    }
    if (is.null(dim(mu))) {
      if (d == 1L && length(mu) == k) {
        mu <- matrix(mu, nrow = 1L)
      } else if (k == 1L && length(mu) == d) {
        mu <- matrix(mu, ncol = 1L)
      } else {
        stop("vector mu must contain one mean per component in one dimension")
      }
    } else {
      mu <- as.matrix(mu)
    }
    if (!identical(dim(mu), c(d, k))) {
      stop("mu must have dimensions d x nComp")
    }
  }

  if (is.null(sigma)) {
    sigma <- array(diag(d), dim = c(d, d, k))
  } else {
    if (!is.numeric(sigma) || any(!is.finite(sigma))) {
      stop("sigma must contain only finite numeric values")
    }
    sigma_dim <- dim(sigma)
    if (is.null(sigma_dim)) {
      if (d != 1L || !length(sigma) %in% c(1L, k)) {
        stop("one-dimensional sigma must contain one variance or nComp variances")
      }
      sigma <- array(rep(as.numeric(sigma), length.out = k), dim = c(1L, 1L, k))
    } else if (length(sigma_dim) == 2L) {
      sigma <- as.matrix(sigma)
      if (!identical(dim(sigma), c(d, d))) {
        stop("matrix sigma must have dimensions d x d")
      }
      sigma <- array(rep(as.numeric(sigma), k), dim = c(d, d, k))
    } else if (length(sigma_dim) != 3L ||
      !identical(as.integer(sigma_dim), c(d, d, k))) {
      stop("sigma must have dimensions d x d x nComp")
    }
  }

  for (component_idx in seq_len(k)) {
    sigma_k <- matrix(sigma[, , component_idx], nrow = d, ncol = d)
    sym_tol <- sqrt(.Machine$double.eps) * max(1, max(abs(sigma_k)))
    if (max(abs(sigma_k - t(sigma_k))) > sym_tol) {
      stop("each component covariance must be symmetric positive definite")
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
      stop("weights must be finite, nonnegative, and contain one value per component with a positive sum")
    }
    weights <- as.numeric(weights) / sum(weights)
  }

  list(nComp = k, mu = mu, sigma = sigma, weights = weights, d = d)
}

#' Sample from a Gaussian mixture model
#'
#' Draws independent observations from the mixture object returned by [gmm()].
#' For each observation, the function samples a component label using the model
#' weights and then draws from that component's Gaussian distribution.
#'
#' This is the simulation counterpart of [likelihoodgmm()] and
#' [scorefunctiongmm()]. It is included so examples can generate data from the
#' same mixture object used as a target model for Stein tests and samplers.
#'
#' The returned observations are independent draws from the mixture. The
#' sampled component labels are used only internally; they are not returned
#' because the public Stein routines work with observations and scores, not
#' latent mixture labels. If labels are needed for a simulation study, reproduce
#' the two-step sampling logic with the same `model$weights`, `model$mu`, and
#' `model$sigma` fields.
#'
#' @param model Gaussian mixture model returned by [gmm()].
#' @param n Number of samples to draw.
#'
#' @return
#' For `model$d == 1`, a numeric vector of length `n`. For `model$d > 1`, an
#' `n x d` numeric matrix with one simulated observation per row. The different
#' shape for one-dimensional mixtures keeps simple examples readable, while the
#' package's validation helpers convert vectors back to `n x 1` matrices when
#' needed.
#'
#' @note Multivariate sampling uses the package dependency `mvtnorm`.
#' @export
#'
#' @examples
#' # Generate 100 samples from default gaussian mixture model
#' model <- gmm()
#' X <- rgmm(model)
#'
#' # Generate 300 samples from 3-d gaussian mixture model
#' model <- gmm(d = 3)
#' X <- rgmm(model, n = 300)
rgmm <- function(model = NULL, n = 100) {
  if (is.null(model)) {
    stop('Supply GMM Model')
  } else {
    n <- validate_positive_integer(n, "n")
    k <- model$nComp
    mu <- model$mu
    sigma <- model$sigma
    weights <- model$weights
    d <- model$d

    components <- sample.int(k, prob = weights, size = n, replace = TRUE)

    # 1 Dimensional case
    if (d == 1) {
      stdev <- rep(0, k)
      for (i in 1:k) {
        stdev[i] <- sqrt(sigma[, , i])
      }
      data <- rnorm(n = n, mean = mu[components], sd = stdev[components])
    } else {
      # Multidimensional Case
      if (!requireNamespace("mvtnorm", quietly = TRUE)) {
        stop("mvtnorm is required for multivariate Gaussian mixture sampling.",
             call. = FALSE)
      }
      # Place each row at the slot corresponding to its component label so the row
      # order follows the random `components` vector and prefixes remain iid.
      data <- matrix(0, nrow = n, ncol = d)
      for (i in seq_len(k)) {
        idx <- which(components == i)
        if (length(idx) > 0L) {
          data[idx, ] <- mvtnorm::rmvnorm(length(idx),
                                          mean = mu[, i],
                                          sigma = sigma[, , i])
        }
      }
    }

    return(data)
    # hcum <- h <- hist(data,breaks=30,plot=FALSE)
    # hcum$counts <- cumsum(hcum$counts)
  }
}


#' Perturb the component means of a Gaussian mixture model
#'
#' Adds independent standard normal noise to each component mean while keeping
#' the covariances and weights unchanged.
#'
#' @details
#' This helper is useful for examples where a fitted or proposed mixture should
#' be close to an existing one but not exactly equal to it. If the original
#' means are `mu_k`, the new means are `mu_k + z_k`, with each `z_k` drawn from
#' a standard normal distribution in the same dimension.
#'
#' It changes only the means because that gives a simple alternative target for
#' goodness-of-fit examples: the modes move, but the number of components,
#' component weights, and covariance shapes stay comparable.
#'
#' The function is meant to create a nearby but different model, not to perform
#' statistical fitting. For example, one can draw data from `model`, build
#' `noisymodel <- perturbgmm(model)`, and then test whether the sample looks
#' consistent with the perturbed score. Keeping weights and covariances fixed
#' makes the difference easy to interpret: failures are driven by shifted
#' component locations rather than by a completely different mixture shape.
#' The perturbation scale is intentionally simple: each coordinate of each
#' component mean receives one standard-normal draw.
#'
#' @param model Gaussian mixture model returned by [gmm()].
#'
#' @return
#' A Gaussian mixture list with the same structure as [gmm()]. The `mu` field is
#' perturbed; `sigma`, `weights`, `nComp`, and `d` are copied from the input
#' model. Returning the full model object, rather than only the perturbed means,
#' lets the result be passed directly to [rgmm()], [likelihoodgmm()],
#' [posteriorgmm()], [scorefunctiongmm()], or [get_score_evaluator()].
#'
#'
#' @export
#'
#' @examples
#' # Add noise to default 1-d gaussian mixture model
#' model <- gmm()
#' noisymodel <- perturbgmm(model)
perturbgmm <- function(model = NULL) {
  if (is.null(model)) {
    stop('Supply GMM Model')
  }
  # As in gmm(), the caller controls the random seed.
  k <- model$nComp
  d <- model$d
  noise <- replicate(k, rnorm(d))
  perturbed_mu <- model$mu + noise
  perturbed_model <- gmm(k, perturbed_mu, model$sigma, model$weights, d)

  return(perturbed_model)
}

#' Compute row-wise log-sum-exp values
#'
#' Computes `log(sum(exp(x)))` for each row of a matrix in a numerically stable
#' way. The largest value in each row is subtracted before exponentiating, then
#' added back at the end:
#' \deqn{\log \sum_j \exp(a_j) = m + \log \sum_j \exp(a_j - m),}
#' where `m = max_j a_j`.
#'
#' @param log_mat Numeric matrix of log values.
#'
#' @return Numeric vector, one value per row of `log_mat`.
#' @examples
#' row_logsumexp(matrix(c(0, 1, 2, 3), nrow = 2))
#' @noRd
row_logsumexp <- function(log_mat) {
  log_mat <- as.matrix(log_mat)
  if (!is.numeric(log_mat) || nrow(log_mat) < 1L || ncol(log_mat) < 1L || anyNA(log_mat)) {
    stop("log_mat must be a non-empty numeric matrix without missing values")
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

#' Extract a component mean from a Gaussian mixture model
#'
#' Returns the mean vector for one mixture component. One-dimensional mixtures
#' may store means as either a vector or a one-row matrix; this function hides
#' that storage detail and always returns a numeric vector.
#'
#' @param model Gaussian mixture model returned by [gmm()].
#' @param component_idx Component index.
#'
#' @return Numeric mean vector for the requested component.
#' @noRd
get_component_mean <- function(model, component_idx) {
  if (model$d == 1) {
    if (is.null(dim(model$mu))) {
      return(as.numeric(model$mu[component_idx]))
    }
    return(as.numeric(model$mu[1, component_idx]))
  }
  as.numeric(model$mu[, component_idx])
}

#' Invert a covariance matrix
#'
#' Symmetrizes a covariance matrix and returns its exact Cholesky-based inverse.
#' Invalid or non-positive-definite covariance matrices are rejected so the
#' returned score remains the gradient of the density represented by the model.
#'
#' @param sigma Numeric covariance matrix, or a one-dimensional variance.
#'
#' @return A numeric precision matrix, meaning the inverse covariance matrix.
#' @noRd
safe_precision_matrix <- function(sigma) {
  if (is.null(dim(sigma))) {
    val <- as.numeric(sigma)
    if (length(val) != 1L || !is.finite(val) || val <= 0) {
      stop("variance must be a positive finite scalar")
    }
    return(matrix(1 / val, nrow = 1, ncol = 1))
  }

  sigma <- as.matrix(sigma)
  d <- nrow(sigma)
  if (ncol(sigma) != d) {
    stop("Covariance matrix must be square")
  }

  sigma <- (sigma + t(sigma)) / 2
  chol_sigma <- tryCatch(chol(sigma), error = function(e) NULL)
  if (is.null(chol_sigma)) stop("covariance matrix must be positive definite")
  chol2inv(chol_sigma)
}

#' Build inverse covariance matrices for mixture components
#'
#' Computes one exact inverse covariance matrix per component.
#'
#' @details
#' These inverse covariance matrices are used in the Gaussian score formula.
#' For a single Gaussian component with mean `mu_k` and covariance `Sigma_k`,
#' the score contribution is
#' \deqn{- (x - \mu_k)^T \Sigma_k^{-1}.}
#'
#' @param model Gaussian mixture model returned by [gmm()].
#' @return A list of inverse covariance matrices, one per mixture component.
#' @noRd
build_precision_cache <- function(model) {
  lapply(seq_len(model$nComp), function(component_idx) {
    safe_precision_matrix(model$sigma[, , component_idx])
  })
}

#' Compute component log densities for a Gaussian mixture
#'
#' Evaluates the log Gaussian density of every sample under every component.
#' The weights are not included here; callers add `log(weights)` when they need
#' mixture probabilities.
#'
#' @param model Gaussian mixture model returned by [gmm()].
#' @param X Numeric vector or matrix of samples.
#'
#' @return Numeric matrix whose `(i, k)` entry is
#'   `log phi(X[i, ], mu[, k], sigma[, , k])`.
#' @noRd
as_gmm_sample_matrix <- function(model, X) {
  if (!is.numeric(X)) stop("X must contain only finite numeric values")
  if (is.null(dim(X))) {
    X <- if (model$d == 1L) matrix(X, ncol = 1L) else matrix(X, nrow = 1L)
  } else {
    X <- as.matrix(X)
  }
  if (nrow(X) < 1L || ncol(X) != model$d || any(!is.finite(X))) {
    stop("X must be a finite numeric matrix with model$d columns")
  }
  X
}


gmm_log_component_densities <- function(model, X) {
  d <- model$d
  k <- model$nComp
  mu <- model$mu
  sigma <- model$sigma

  x_mat <- as_gmm_sample_matrix(model, X)
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

#' Compute posterior component probabilities for a Gaussian mixture
#'
#' For each observation, this function computes the probability that the
#' observation came from each mixture component.
#'
#' @details
#' The posterior probability for component `k` at observation `x_i` is
#' \deqn{
#' P(Z_i = k | x_i) =
#' \frac{w_k \phi(x_i; \mu_k, \Sigma_k)}
#'      {\sum_l w_l \phi(x_i; \mu_l, \Sigma_l)}.
#' }
#' The calculation is done on the log scale first, which avoids numerical
#' underflow when densities are very small. If every component log density is
#' `-Inf` for a row, the function stops instead of returning arbitrary
#' responsibilities.
#'
#' These posterior probabilities are not used to classify observations in the
#' package; they are an intermediate quantity for [scorefunctiongmm()]. The
#' mixture score is a posterior-weighted average of component Gaussian scores.
#' Returning the full responsibility matrix, instead of only the most likely
#' component label, preserves the uncertainty needed for that weighted average.
#'
#' @param model Gaussian mixture model created by `gmm()`.
#' @param X Numeric vector or matrix of samples.
#'
#' @return
#' Numeric matrix with `nrow(X)` rows and `model$nComp` columns. Entry `(i, k)`
#' is the posterior probability that sample row `i` came from component `k`;
#' each row sums to one up to numerical rounding.
#' The column order matches the component order in `model$mu`, `model$sigma`,
#' and `model$weights`, so the matrix can be multiplied component-by-component
#' with Gaussian score contributions.
#' @examples
#' model <- gmm()
#' X <- rgmm(model, n = 5)
#' posteriorgmm(model, X)
#' @export
posteriorgmm <- function(model = NULL, X = NULL) {
  if (is.null(model) || is.null(X)) {
    stop('Supply Model and Data')
  }

  weights <- model$weights
  log_comp <- gmm_log_component_densities(model, X)
  log_w <- log(as.numeric(weights))
  log_joint <- sweep(log_comp, 2, log_w, "+")

  row_max <- apply(log_joint, 1, max, na.rm = TRUE)
  is_inf_row <- is.infinite(row_max) & row_max < 0
  if (any(is_inf_row)) {
    stop("all component log densities are -Inf for at least one sample row")
  }
  post <- matrix(0, nrow = nrow(log_joint), ncol = ncol(log_joint))
  valid <- !is_inf_row

  if (any(valid)) {
    shifted <- log_joint[valid, , drop = FALSE] - row_max[valid]
    post[valid, ] <- exp(shifted) / rowSums(exp(shifted))
  }
  return(post)
}

#' Compute the mixture density for samples
#'
#' Evaluates the Gaussian mixture density at each supplied sample.
#'
#' @details
#' For observation `x_i`, the returned value is
#' \deqn{p(x_i) = \sum_{k=1}^K w_k \phi(x_i; \mu_k, \Sigma_k).}
#' The component densities are combined on the log scale and then converted
#' back to ordinary density values.
#'
#' This function returns the density itself, not a log density. Use it for plots
#' or simple likelihood checks. For MCMC transitions such as [mala()] and
#' [grwmetrop()], pass a separate `log_p` function because those transitions
#' work on the log-density scale.
#'
#' The function sits beside [posteriorgmm()] and [scorefunctiongmm()]. All three
#' evaluate the same mixture at supplied sample rows, but they answer different
#' questions: `likelihoodgmm()` returns the marginal density `p(x)`,
#' [posteriorgmm()] returns component responsibilities conditional on `x`, and
#' [scorefunctiongmm()] returns the gradient of `log p(x)` needed by Stein
#' algorithms.
#'
#' @param model Gaussian mixture model returned by [gmm()].
#' @param X Numeric vector or matrix of samples.
#'
#' @return
#' Numeric vector of length `nrow(X)` containing `p(x_i)` for each sample row.
#' Values are nonnegative density values and do not sum to one over the sample.
#' They are pointwise density evaluations, so their scale depends on the
#' dimension and covariance matrices of the mixture.
#'
#'
#' @export
#'
#' @examples
#' # compute likelihood for a default 1-d gaussian mixture model
#' # and dataset generated from it
#' model <- gmm()
#' X <- rgmm(model)
#' p <- likelihoodgmm(model = model, X = X)
likelihoodgmm <- function(model = NULL, X = NULL) {
  if (is.null(model) || is.null(X)) {
    stop('Supply Model and Data')
  } else {
    weights <- model$weights

    log_comp <- gmm_log_component_densities(model, X)
    log_w <- log(as.numeric(weights))
    log_joint <- sweep(log_comp, 2, log_w, "+")
    sum_prob <- exp(row_logsumexp(log_joint))
  }

  return(sum_prob)
}


#' Compute the score of a Gaussian mixture density
#'
#' The score is the gradient of the log mixture density with respect to `x`.
#'
#' @details
#' For a mixture, the score is a weighted average of the component scores:
#' \deqn{
#' \nabla_x \log p(x_i) =
#' \sum_{k=1}^K P(Z_i = k | x_i)
#' \left[-\Sigma_k^{-1}(x_i - \mu_k)\right].
#' }
#' The weights in this average are the posterior component probabilities
#' returned by [posteriorgmm()].
#'
#' This is the Gaussian mixture helper most directly connected to the Stein
#' routines. [ksd_u_test()], [fssd_test()], [stein_points()], and related
#' functions need `function(X)` returning the score matrix; this function
#' computes that matrix for one supplied batch.
#'
#' If the target were a single Gaussian, the score would be one linear term
#' `-Sigma^{-1}(x - mu)`. For a mixture, the function computes that linear
#' score for every component and averages the component scores using the
#' posterior probabilities from [posteriorgmm()]. This is why both the density
#' calculation and the inverse covariance matrices appear in the implementation.
#'
#' @param model Gaussian mixture model returned by [gmm()].
#' @param X Numeric vector or matrix of samples.
#'
#' @return
#' Numeric matrix with `nrow(X)` rows and `model$d` columns. Row `i` is
#' `nabla_x log p(x_i)`, the score vector used by Stein discrepancies and
#' score-based samplers. The matrix has the same row order as the supplied
#' sample matrix, which is important because KSD, FSSD, Stein thinning, and
#' Stein Points pair each sample row with its corresponding score row.
#'
#'
#' @export
#'
#' @examples
#' # Compute the score for a Gaussian mixture model and dataset
#' model <- gmm()
#' X <- rgmm(model)
#' score <- scorefunctiongmm(model = model, X = X)
scorefunctiongmm <- function(model = NULL, X = NULL) {
  if (is.null(model) || is.null(X)) {
    stop('Supply Model and Data')
  } else {
    x_mat <- as_gmm_sample_matrix(model, X)
    P <- posteriorgmm(model, x_mat)
    d <- model$d
    precision_cache <- build_precision_cache(model)
    n <- nrow(x_mat)

    score <- matrix(0, nrow = n, ncol = d)

    for (component_idx in 1:model$nComp) {
      mean_k <- get_component_mean(model, component_idx)
      if (d == 1) {
        diff <- x_mat[, 1] - mean_k
        comp_score <- -matrix(diff, ncol = 1) %*% precision_cache[[component_idx]]
      } else {
        diff <- sweep(x_mat, 2, mean_k, "-")
        comp_score <- -diff %*% precision_cache[[component_idx]]
      }
      score <- score + comp_score * P[, component_idx]

    }
  }

  return(score)
}


#' Plot a one-dimensional Gaussian mixture sample
#'
#' Draws a histogram with a kernel density overlay. Optional component means
#' are shown as vertical reference lines.
#'
#' @details
#' The function is intended as a quick visual check for one-dimensional
#' simulation examples. The histogram shows the empirical sample distribution,
#' the smooth line shows a kernel density estimate, and the optional vertical
#' lines mark supplied component means.
#'
#' This plotting helper is deliberately separate from the mixture-density
#' functions. It visualizes sampled data; it does not evaluate the exact mixture
#' density and it returns no numerical object for downstream Stein calculations.
#'
#' @param data Numeric vector of one-dimensional samples.
#' @param mu Optional component means to mark on the plot.
#'
#' @return
#' The function is called for its plotting side effect and returns `NULL`
#' invisibly. It does not return density values, posterior probabilities, or
#' scores; use [likelihoodgmm()], [posteriorgmm()], or [scorefunctiongmm()] for
#' those numerical quantities.
#'
#' @export
#'
#' @examples
#' # Plot a histogram for a given dataset
#' model <- gmm()
#' X <- rgmm(model)
#' plotgmm(data = X)
#'
#' # Plot the histogram with lines indicating the component means
#' model <- gmm()
#' mu <- model$mu
#' X <- rgmm(model)
#' plotgmm(data = X, mu = mu)
plotgmm <- function(data, mu = NULL) {
  hcum <- h <- hist(data, breaks = 30, plot = FALSE)
  hcum$counts <- cumsum(hcum$counts)

  ## ----- Plot only pdf----------------
  old_par <- par(mar = c(1, 1, 1, 1))
  on.exit(par(old_par), add = TRUE)
  plot(h, xlim = c(-20, 20), col = "grey")
  d <- density(data)
  lines(x = d$x, y = d$y * length(data) * diff(h$breaks)[1], lwd = 2)
  if (!is.null(mu)) {
    abline(v = mu, col = "royalblue", lwd = 2)
  }
}


#' Create a score function for a fixed Gaussian mixture model
#'
#' Returns a function of `X` only, ready to pass to routines that need a score
#' function.
#'
#' @details
#' Many Stein routines ask for a score function with signature `function(X)`.
#' This helper stores the model and its inverse covariance matrices once, then
#' returns a function that evaluates the Gaussian mixture score for new samples.
#'
#' This is the preferred bridge from the Gaussian mixture helpers to the main
#' Stein algorithms. Instead of repeatedly passing both `model` and `X` to
#' [scorefunctiongmm()], create `score <- get_score_evaluator(model)` once and
#' then pass `score` to [ksd_u_test()], [fssd_test()], [stein_points()],
#' [sp_mcmc()], or [stein_thinning()].
#'
#' The closure caches the inverse covariance matrices because they depend only
#' on the mixture model, not on the evaluation points. That design keeps the
#' returned score function cheap enough to call repeatedly inside iterative
#' algorithms such as SVGD, Stein Points, and SP-MCMC.
#'
#' @param model Gaussian mixture model returned by [gmm()].
#'
#' @return
#' A function with signature `function(X)`. For matrix input it returns an
#' `n x d` score matrix for new samples, using cached inverse covariance
#' matrices from `model`. For vector input it returns a numeric vector, matching
#' the package's convention that a vector is a batch of one-dimensional samples
#' when `model$d == 1` and one multivariate sample when `model$d > 1`. The main
#' Stein routines call score functions with matrix inputs, so they receive the
#' matrix form needed to align rows of samples and scores.
#'
#' @export
#'
#' @examples
#' model <- gmm()
#' grad_log_prob <- get_score_evaluator(model)
#' X <- rgmm(model)
#' G <- grad_log_prob(X)
get_score_evaluator <- function(model) {
  precision_cache <- build_precision_cache(model)

  function(X) {
    d <- model$d
    X_mat <- as_gmm_sample_matrix(model, X)

    post <- posteriorgmm(model = model, X = X_mat)
    n <- nrow(X_mat)
    grad <- matrix(0, nrow = n, ncol = d)

    for (component_idx in seq_len(model$nComp)) {
      mean_k <- get_component_mean(model, component_idx)
      if (d == 1) {
        diff <- X_mat[, 1] - mean_k
        comp_grad <- -matrix(diff, ncol = 1) %*% precision_cache[[component_idx]]
      } else {
        diff <- sweep(X_mat, 2, mean_k, "-")
        comp_grad <- -diff %*% precision_cache[[component_idx]]
      }
      grad <- grad + comp_grad * post[, component_idx]
    }

    if (is.null(dim(X))) {
      return(as.vector(grad))
    }

    grad
  }
}

# Snake_case aliases
gaussian_mixture_model <- gmm
sample_gmm <- rgmm
perturb_gmm <- perturbgmm
posterior_gmm <- posteriorgmm
likelihood_gmm <- likelihoodgmm
score_function_gmm <- scorefunctiongmm
plot_gmm <- plotgmm
