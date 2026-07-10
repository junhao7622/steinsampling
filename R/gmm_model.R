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
#' If `nComp` is `NULL`, the function creates a default one-dimensional
#' five-component mixture. If means or covariances are omitted, random means and
#' identity covariance matrices are filled in.
#'
#' The returned object is deliberately a simple list because the package uses it
#' as an example target distribution rather than as a fitted statistical model.
#' The fields are exactly the quantities needed by [rgmm()],
#' [likelihoodgmm()], [posteriorgmm()], [scorefunctiongmm()], and
#' [get_score_evaluator()]: weights for component probabilities, means and
#' covariances for Gaussian densities, and dimension `d` for input checking.
#'
#' @param nComp Number of mixture components. If `NULL`, a five-component
#'   one-dimensional mixture is generated.
#' @param mu Component means, stored as a `d x nComp` matrix.
#' @param sigma Component covariance matrices, stored as a `d x d x nComp`
#'   array. A vector is treated as one-dimensional variances.
#' @param weights Optional mixture weights. Values are normalized to sum to one.
#' @param d Dimension of each sample. Used when `mu` or `sigma` is generated.
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
#' # Default 1-d gaussian mixture model
#' model <- gmm()
#'
#' # 1-d Gaussian mixture model with 3 components
#' model <- gmm(nComp = 3)
#'
#' # 3-d Gaussian mixture model with 3 components, with specified mu,sigma and weights
#' mu <- matrix(c(1, 2, 3, 2, 3, 4, 5, 6, 7), ncol = 3)
#' sigma <- array(diag(3), c(3, 3, 3))
#' model <- gmm(nComp = 3, mu = mu, sigma = sigma, weights = c(0.2, 0.4, 0.4), d = 3)
gmm <- function(nComp = NULL, mu = NULL, sigma = NULL, weights = NULL, d = NULL) {
  # NOTE: set.seed(0) intentionally removed.
  # Hardcoding set.seed() inside a function that is called from parallel
  # workers (foreach / doRNG) resets every worker's RNG state on every call,
  # which corrupts doRNG's reproducibility and causes all parallel trials to
  # silently fail or return garbage.  Callers are responsible for seeding.
  #
  # Generate a default Gaussian Mixture Model
  if (is.null(nComp)) {
    d <- 1; k <- 5
    mu <- 10 * runif(k)
    mu <- mu - mean(mu)

    sigma <- array(diag(d), c(d, d, k));
  } else {
    k <- nComp
  }

  # Case when mean is not specified
  if (is.null(mu)) {
    if (is.null(d)) d <- 1

    mu <- 10 * replicate(k, runif(d))
    mu <- mu - mean(mu)

  }

  # Case when sigma is not specified
  if (is.null(sigma)) {
    if (is.null(d)) {
      d <- dim(mu)[1]
    }

    sigma <- array(diag(d), c(d, d, k));
  }

  # If sigma is one-dimensional
  if (is.null(dim(sigma))) {
    sigma <- array(sigma, c(1, 1, k))
  }

  # Handle the cases for the weights
  if (is.null(weights)) {
    weights = rep(1, k) / k
  } else if (any(weights < 0)) {
    stop('Non-positive weights')
  } else if (sum(weights) != 1) {
    weights <- weights / sum(weights)
  }

  model <- list("nComp" = k, "mu" = mu, "sigma" = sigma,
                "weights" = weights, "d" = d)

  return(model)
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
#' @note Multivariate sampling uses the suggested package `mvtnorm`.
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
    k <- model$nComp
    mu <- model$mu
    sigma <- model$sigma
    weights <- model$weights
    d <- model$d

    components <- sample(1:k, prob = weights, size = n, replace = TRUE)

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
        stop("mvtnorm needed for this demo to work. Please install it.",
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
  # NOTE: set.seed(0) intentionally removed — same reason as gmm().
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
  row_max <- apply(log_mat, 1, max)
  shifted <- log_mat - row_max
  row_max + log(rowSums(exp(shifted)))
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

#' Safely invert a covariance matrix
#'
#' Symmetrizes a covariance matrix and returns its inverse. If the matrix is
#' close to singular, a small value is added to the diagonal before inversion.
#'
#' @param sigma Numeric covariance matrix, or a one-dimensional variance.
#' @param ridge Positive diagonal value used when the matrix needs stabilizing.
#' @param cond_threshold Condition-number value above which the diagonal
#'   stabilizer is added before inversion.
#'
#' @return A numeric precision matrix, meaning the inverse covariance matrix.
#' @noRd
safe_precision_matrix <- function(sigma, ridge = 1e-6, cond_threshold = 1e10) {
  if (is.null(dim(sigma))) {
    val <- as.numeric(sigma)
    if (!is.finite(val) || val <= ridge) {
      val <- ridge
    }
    return(matrix(1 / val, nrow = 1, ncol = 1))
  }

  sigma <- as.matrix(sigma)
  d <- nrow(sigma)
  if (ncol(sigma) != d) {
    stop("Covariance matrix must be square")
  }

  sigma <- (sigma + t(sigma)) / 2
  diag_jitter <- ridge

  cond_num <- suppressWarnings(kappa(sigma, exact = FALSE))
  if (!is.finite(cond_num) || cond_num > cond_threshold) {
    sigma <- sigma + diag(diag_jitter, d)
  }

  chol_attempt <- try(chol(sigma), silent = TRUE)
  if (!inherits(chol_attempt, "try-error")) {
    return(chol2inv(chol_attempt))
  }

  sigma_reg <- sigma + diag(diag_jitter, d)
  chol_attempt_reg <- try(chol(sigma_reg), silent = TRUE)
  if (!inherits(chol_attempt_reg, "try-error")) {
    return(chol2inv(chol_attempt_reg))
  }

  if (requireNamespace("MASS", quietly = TRUE)) {
    return(MASS::ginv(sigma_reg))
  }

  qr.solve(sigma_reg, diag(d))
}

#' Build inverse covariance matrices for mixture components
#'
#' Computes one inverse covariance matrix per component. If a covariance matrix
#' is nearly singular, a small ridge term is added before inversion.
#'
#' @details
#' These inverse covariance matrices are used in the Gaussian score formula.
#' For a single Gaussian component with mean `mu_k` and covariance `Sigma_k`,
#' the score contribution is
#' \deqn{- (x - \mu_k)^T \Sigma_k^{-1}.}
#'
#' @param model Gaussian mixture model returned by [gmm()].
#' @param ridge Positive ridge value used when regularization is needed.
#' @param cond_threshold Condition-number threshold above which ridge
#'   regularization is applied.
#'
#' @return A list of inverse covariance matrices, one per mixture component.
#' @noRd
build_precision_cache <- function(model, ridge = 1e-6, cond_threshold = 1e10) {
  lapply(seq_len(model$nComp), function(component_idx) {
    safe_precision_matrix(
      model$sigma[, , component_idx],
      ridge = ridge,
      cond_threshold = cond_threshold
    )
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
gmm_log_component_densities <- function(model, X) {
  d <- model$d
  k <- model$nComp
  mu <- model$mu
  sigma <- model$sigma

  if (is.null(dim(X))) {
    if (d == 1) {
      x_mat <- matrix(as.numeric(X), ncol = 1)
    } else {
      x_mat <- matrix(as.numeric(X), nrow = 1)
    }
  } else {
    x_mat <- as.matrix(X)
  }

  n <- nrow(x_mat)
  if (ncol(x_mat) != d) {
    stop("X dimension does not match model dimension")
  }

  log_comp <- matrix(NA_real_, nrow = n, ncol = k)

  if (d == 1) {
    x_vec <- as.numeric(x_mat[, 1])
    for (i in seq_len(k)) {
      stdev <- sqrt(sigma[, , i])
      log_comp[, i] <- stats::dnorm(x_vec, mean = mu[i], sd = stdev, log = TRUE)
    }
  } else {
    if (!requireNamespace("mvtnorm", quietly = TRUE)) {
      stop("mvtnorm needed for this demo to work. Please install it.",
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
#' underflow when densities are very small.
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
  post <- matrix(0, nrow = nrow(log_joint), ncol = ncol(log_joint))
  valid <- !is_inf_row

  if (any(valid)) {
    shifted <- log_joint[valid, , drop = FALSE] - row_max[valid]
    post[valid, ] <- exp(shifted) / rowSums(exp(shifted))
  }
  if (any(is_inf_row)) {
    post[is_inf_row, ] <- 1 / model$nComp
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
#' # Compute score for a given gaussianmixture model and dataset
#' model <- gmm()
#' X <- rgmm(model)
#' score <- scorefunctiongmm(model = model, X = X)
scorefunctiongmm <- function(model = NULL, X = NULL) {
  if (is.null(model) || is.null(X)) {
    stop('Supply Model and Data')
  } else {
    P <- posteriorgmm(model, X)
    d <- model$d
    precision_cache <- build_precision_cache(model)
    if (d == 1) {
      n <- length(X)
      x_mat <- matrix(as.numeric(X), ncol = 1)
    } else {
      n <- dim(X)[1]
      x_mat <- as.matrix(X)
    }

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
#' # Plot pdf histogram for a given dataset
#' model <- gmm()
#' X <- rgmm(model)
#' plotgmm(data = X)
#'
#' # Plot pdf histogram for a given dataset, with lines that indicate the mean
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

    if (is.null(dim(X))) {
      if (d == 1) {
        X_mat <- matrix(X, ncol = 1)
      } else {
        X_mat <- matrix(X, nrow = 1)
      }
    } else {
      X_mat <- as.matrix(X)
    }

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
