# Stein thinning (Riabiz et al., 2022), Algorithm 1.

# ---- Public entry --------------------------------------------------------

#' Stein thinning index selection.
#'
#' Implements Algorithm 1 from Riabiz et al. (2022):
#' at each step choose the point minimising
#' `1/2 kP(x_i, x_i) + sum_{j' < j} kP(x_pi[j'], x_i)`.
#'
#' @param X Numeric matrix (`n` x `d`) of sample states.
#' @param S Optional score matrix with the same shape as `X`.
#' @param m Number of points to select.
#' @param score_function Optional `function(X)` returning scores.
#' @param pre One of the paper's three preconditioners:
#'   `"sclmed"`, `"med"`, or `"smpcov"`.
#' @param kernel `"imq"` (default), `"gaussian_rbf"`, or a `SteinKernel`
#'   object, including objects made by [custom_stein_kernel()].
#' @param pre_subsample Max rows used for the median heuristic, or an integer
#'   vector of row indices to use directly.
#' @param pre_subsample_method How to choose rows when `pre_subsample` is a
#'   scalar: `"first"` (default), `"even"`, or `"random"`.
#' @param sclmed_log Quantity inside the `sclmed` logarithm: `"m"` (default) or
#'   `"pre_subsample"`.
#' @param verbose_rbf_warning Warn when using RBF instead of the paper's IMQ
#'   default.
#' @param ... Kernel parameters forwarded to [stein_kernel()], such as `c`,
#'   `beta`, `h`, `sigma`, or `sigma2` for RBF.
#'
#' @return Integer vector of length `m` giving selected row indices of `X`.
#'
#' @examples
#' X <- matrix(rnorm(6), ncol = 1)
#' S <- -X
#' stein_thinning(X, S = S, m = 2, pre_subsample = 3)
#' @export
stein_thinning <- function(X, S = NULL, m,
                           score_function = NULL,
                           pre = c("sclmed", "med", "smpcov"),
                           kernel = "imq",
                           pre_subsample = 1000L,
                           pre_subsample_method = c("first", "even", "random"),
                           sclmed_log = c("m", "pre_subsample"),
                           verbose_rbf_warning = TRUE, ...) {
  inputs <- .prepare_thinning_inputs(X, S, score_function)
  X <- inputs$X
  S <- inputs$S

  m <- as.integer(m)
  pre <- match.arg(pre)
  pre_subsample_method <- match.arg(pre_subsample_method)
  sclmed_log <- match.arg(sclmed_log)

  precon <- .build_thinning_precon(
    X, m, pre, pre_subsample, pre_subsample_method, sclmed_log
  )
  kernel <- .make_thinning_kernel(kernel, precon, ...)

  if (inherits(kernel, "SteinKernel_gaussian_rbf") && isTRUE(verbose_rbf_warning)) {
    warning(
      "Gaussian RBF is supported, but Riabiz et al. (2022) use IMQ as the core Stein thinning default.",
      call. = FALSE
    )
  }

  objective <- 0.5 * .kP_diag_vector(X, S, kernel, precon)
  selected <- integer(m)

  for (j in seq_len(m)) {
    idx <- which.min(objective)
    selected[j] <- idx
    objective <- objective + .kP_row_vector(idx, X, S, kernel, precon)
  }

  selected
}


# ---- Algorithm 1 kernel calls ------------------------------------------

.kP_row_vector <- function(idx, X, S, kernel, precon) {
  as.numeric(stein_kernel_matrix(
    kernel,
    X[idx, , drop = FALSE],
    S[idx, , drop = FALSE],
    X,
    S,
    precon = precon
  ))
}

.kP_diag_vector <- function(X, S, kernel, precon) {
  vapply(seq_len(nrow(X)), function(i) {
    stein_kernel_matrix(
      kernel,
      X[i, , drop = FALSE],
      S[i, , drop = FALSE],
      X[i, , drop = FALSE],
      S[i, , drop = FALSE],
      precon = precon
    )[1, 1]
  }, numeric(1))
}


# ---- Preconditioners from Section 2.3 ----------------------------------

.build_thinning_precon <- function(X, m, pre, pre_subsample,
                                   pre_subsample_method = "first",
                                   sclmed_log = "m") {
  d <- ncol(X)
  pre_subsample_method <- match.arg(pre_subsample_method, c("first", "even", "random"))
  sclmed_log <- match.arg(sclmed_log, c("m", "pre_subsample"))

  if (identical(pre, "smpcov")) {
    return(solve(stats::cov(X)))
  }

  med_sq <- .med_squared_distance(X, pre_subsample, pre_subsample_method)
  log_arg <- if (identical(sclmed_log, "m")) m else attr(med_sq, "n0", exact = TRUE)
  if (identical(pre, "sclmed") && log_arg <= 1L) {
    stop("sclmed preconditioner requires log argument > 1.", call. = FALSE)
  }
  scale <- if (identical(pre, "sclmed")) log(log_arg) / med_sq else 1 / med_sq
  diag(d) * scale
}

.med_squared_distance <- function(X, pre_subsample, pre_subsample_method = "first") {
  rows <- .pre_subsample_rows(nrow(X), pre_subsample, pre_subsample_method)
  X <- X[rows, , drop = FALSE]

  med_sq <- stats::median(stats::dist(X))^2
  if (!is.finite(med_sq) || med_sq <= 0) {
    stop("Too few unique samples in X.", call. = FALSE)
  }
  attr(med_sq, "n0") <- length(rows)
  med_sq
}

.pre_subsample_rows <- function(n, pre_subsample, method) {
  if (length(pre_subsample) > 1L) {
    rows <- as.integer(pre_subsample)
  } else {
    n0 <- if (is.infinite(pre_subsample)) n else as.integer(pre_subsample)
    n0 <- min(n, n0)
    if (!is.finite(n0) || n0 < 1L) {
      stop("pre_subsample must select at least one row.", call. = FALSE)
    }
    rows <- switch(
      method,
      first = seq_len(n0),
      even = as.integer(seq(1L, n, length.out = n0)),
      random = sample.int(n, n0)
    )
  }

  if (length(rows) < 1L || anyNA(rows) || any(rows < 1L | rows > n)) {
    stop("pre_subsample indices must be valid row indices.", call. = FALSE)
  }
  rows
}


# ---- Kernel selection ---------------------------------------------------

.make_thinning_kernel <- function(kernel, precon, ...) {
  if (inherits(kernel, "SteinKernel")) {
    if (inherits(kernel, "SteinKernel_imq") ||
        inherits(kernel, "SteinKernel_gaussian_rbf")) {
      kernel$precon <- precon
    }
    return(kernel)
  }

  kernel <- match.arg(kernel, c("imq", "gaussian_rbf"))
  dots <- list(...)

  if (identical(kernel, "imq")) {
    c_len <- if (is.null(dots$c)) 1 else dots$c
    beta <- if (is.null(dots$beta)) -0.5 else dots$beta
    if (is.numeric(beta) && length(beta) == 1L && is.finite(beta) && beta <= -1) {
      warning("Riabiz et al. (2022) assume IMQ beta in (-1, 0); continuing with supplied beta.", call. = FALSE)
    }
    return(stein_kernel(type = "imq", c = c_len, beta = beta, precon = precon))
  }

  h <- dots$h
  sigma <- dots$sigma
  if (is.null(h) && is.null(sigma) && !is.null(dots$sigma2)) {
    h <- sqrt(dots$sigma2 / 2)
  } else if (is.null(h) && is.null(sigma)) {
    h <- sqrt(1 / 2)
  }
  stein_kernel(type = "gaussian_rbf", h = h, sigma = sigma, precon = precon)
}


# ---- Input assembly -----------------------------------------------------

.prepare_thinning_inputs <- function(X, S, score_function) {
  X <- as.matrix(X)
  if (!is.null(score_function)) {
    S <- score_function(X)
  } else if (is.null(S)) {
    stop("Provide S or score_function.")
  }

  S <- as.matrix(S)
  if (!identical(dim(X), dim(S))) stop("X and S must have the same dimensions.")
  list(X = X, S = S)
}
