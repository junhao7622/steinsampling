# Stein thinning (Riabiz et al., 2022), Algorithm 1.

# ---- Public entry --------------------------------------------------------

#' Compress existing samples with Stein thinning
#'
#' Implements Algorithm 1 of Riabiz et al. (2022). Given an existing MCMC output
#' `X`, the function returns `m` row indices that define a compressed empirical
#' measure with small kernel Stein discrepancy with respect to the target score.
#'
#' @details
#' Stein thinning post-processes rows `X[1, ],...,X[n, ]` already produced by a
#' sampler: it neither simulates a chain, moves particles, nor optimizes over
#' continuous space. It returns indices for subsetting those rows. Score matrix
#' `S` must match `X`, with row `S[i, ]` equal to
#' \eqn{s_p(X[i,])=\nabla_x\log p(x)|_{x=X[i,]}}. Alternatively,
#' `score_function` is evaluated once to create `S`; the Stein kernel `k0` then
#' uses these scores between rows of `X`.
#'
#' Starting from an empty sequence, after selecting
#' \eqn{s_1,\ldots,s_{j-1}} the method minimizes
#' \deqn{
#' \frac{1}{2} k0(x_i, x_i) +
#' \sum_{l=1}^{j-1} k0(x_{s_l}, x_i).
#' }
#' Although algebraically shaped like the marginal KSD decrease in greedy Stein
#' Points, this selects indices from fixed support
#' \eqn{\{x_1,\ldots,x_n\}} rather than generating locations. For returned
#' \eqn{idx=(s_1,\ldots,s_m)}, the diagnostic is
#' \deqn{KSD_m^2 = \frac{1}{m^2}
#'        \sum_{a=1}^m \sum_{b=1}^m k0(x_{s_a}, x_{s_b}).}
#' Each selected row's interactions are added to a running objective, equivalent
#' to recomputing the sum but avoiding that work. If several rows have the same
#' smallest value, the code takes the first one; the paper says any one may be
#' used. Indices form a sequence, not necessarily a set: rows may
#' repeat when this best lowers the objective.
#'
#' The integer vector directly represents the paper's support points and
#' multiplicities: use `X[idx, , drop = FALSE]` and
#' `S[idx, , drop = FALSE]`. A repeated index unambiguously gives the same
#' original support point repeated mass.
#'
#' Thus this is compression, not another point generator. It shares the kernel
#' layer with [stein_points()] and [sp_mcmc()], but those methods propose new
#' locations; Stein thinning keeps a short indexed subsequence of an existing,
#' typically MCMC-produced support set.
#'
#' Built-in kernels receive the paper's positive definite preconditioner `M`
#' directly in
#' \deqn{r(x, y) = (x - y)^T M (x - y).}
#' If `med2` is the median squared Euclidean distance in the chosen
#' preconditioning subsample, `"med"` uses
#' \deqn{M = I / med2.}
#' For zero median the paper recommends a positive exception such as length
#' scale `ell = 1`; accordingly, the implementation warns and replaces
#' `med2 = 0` by `1`. This remains positive definite but is no longer
#' data-adaptive; zero commonly indicates repeated states or poor MCMC mixing
#' and should be diagnosed, not treated as a successful estimate.
#'
#' The paper's default scaled-median rule, `"sclmed"`, is
#' \deqn{M = \log(m) I / med2,}
#' equivalent to \eqn{\Gamma=\ell^2I},
#' \eqn{\ell=med/\sqrt{\log(m)}} and \eqn{M=\Gamma^{-1}}. This rule requires
#' `m > 1`. It heuristically balances aggregate kernel interactions, but changes
#' the kernel with `m`; Riabiz et al. therefore state that their preceding
#' fixed-kernel theory does not cover `"sclmed"`.
#'
#' For both median rules, unaffected consistency assumes a fixed `n0` MCMC
#' states construct the preconditioner. `pre_subsample` controls this here;
#' using all rows or letting its size grow lies outside that statement.
#' `"smpcov"` uses the inverse empirical covariance of `X`.
#'
#' @param X Numeric matrix with samples in rows.
#' @param S Optional score matrix with the same shape as `X`.
#' @param m Positive integer number of points to select. The default
#'   `pre = "sclmed"` additionally requires `m > 1`.
#' @param score_function Optional `function(X)` returning scores.
#' @param pre Preconditioning rule:
#'   `"sclmed"`, `"med"`, or `"smpcov"`.
#' @param kernel `"imq"` (default), `"gaussian_rbf"`, or a `SteinKernel`
#'   object, including objects made by [custom_stein_kernel()]. A supplied
#'   Gaussian RBF object must have a fixed positive bandwidth `h`; Stein
#'   thinning uses one fixed kernel throughout Algorithm 1.
#' @param pre_subsample Maximum number of rows used for median-based scaling,
#'   or a vector of row indices to use directly.
#' @param pre_subsample_method How to choose rows when `pre_subsample` is a
#'   scalar: `"first"`, `"even"`, or `"random"`.
#' @param verbose_rbf_warning Warn when using RBF instead of the usual IMQ
#'   Stein thinning default.
#' @param ... Kernel parameters forwarded to [stein_kernel()], such as `c`,
#'   `beta`, `h`, `sigma`, or `sigma2` for RBF.
#'
#' @return
#' Length-`m` integer vector of one-based row indices in selection order. It is
#' unsorted and may repeat; `X[idx, , drop = FALSE]` is the thinned sample, with
#' repeats representing repeated support points in the empirical measure.
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
                           verbose_rbf_warning = TRUE, ...) {
  inputs <- .prepare_thinning_inputs(X, S, score_function)
  X <- inputs$X
  S <- inputs$S

  m <- validate_positive_integer(m, "m")
  pre <- match.arg(pre)
  pre_subsample_method <- match.arg(pre_subsample_method)

  precon <- .build_thinning_precon(
    X, m, pre, pre_subsample, pre_subsample_method
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
    # which.min() returns the first minimum, giving deterministic tie-breaking.
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
                                   pre_subsample_method = "first") {
  d <- ncol(X)
  pre_subsample_method <- match.arg(pre_subsample_method, c("first", "even", "random"))

  if (identical(pre, "smpcov")) {
    return(solve(stats::cov(X)))
  }

  med_sq <- .med_squared_distance(X, pre_subsample, pre_subsample_method)
  if (identical(pre, "sclmed") && m <= 1L) {
    stop("sclmed preconditioner requires m > 1 because its scale uses log(m).",
         call. = FALSE)
  }
  scale <- if (identical(pre, "sclmed")) log(m) / med_sq else 1 / med_sq
  diag(d) * scale
}

.med_squared_distance <- function(X, pre_subsample, pre_subsample_method = "first") {
  rows <- .pre_subsample_rows(nrow(X), pre_subsample, pre_subsample_method)
  if (length(rows) < 2L) {
    stop("Median preconditioning requires at least two selected rows.",
         call. = FALSE)
  }
  X <- X[rows, , drop = FALSE]

  med_sq <- stats::median(stats::dist(X))^2
  if (!is.finite(med_sq)) {
    stop("Median pairwise distance is not finite.", call. = FALSE)
  }
  if (med_sq == 0) {
    warning(
      paste0(
        "Median pairwise distance is zero; using the Riabiz et al. ",
        "positive exception ell = 1 (med2 = 1). Check for repeated states ",
        "or poor MCMC mixing."
      ),
      call. = FALSE
    )
    med_sq <- 1
  }
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
    require_fixed_gaussian_rbf(kernel, "Stein thinning")
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
