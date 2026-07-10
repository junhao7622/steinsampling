# Stein thinning (Riabiz et al., 2022), Algorithm 1.

# ---- Public entry --------------------------------------------------------

#' Compress existing samples with Stein thinning
#'
#' Implements Algorithm 1 of Riabiz et al. (2022). Given an existing MCMC output
#' `X`, the function returns `m` row indices that define a compressed empirical
#' measure with small kernel Stein discrepancy with respect to the target score.
#'
#' @details
#' Stein thinning is a post-processing compression method. It starts after a
#' sampler has already produced rows `X[1, ], ..., X[n, ]`; it does not
#' simulate a new chain, it does not move particles, and it does not optimize
#' over continuous space. Its output is an index sequence, so the compressed
#' sample is obtained by subsetting the original rows. The input score matrix
#' `S` must have the same dimensions as `X`; row `S[i, ]` is the target score
#' \eqn{s_p(X[i,])=\nabla_x\log p(x)|_{x=X[i,]}}. Alternatively,
#' `score_function` is evaluated once to create that matrix. The Stein kernel
#' `k0` is then evaluated between rows of `X` using these scores.
#'
#' The method starts with an empty index sequence. If the selected indices after
#' `j - 1` steps are \eqn{s_1,\ldots,s_{j-1}}, the next index is chosen by minimizing
#' \deqn{
#' \frac{1}{2} k0(x_i, x_i) +
#' \sum_{l=1}^{j-1} k0(x_{s_l}, x_i).
#' }
#' The expression has the same algebraic shape as the marginal KSD decrease in
#' greedy Stein Points, but the problem being solved is different. Stein
#' thinning is choosing indices from the fixed support
#' \eqn{\{x_1,\ldots,x_n\}} to compress an existing empirical measure; it is not
#' generating new point locations. If \eqn{idx=(s_1,\ldots,s_m)} is the returned
#' sequence, the associated compressed-measure diagnostic is
#' \deqn{KSD_m^2 = \frac{1}{m^2}
#'        \sum_{a=1}^m \sum_{b=1}^m k0(x_{s_a}, x_{s_b}).}
#' After an index is selected, its Stein-kernel interaction with every candidate
#' row is added to the running objective vector. This gives the same choice as
#' recomputing the displayed objective from scratch, but avoids rebuilding the
#' whole sum at each step. If several rows have the same minimum objective
#' value, the implementation selects the first such row, matching the
#' deterministic smallest-index tie-breaking rule discussed in the paper. The
#' returned indices are a sequence, not necessarily a set: the same row may be
#' selected more than once when that best reduces the compression objective.
#'
#' The returned object is intentionally just an integer vector. If `idx` is the
#' result, the thinned sample is `X[idx, , drop = FALSE]` and the associated
#' scores are `S[idx, , drop = FALSE]`. This keeps the function close to the
#' paper's algorithm, where the selected empirical measure is represented by
#' selected support points and their multiplicities. Returning indices rather
#' than a copied sample also makes repeated selections unambiguous: a repeated
#' index means the same original support point receives repeated mass in the
#' selected empirical measure.
#'
#' This function should be read as a compression tool, not as another Stein
#' Points generator. It shares the Stein-kernel layer with [stein_points()] and
#' [sp_mcmc()], but the workflow is reversed: Stein Points and SP-MCMC construct
#' a support set by proposing new candidate locations, whereas Stein thinning
#' starts with a support set already produced by MCMC and keeps a short indexed
#' subsequence. Moving from Stein thinning to Stein Points or SP-MCMC therefore
#' changes the task from compression of existing rows to construction of new
#' support points.
#'
#' The paper describes preconditioning through a positive definite matrix in the
#' kernel distance. In this package the built-in kernels receive `M`, the matrix
#' used directly in the squared distance
#' \deqn{r(x, y) = (x - y)^T M (x - y).}
#' Let `med2` be the median squared Euclidean distance in the chosen
#' preconditioning subsample. The `"med"` option uses
#' \deqn{M = I / med2.}
#' The paper explicitly recommends a positive exception when the median distance
#' is zero, for example setting its length scale `ell = 1`. This implementation
#' follows that recommendation by replacing `med2 = 0` with `med2 = 1` and
#' issuing a warning. The fallback keeps the preconditioner positive definite,
#' but it is no longer data-adaptive; a zero median commonly indicates many
#' repeated states or poor MCMC mixing and should be diagnosed rather than
#' interpreted as a successful median estimate.
#'
#' The default `"sclmed"` option implements the scaled-median rule from the
#' Stein thinning paper:
#' \deqn{M = \log(m) I / med2,}
#' which is equivalent to the paper's
#' \eqn{\Gamma=\ell^2I} with
#' \eqn{\ell=med/\sqrt{\log(m)}} and \eqn{M=\Gamma^{-1}}. This rule requires
#' `m > 1`. It is a heuristic proposed to balance the aggregate kernel
#' interactions; because its kernel changes with the requested output size `m`,
#' Riabiz et al. explicitly state that the preceding fixed-kernel theoretical
#' analysis does not apply to `"sclmed"`.
#'
#' For both median rules, the paper's unaffected-consistency statement assumes
#' that a fixed number `n0` of MCMC states is used to construct the
#' preconditioner. The `pre_subsample` controls that computation here; choosing
#' all rows or otherwise allowing its size to grow with the input sample is
#' outside that fixed-`n0` statement. The `"smpcov"` option uses the inverse
#' empirical covariance matrix of `X`.
#'
#' @param X Numeric matrix with samples in rows.
#' @param S Optional score matrix with the same shape as `X`.
#' @param m Positive integer number of points to select. The default
#'   `pre = "sclmed"` additionally requires `m > 1`.
#' @param score_function Optional `function(X)` returning scores.
#' @param pre Preconditioning rule:
#'   `"sclmed"`, `"med"`, or `"smpcov"`.
#' @param kernel `"imq"` (default), `"gaussian_rbf"`, or a `SteinKernel`
#'   object, including objects made by [custom_stein_kernel()].
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
#' Integer vector of length `m` containing one-based row indices into `X`, in
#' the order selected by the compression algorithm. The vector is not sorted,
#' and entries may repeat. If the return value is `idx`, the thinned sample is
#' `X[idx, , drop = FALSE]`; repeated indices represent repeated support points
#' in the compressed empirical measure.
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
