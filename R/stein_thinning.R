# Stein thinning. `stein_thinning()` selects an ordered, repeatable sequence of
# rows from an existing sample. Private helpers prepare scores, construct the
# preconditioner, choose the kernel, and update the running greedy objective.

# ---- Public entry --------------------------------------------------------

#' Select existing samples by Stein thinning
#'
#' Selects an ordered sequence of row indices from an existing sample using the
#' greedy Stein-thinning criterion of Riabiz et al. (2022). The function does
#' not simulate or move sample points.
#'
#' @details
#' Let rows of `X` be \eqn{z_1^\top,\ldots,z_n^\top} and
#' \deqn{K_{ab}=k_{0,p}(z_a,z_b).}
#' Scores may be supplied through `S` or computed once using `score_function`.
#' If the first \eqn{j-1} selected indices are
#' \eqn{\pi(1),\ldots,\pi(j-1)}, the next index is
#' \deqn{\pi(j)\in\operatorname*{arg\,min}_{i\in\{1,\ldots,n\}}
#' \left\{K_{ii}+2\sum_{\ell=1}^{j-1}K_{\pi(\ell),i}\right\}.}
#' The implementation stores one half of this objective,
#' \deqn{\frac12K_{ii}+\sum_{\ell=1}^{j-1}K_{\pi(\ell),i},}
#' which has the same minimizer. Each step adds one Stein-kernel row to the
#' running objective, giving selection cost \eqn{O(nmd)} after the initial
#' diagonal calculation.
#'
#' Ties use the first minimum. An index may be selected more than once, so `m`
#' may exceed \eqn{n}; repeated indices represent repeated mass on the
#' corresponding input rows.
#'
#' Let \eqn{M=\texttt{precon}} and
#' \deqn{r_M(x,y)=(x-y)^\top M(x-y).}
#' For the median rules, let \eqn{\rho} be the median pairwise Euclidean
#' distance among the rows selected by `pre_subsample`. The rules are
#' \deqn{M_{\mathrm{med}}=\frac{1}{\rho^2}I,\qquad
#' M_{\mathrm{sclmed}}=\frac{\log(m)}{\rho^2}I,\qquad
#' M_{\mathrm{smpcov}}=\widehat{\operatorname{Cov}}(X)^{-1}.}
#' The default `"sclmed"` requires \eqn{m>1}. Both median rules require at
#' least two preconditioning rows. If \eqn{\rho=0}, the package replaces
#' \eqn{\rho^2} by 1 and warns. `"smpcov"` uses all rows and requires a
#' nonsingular empirical covariance matrix.
#'
#' When `pre_subsample` is scalar, `"first"`, `"even"`, and `"random"` use
#' initial, evenly spaced, and randomly sampled rows. An explicit vector of row
#' indices is used directly.
#'
#' The default base kernel is
#' \deqn{k(x,y)=\{1+r_M(x,y)\}^{-1/2}.}
#' A Gaussian RBF kernel is supported but must have a fixed positive bandwidth;
#' by default its use produces a warning. For supplied built-in RBF or IMQ
#' objects, `pre` replaces the object's existing preconditioner. Custom
#' callbacks may use the `precon` argument passed by Stein thinning.
#'
#' @param X Finite numeric vector or matrix containing \eqn{n} existing
#'   samples. Rows are samples and columns are coordinates; a vector is treated
#'   as an \eqn{n\times1} matrix.
#' @param S Optional finite numeric \eqn{n\times d} score matrix with row
#'   \eqn{s_p(z_i)^\top}. Supply `S` or `score_function`.
#' @param m Positive integer number of indices to select. Repetition is
#'   allowed, so `m` may exceed \eqn{n}. The default `"sclmed"` rule requires
#'   `m > 1`.
#' @param score_function Optional function that accepts `X` and returns an
#'   \eqn{n\times d} score matrix. When supplied, its result is used instead of
#'   `S`.
#' @param pre Preconditioning rule: `"sclmed"`, `"med"`, or `"smpcov"`.
#' @param kernel Either `"imq"`, `"gaussian_rbf"`, or a `SteinKernel` object.
#'   The default is `"imq"`. A supplied Gaussian RBF object must have a fixed
#'   positive bandwidth.
#' @param pre_subsample For `"med"` and `"sclmed"`, either a positive integer
#'   giving the maximum number of rows used to estimate \eqn{\rho}, `Inf` for
#'   all rows, or an explicit vector of row indices. Ignored for `"smpcov"`.
#' @param pre_subsample_method Row-selection method used when `pre_subsample`
#'   is scalar: `"first"`, `"even"`, or `"random"`.
#' @param verbose_rbf_warning Logical; when `TRUE` (the default), supplying a
#'   Gaussian RBF kernel warns that Riabiz et al. (2022) use IMQ as the Stein
#'   thinning default. Set `FALSE` to silence it.
#' @param ... Kernel parameters used when `kernel` is a character string: `c`
#'   and `beta` for IMQ, or `h` and `sigma` for Gaussian RBF. The compatibility
#'   argument `sigma2` is also accepted for RBF when neither `h` nor `sigma` is
#'   supplied.
#'
#' @return
#' Integer vector \eqn{(\pi(1),\ldots,\pi(m))} containing one-based row indices
#' in selection order. It is not sorted and may contain repeated indices. The
#' selected sample is `X[idx, , drop = FALSE]`.
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
  scores <- inputs$S

  m <- validate_integer(m, "m")
  pre <- match.arg(pre)
  pre_subsample_method <- match.arg(pre_subsample_method)

  precon <- .build_thinning_precon(
    X, m, pre, pre_subsample, pre_subsample_method
  )
  kernel_obj <- .make_thinning_kernel(kernel, precon, ...)

  if (inherits(kernel_obj, "SteinKernel_gaussian_rbf") &&
      isTRUE(verbose_rbf_warning)) {
    warning(
      "Gaussian RBF is supported, but Riabiz et al. (2022) use IMQ as the core Stein thinning default.",
      call. = FALSE
    )
  }

  objective <- 0.5 * k0_diag(kernel_obj, X, scores, precon = precon)
  selected <- integer(m)

  for (j in seq_len(m)) {
    # which.min() returns the first minimum, giving deterministic tie-breaking.
    idx <- which.min(objective)
    selected[j] <- idx
    objective <-
      objective + .kP_row_vector(idx, X, scores, kernel_obj, precon)
  }

  selected
}


# ---- Algorithm 1 kernel calls --------------------------------------------

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

# ---- Preconditioners ------------------------------------

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
    if (!is.numeric(pre_subsample) || any(!is.finite(pre_subsample)) ||
        any(pre_subsample < 1) ||
        any(abs(pre_subsample - round(pre_subsample)) >
            sqrt(.Machine$double.eps))) {
      stop("pre_subsample indices must be positive integers", call. = FALSE)
    }
    rows <- as.integer(round(pre_subsample))
  } else {
    if (!is.numeric(pre_subsample) || length(pre_subsample) != 1L ||
        is.na(pre_subsample) || pre_subsample <= 0 ||
        (!is.infinite(pre_subsample) &&
         abs(pre_subsample - round(pre_subsample)) > sqrt(.Machine$double.eps))) {
      stop(
        "pre_subsample must be a positive integer, Inf, or integer row indices",
        call. = FALSE
      )
    }
    n0 <- if (is.infinite(pre_subsample)) {
      n
    } else {
      as.integer(round(pre_subsample))
    }
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


# ---- Kernel selection ----------------------------------------------------

.make_thinning_kernel <- function(kernel, precon, ...) {
  if (inherits(kernel, "SteinKernel")) {
    require_fixed_gaussian_rbf(kernel, "Stein thinning")
    if (inherits(kernel, "SteinKernel_imq") ||
        inherits(kernel, "SteinKernel_gaussian_rbf")) {
      if (!is.null(kernel$precon) &&
          !isTRUE(all.equal(unname(kernel$precon), unname(precon)))) {
        warning("`pre` replaces the preconditioner carried by `kernel`.",
                call. = FALSE)
      }
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


# ---- Input assembly ------------------------------------------------------

.prepare_thinning_inputs <- function(X, S, score_function) {
  X <- .as_rows(X, "X")

  if (!is.null(score_function)) {
    if (!is.function(score_function)) {
      stop("`score_function` must be a function", call. = FALSE)
    }
    S <- .as_score_matrix(score_function(X), X)
  } else if (is.null(S)) {
    stop("provide `S` or `score_function`", call. = FALSE)
  } else {
    S <- .as_rows(S, "S")
    if (!identical(dim(X), dim(S))) {
      stop("`X` and `S` must have the same dimensions", call. = FALSE)
    }
  }
  list(X = X, S = S)
}
