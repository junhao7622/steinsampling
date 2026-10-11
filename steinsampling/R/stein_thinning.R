#' Select existing samples by Stein thinning
#'
#' Selects an ordered sequence of row indices from an existing sample using a
#' greedy Stein-discrepancy criterion. The function does not simulate or move
#' sample points.
#'
#' @details
#' Let rows of `X` be \eqn{z_1^\top,\ldots,z_n^\top} and
#' \eqn{K_{ab}=k_{0,p}(z_a,z_b)}, with \eqn{k_{0,p}} defined in
#' [stein_kernel_matrix()]. Scores may be supplied through `S` or
#' computed once using `score_function`. If the first \eqn{j-1} selected
#' indices are \eqn{\pi(1),\ldots,\pi(j-1)}, the next index is
#' \deqn{\pi(j)\in\operatorname*{arg\,min}_{i\in\{1,\ldots,n\}}
#' \left\{K_{ii}+2\sum_{\ell=1}^{j-1}K_{\pi(\ell),i}\right\}.}
#' Before the next selection, the chosen Stein-kernel row is added to the
#' running objective. Selection costs \eqn{O(nmd)} after the initial diagonal
#' calculation. Ties use the first minimum. Every row remains eligible, so an
#' index may be selected more than once and `m` may exceed \eqn{n}; repeated
#' indices represent repeated mass on the corresponding rows.
#'
#' `pre` chooses the matrix \eqn{M} in the kernel distance
#' \eqn{r_M(x,y)=(x-y)^\top M(x-y)}. With \eqn{\rho} the median pairwise
#' Euclidean distance among the rows selected by `pre_subsample`, the three
#' rules are
#' \deqn{M_{\mathrm{med}}=\frac{1}{\rho^2}I,\qquad
#' M_{\mathrm{sclmed}}=\frac{\log(m)}{\rho^2}I,\qquad
#' M_{\mathrm{smpcov}}=\widehat{\operatorname{Cov}}(X)^{-1}.}
#' `"med"` and `"sclmed"` require at least two preconditioning rows, and the
#' default `"sclmed"` also requires \eqn{m>1}; if \eqn{\rho=0}, \eqn{\rho^2}
#' is replaced by 1 with a warning. `"smpcov"` uses all rows and requires a
#' nonsingular empirical covariance matrix.
#'
#' The default base kernel is
#' \deqn{k(x,y)=\{1+r_M(x,y)\}^{-1/2}.}
#' A Gaussian RBF kernel must have a fixed positive bandwidth. For supplied
#' built-in RBF or IMQ objects, `pre` replaces a stored preconditioner and
#' warns when the matrices differ. Custom callbacks receive
#' the thinning preconditioner as `M`; see [custom_stein_kernel()] for their
#' interface.
#'
#' @param X Finite numeric vector or matrix containing \eqn{n} existing
#'   samples. Rows are samples and columns are coordinates; a vector is treated
#'   as an \eqn{n\times 1} matrix.
#' @param S Optional finite numeric \eqn{n\times d} score matrix with row
#'   \eqn{s_p(z_i)^\top}. Supply `S` or `score_function`.
#' @param m Positive integer number of indices to select.
#' @param score_function Optional function taking `X` and returning the
#'   corresponding \eqn{n\times d} matrix of target scores. When supplied, its
#'   result is used instead of `S`.
#' @param pre Preconditioner: `"sclmed"`, `"med"`, `"smpcov"`, or a
#'   symmetric positive-definite \eqn{d\times d} matrix used directly.
#' @param kernel `"imq"` (default), `"gaussian_rbf"`, or a `SteinKernel` object.
#'   Character inputs use fixed kernel parameters. To set these parameters,
#'   supply an object from [stein_kernel()]. A
#'   supplied Gaussian RBF object must have a fixed positive bandwidth.
#' @param pre_subsample For `"med"` and `"sclmed"`, either a positive integer
#'   giving the maximum number of rows used to estimate \eqn{\rho}, `Inf` for
#'   all rows, or an explicit vector of row indices. Ignored for `"smpcov"`
#'   and matrix `pre`.
#' @param pre_subsample_method Row-selection method used when `pre_subsample`
#'   is scalar: `"first"` uses the initial rows, `"even"` uses evenly spaced
#'   rows, and `"random"` samples rows randomly.
#'
#' @return
#' Integer vector \eqn{(\pi(1),\ldots,\pi(m))} containing one-based row indices
#' in selection order. It is not sorted and may contain repeated indices.
#' Extract the selected values with `X[idx]` for a vector, or
#' `X[idx, , drop = FALSE]` for a matrix.
#' @references
#' Riabiz, M., Chen, W. Y., Cockayne, J., Swietach, P., Niederer, S. A.,
#' Mackey, L., and Oates, C. J. (2022).
#' Optimal Thinning of MCMC Output.
#' Journal of the Royal Statistical Society: Series B (Statistical
#' Methodology), 84(4), 1059--1081. \doi{10.1111/rssb.12503}.
#'
#' @examples
#' # Target: N(0, 1).
#' set.seed(1)
#' X <- matrix(rnorm(500), ncol = 1)
#' idx <- stein_thinning(X, score_function = function(X) -X, m = 50)
#' selected <- X[idx, , drop = FALSE]
#' c(mean = mean(selected), second_moment = mean(selected^2))  # Targets: 0 and 1
#' @export
stein_thinning <- function(X, S = NULL, m,
                           score_function = NULL,
                           pre = c("sclmed", "med", "smpcov"),
                           kernel = "imq",
                           pre_subsample = 1000L,
                           pre_subsample_method = c("first", "even", "random")) {
  inputs <- .prepare_thinning_inputs(X, S, score_function)
  X <- inputs$X
  scores <- inputs$S

  m <- validate_integer(m, "m")
  pre <- if (is.character(pre)) {
    match.arg(pre)
  } else {
    validate_kernel_precon(pre, ncol(X))
  }
  precon <- .build_thinning_precon(
    X, m, pre, pre_subsample, pre_subsample_method
  )
  kernel_obj <- .make_thinning_kernel(kernel, precon)

  # Half the greedy objective has the same minimizer and lets each step add one
  # kernel row instead of two.
  objective <- 0.5 * k0_diag(kernel_obj, X, scores, precon = precon)
  selected <- integer(m)

  # Keep every row eligible so indices may repeat.
  for (j in seq_len(m)) {
    # which.min returns the first minimum, so ties resolve to the lowest index.
    idx <- which.min(objective)
    selected[j] <- idx
    if (j < m) {
      objective <- objective + as.numeric(stein_kernel_matrix(
        kernel_obj, X[idx, , drop = FALSE], scores[idx, , drop = FALSE],
        X, scores, precon = precon
      ))
    }
  }

  selected
}


.build_thinning_precon <- function(X, m, pre, pre_subsample,
                                   pre_subsample_method = "first") {
  if (!is.character(pre)) return(pre)

  d <- ncol(X)

  if (identical(pre, "smpcov")) {
    # smpcov uses all rows. Only the median rules use pre_subsample.
    return(solve(stats::cov(X)))
  }

  if (identical(pre, "sclmed") && m <= 1L) {
    stop("sclmed preconditioner requires m > 1 because its scale uses log(m)",
         call. = FALSE)
  }
  med_sq <- .med_squared_distance(X, pre_subsample, pre_subsample_method)
  # sclmed scales the median rule by log m, as in Riabiz et al. 2022.
  scale <- if (identical(pre, "sclmed")) log(m) / med_sq else 1 / med_sq
  diag(d) * scale
}

.med_squared_distance <- function(X, pre_subsample, pre_subsample_method = "first") {
  rows <- .pre_subsample_rows(nrow(X), pre_subsample, pre_subsample_method)
  if (length(rows) < 2L) {
    stop("Median preconditioning requires at least two selected rows",
         call. = FALSE)
  }
  X <- X[rows, , drop = FALSE]

  med_sq <- stats::median(stats::dist(X))^2
  if (!is.finite(med_sq)) {
    stop("Median pairwise distance is not finite", call. = FALSE)
  }
  if (med_sq == 0) {
    warning(
      "Median pairwise distance is zero; using 1 instead; check `X` for repeated rows",
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
    if (any(round(pre_subsample) > n)) {
      stop("pre_subsample indices must be valid row indices", call. = FALSE)
    }
    rows <- as.integer(round(pre_subsample))
  } else {
    method <- match.arg(method, c("first", "even", "random"))
    if (!is.numeric(pre_subsample) || length(pre_subsample) != 1L ||
        is.na(pre_subsample) || pre_subsample <= 0 ||
        (!is.infinite(pre_subsample) &&
         abs(pre_subsample - round(pre_subsample)) > sqrt(.Machine$double.eps))) {
      stop(
        "pre_subsample must be a positive integer, Inf, or integer row indices",
        call. = FALSE
      )
    }
    n0 <- as.integer(min(n, round(pre_subsample)))
    if (n0 < 1L) {
      stop("pre_subsample must select at least one row", call. = FALSE)
    }
    rows <- switch(
      method,
      first = seq_len(n0),
      even = as.integer(seq(1L, n, length.out = n0)),
      random = sample.int(n, n0)
    )
  }
  rows
}


.make_thinning_kernel <- function(kernel, precon) {
  if (inherits(kernel, "SteinKernel")) {
    require_fixed_gaussian_rbf(kernel, "Stein thinning")
    if (kernel$type %in% c("imq", "gaussian_rbf")) {
      if (!is.null(kernel$precon) &&
          !isTRUE(all.equal(unname(kernel$precon), unname(precon)))) {
        warning("`pre` replaces the preconditioner carried by `kernel`",
                call. = FALSE)
      }
    }
    return(kernel)
  }

  # The preconditioner sets the scale, so the built-in kernels use unit
  # parameters, c = 1 for IMQ and 2h^2 = 1 for RBF.
  kernel <- match.arg(kernel, c("imq", "gaussian_rbf"))
  if (identical(kernel, "imq")) {
    return(stein_kernel(type = "imq", precon = precon))
  }
  stein_kernel(type = "gaussian_rbf", h = sqrt(1 / 2), precon = precon)
}


.prepare_thinning_inputs <- function(X, S, score_function) {
  X <- .as_rows(X, "X")

  if (!is.null(score_function)) {
    # score_function takes precedence when both inputs are supplied.
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
