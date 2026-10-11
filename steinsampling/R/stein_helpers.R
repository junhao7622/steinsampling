# Helpers shared by several files, mainly argument checks, the median scale and
# the KSD input preparation.

#' @rdname kernel_scales
#' @export
find_median_distance <- function(Z) {
  Z <- .as_rows(Z, "Z", 2L)

  dists <- as.numeric(stats::dist(Z, method = "euclidean"))
  med <- stats::median(dists)^2
  if (!is.finite(med)) {
    stop("Pairwise distances in Z are not finite", call. = FALSE)
  }
  if (med == 0) {
    warning(
      "Median pairwise distance is zero; using 1e-5 instead; check `Z` for repeated rows",
      call. = FALSE
    )
    med <- 1e-5
  }

  med
}


.as_rows <- function(x, arg_name = "X", min_rows = 1L) {
  if (is.null(dim(x))) {
    if (!is.numeric(x)) {
      stop(sprintf("`%s` must be numeric", arg_name), call. = FALSE)
    }
    x <- matrix(x, ncol = 1)
  } else if (is.data.frame(x)) {
    if (!all(vapply(x, is.numeric, logical(1L)))) {
      stop(sprintf("`%s` must be numeric", arg_name), call. = FALSE)
    }
    x <- as.matrix(x)
  } else {
    x <- as.matrix(x)
  }

  if (!is.numeric(x) || any(!is.finite(x))) {
    stop(
      sprintf("`%s` must contain only finite numeric values", arg_name),
      call. = FALSE
    )
  }
  if (nrow(x) < min_rows) {
    stop(
      sprintf("`%s` must have nrow >= %d", arg_name, min_rows),
      call. = FALSE
    )
  }

  x
}

.as_score_matrix <- function(scores, x_mat) {
  n <- nrow(x_mat)
  d <- ncol(x_mat)

  if (is.null(dim(scores))) {
    if (!is.numeric(scores) || d != 1L || length(scores) != n) {
      stop(
        "`score_function(X)` must have shape n x d (or length n when d = 1)",
        call. = FALSE
      )
    }
    scores <- matrix(scores, ncol = 1L)
  } else {
    scores <- as.matrix(scores)
  }

  if (!is.numeric(scores) || nrow(scores) != n || ncol(scores) != d ||
    any(!is.finite(scores))) {
    stop(
      "`score_function(X)` must be a finite numeric matrix with shape n x d",
      call. = FALSE
    )
  }

  scores
}

validate_integer <- function(x, arg_name, min_value = 1L) {
  if (!is.numeric(x) || length(x) != 1 || !is.finite(x) ||
    abs(x - round(x)) > sqrt(.Machine$double.eps) || round(x) < min_value ||
    round(x) > .Machine$integer.max) {
    stop(sprintf(
      "`%s` must be a %s integer no larger than %d", arg_name,
      if (min_value > 0L) "positive" else "nonnegative", .Machine$integer.max
    ), call. = FALSE)
  }
  as.integer(round(x))
}

validate_flag <- function(x, arg_name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("`%s` must be TRUE or FALSE", arg_name), call. = FALSE)
  }
  x
}

validate_positive_scalar <- function(x, arg_name) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) || x <= 0) {
    stop(sprintf("`%s` must be a finite positive scalar", arg_name), call. = FALSE)
  }
  as.numeric(x)
}

# Compare the off-diagonal size with the diagonal, so the check does not depend
# on the scale of K0.
.warn_if_degenerate_stein_matrix <- function(offdiag_max, diag_max,
                                             tol = 1e-12) {
  degenerate <- offdiag_max == 0 ||
    (diag_max > 0 && offdiag_max <= tol * diag_max)
  if (isTRUE(degenerate)) {
    warning("The Stein kernel matrix is nearly diagonal; the KSD bootstrap may be degenerate",
            call. = FALSE)
  }
  invisible(NULL)
}

.resolve_block_settings <- function(n, block_size, block_threshold) {
  n <- validate_integer(n, "n")
  block_threshold <- validate_integer(block_threshold, "block_threshold")

  auto_block <- n > block_threshold
  block_size <- if (is.null(block_size)) {
    if (auto_block) min(1024L, n) else n
  } else {
    min(validate_integer(block_size, "block_size"), n)
  }

  list(
    use_block_mode = auto_block || block_size < n,
    block_size = block_size
  )
}

.prepare_ksd_inputs <- function(X, score_function, scaling = NULL,
                                kernel = c("gaussian_rbf", "imq"),
                                imq_beta = -0.5) {
  if (!is.function(score_function)) {
    stop("`score_function` must be a function", call. = FALSE)
  }
  x_mat <- .as_rows(X, "X", 2L)
  scores <- .as_score_matrix(score_function(x_mat), x_mat)

  if (inherits(kernel, "SteinKernel")) {
    kernel_obj <- kernel
    kernel_name <- kernel_obj$type
    kernel_scale <- scale2_kernel(kernel_obj)
    if (is.null(kernel_scale)) kernel_scale <- NA_real_
    if (identical(kernel_obj$type, "gaussian_rbf") &&
        !is.finite(kernel_scale)) {
      if (is.null(scaling)) {
        scaling <- resolve_gaussian_rbf_scale2(kernel_obj, x_mat)
      }
      validate_positive_scalar(scaling, "scaling")
      kernel_obj <- scale2_kernel(kernel_obj, scaling)
    } else {
      if (!is.null(scaling)) {
        warning("`scaling` is ignored because the kernel object sets its own scale",
                call. = FALSE)
      }
      scaling <- kernel_scale
    }
  } else {
    kernel_name <- match.arg(kernel, c("gaussian_rbf", "imq"))
    if (is.null(scaling)) {
      scaling <- find_median_distance(x_mat)
    }
    validate_positive_scalar(scaling, "scaling")
    kernel_obj <- instantiate_kernel(kernel_name, scaling, imq_beta)
  }

  list(
    X = x_mat,
    scores = scores,
    kernel_obj = kernel_obj,
    kernel_name = kernel_name,
    scaling = scaling
  )
}

.validate_stein_matrix <- function(K0) {
  if (is.null(dim(K0))) {
    stop("K0 must be a square numeric matrix", call. = FALSE)
  }
  K0 <- as.matrix(K0)
  if (!is.numeric(K0) || nrow(K0) != ncol(K0) || nrow(K0) < 2L ||
      any(!is.finite(K0))) {
    stop("K0 must be a finite square numeric matrix with at least two rows", call. = FALSE)
  }
  K0
}

# The KSD of j points is the square root of their kernel sum ss, divided by j.
# Roundoff can make ss slightly negative, so it is clipped at 0.
.running_ksd <- function(ss, j) {
  if (ss < -sqrt(.Machine$double.eps)) warning("Accumulated squared KSD is negative; using 0", call. = FALSE)
  sqrt(max(ss, 0)) / j
}

.last_value <- function(v) if (length(v)) v[length(v)] else NA_real_

.point_set_summary <- function(object, evaluation_label, class) {
  X <- object$X
  structure(
    list(
      method = object$method,
      kernel = object$kernel$type,
      n = nrow(X),
      d = ncol(X),
      evaluation_label = evaluation_label,
      n_eval_total = .last_value(object$cum_n_eval),
      call = object$call
    ),
    class = class
  )
}
