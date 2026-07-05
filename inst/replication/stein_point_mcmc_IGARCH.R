## SP-MCMC IGARCH example
##
## Runs the IGARCH experiment from Chen et al. (2019), compares several
## sampling methods, and saves the energy-distance trajectory plot.

replication_dir <- normalizePath(
  if (dir.exists(file.path("inst", "replication"))) file.path("inst", "replication") else ".",
  mustWork = FALSE
)
suppressPackageStartupMessages(library(steinsampling))

set.seed(2026L)


## ------------------------------------------------------------------
## Configuration
## ------------------------------------------------------------------

# Full replication: N_POINTS <- 1000L
N_POINTS <- 30L
# Full replication: N_REF_MCMC <- 100000L
N_REF_MCMC <- 300L
# Full replication: N_REF_KEEP <- 20000L
N_REF_KEEP <- 150L
# Full replication: N_SVGD_ITER <- 200L
N_SVGD_ITER <- 20L
N_DIM <- 2L
# Full replication: N_THIN <- 5L
N_THIN <- 2L
# Full replication: M_SPMCMC <- 5L
M_SPMCMC <- 2L

LB <- c(0.002, 0.05)
UB <- c(0.04, 0.2)
X0 <- (LB + UB) / 2
V_MCMC <- diag(5e-4, N_DIM)

# Full replication: ADAPT_EPOCH <- c(rep(500L, 15L), 1000L, 5000L, N_REF_MCMC)
ADAPT_EPOCH <- c(50L, 100L, N_REF_MCMC)
FIGURE3_PDF <- file.path(replication_dir, "stein_point_mcmc_IGARCH_energy.pdf")
SPX_DATA <- file.path(
  replication_dir, "data", "stein_point_mcmc_igarch",
  "data_spx.mat"
)


## ------------------------------------------------------------------
## SPX data reader
## ------------------------------------------------------------------

read_u32 <- function(x, pos) {
  sum(as.numeric(as.integer(x[pos + 0:3])) * c(1, 256, 256^2, 256^3))
}

read_mat_element <- function(x, pos) {
  raw_tag <- read_u32(x, pos)
  small_type <- raw_tag %% 65536
  small_len <- floor(raw_tag / 65536)
  if (small_len > 0 && small_len <= 4) {
    data_start <- pos + 4L
    return(list(
      type = small_type, bytes = small_len,
      data = x[data_start:(data_start + small_len - 1L)],
      next_pos = pos + 8L
    ))
  }

  bytes <- read_u32(x, pos + 4L)
  data_start <- pos + 8L
  data_end <- data_start + bytes - 1L
  pad <- (8L - (bytes %% 8L)) %% 8L
  list(
    type = raw_tag, bytes = bytes, data = x[data_start:data_end],
    next_pos = data_end + 1L + pad
  )
}

read_from_raw <- function(raw_data, what, n, size, signed = TRUE) {
  con <- rawConnection(raw_data)
  on.exit(close(con), add = TRUE)
  readBin(con, what, n = n, size = size, endian = "little", signed = signed)
}

parse_mat_matrix <- function(payload) {
  outer <- read_mat_element(payload, 1L)
  if (outer$type != 14L) stop("Expected miMATRIX element in MAT file")

  x <- outer$data
  pos <- 1L
  flags <- read_mat_element(x, pos)
  pos <- flags$next_pos
  dims_el <- read_mat_element(x, pos)
  pos <- dims_el$next_pos
  name_el <- read_mat_element(x, pos)
  pos <- name_el$next_pos
  data_el <- read_mat_element(x, pos)

  dims <- read_from_raw(dims_el$data, integer(),
    n = dims_el$bytes / 4L,
    size = 4L
  )
  name <- rawToChar(name_el$data)

  vals <- switch(as.character(data_el$type),
    "9" = read_from_raw(data_el$data, numeric(), n = prod(dims), size = 8L),
    "4" = read_from_raw(data_el$data, integer(),
      n = prod(dims), size = 2L,
      signed = FALSE
    ),
    stop(sprintf(
      "Unsupported MAT numeric type %d for variable %s",
      data_el$type, name
    ))
  )

  list(name = name, value = matrix(vals, nrow = dims[1L], ncol = dims[2L]))
}

read_mat5_numeric <- function(path, variable) {
  bytes <- readBin(path, "raw", n = file.info(path)$size)
  pos <- 129L

  while (pos + 7L <= length(bytes)) {
    typ <- read_u32(bytes, pos)
    n_bytes <- read_u32(bytes, pos + 4L)
    payload <- bytes[(pos + 8L):(pos + 8L + n_bytes - 1L)]
    pos <- pos + 8L + n_bytes
    if (typ != 15L) next

    mat <- parse_mat_matrix(memDecompress(payload, type = "unknown"))
    if (identical(mat$name, variable)) {
      return(mat$value)
    }
  }

  stop("Variable not found in MAT file: ", variable, call. = FALSE)
}

ensure_spx_data <- function(path) {
  if (file.exists(path)) {
    return(invisible(path))
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  url <- paste0(
    "https://raw.githubusercontent.com/wilson-ye-chen/",
    "sp-mcmc/master/src/lib/data_spx.mat"
  )
  utils::download.file(url, path, mode = "wb", quiet = TRUE)
  invisible(path)
}


## ------------------------------------------------------------------
## IGARCH target
## ------------------------------------------------------------------

as_theta_matrix <- function(X) {
  X <- as.matrix(X)
  if (ncol(X) != 2L) stop("IGARCH theta must have two columns")
  X
}

igarch_valid_theta <- function(theta) {
  theta <- as_theta_matrix(theta)
  theta[, 1L] > 0 & theta[, 2L] > 0 & theta[, 2L] < 1
}

igarch_logp <- function(theta, y, h1) {
  theta <- as_theta_matrix(theta)
  n <- nrow(theta)
  theta1 <- theta[, 1L]
  theta2 <- theta[, 2L]
  valid <- is.finite(theta1) & is.finite(theta2) &
    theta1 > 0 & theta2 > 0 & theta2 < 1

  y_sq <- y^2
  h <- rep(h1, n)
  logp <- numeric(n)

  for (i in 2:length(y)) {
    h <- theta1 + theta2 * y_sq[i - 1L] + (1 - theta2) * h
    valid <- valid & is.finite(h) & h > 0
    idx <- which(valid)
    if (length(idx) > 0L) {
      logp[idx] <- logp[idx] - 0.5 * log(2 * pi) - 0.5 * log(h[idx]) -
        y_sq[i] / (2 * h[idx])
    }
  }

  logp[!valid] <- -Inf
  logp
}

igarch_score <- function(theta, y, h1) {
  theta <- as_theta_matrix(theta)
  n <- nrow(theta)
  theta1 <- theta[, 1L]
  theta2 <- theta[, 2L]
  valid <- is.finite(theta1) & is.finite(theta2) &
    theta1 > 0 & theta2 > 0 & theta2 < 1

  dh1 <- numeric(n)
  dh2 <- numeric(n)
  h <- rep(h1, n)
  score1 <- numeric(n)
  score2 <- numeric(n)
  y_sq <- y^2
  beta <- 1 - theta2

  for (i in 2:length(y)) {
    dh1 <- 1 + beta * dh1
    dh2 <- y_sq[i - 1L] - h + beta * dh2
    h <- theta1 + theta2 * y_sq[i - 1L] + beta * h
    valid <- valid & is.finite(h) & h > 0

    idx <- which(valid)
    if (length(idx) > 0L) {
      score_h <- -1 / (2 * h[idx]) + y_sq[i] / (2 * h[idx]^2)
      score1[idx] <- score1[idx] + score_h * dh1[idx]
      score2[idx] <- score2[idx] + score_h * dh2[idx]
    }
  }

  out <- cbind(score1, score2)
  out[!valid, ] <- 0
  out
}

## ------------------------------------------------------------------
## Markov chain helpers
## ------------------------------------------------------------------

logistic <- function(x, k) 1 / (1 + exp(-k * x))

nearest_spd <- function(S, eps = 1e-10) {
  S <- (S + t(S)) / 2
  eig <- eigen(S, symmetric = TRUE)
  vals <- pmax(eig$values, eps)
  out <- eig$vectors %*% (vals * t(eig$vectors))
  (out + t(out)) / 2
}

rmvn <- function(n, mu, Sigma) {
  mu <- as.numeric(mu)
  U <- chol(nearest_spd(Sigma))
  sweep(matrix(stats::rnorm(n * length(mu)), nrow = n) %*% U, 2L, mu, "+")
}

draw_adaptive_mc <- function(n, mu0, S0, lambda, X_curr) {
  d <- length(mu0)
  if (nrow(X_curr) == 0L) {
    return(rmvn(n, mu0, S0))
  }

  idx <- sample.int(nrow(X_curr), n, replace = TRUE)
  X_curr[idx, , drop = FALSE] + rmvn(n, rep(0, d), lambda * diag(d))
}

mala_reference <- function(log_p, score_function, x0, h, C, n_iter) {
  d <- length(x0)
  hh <- h^2
  S <- hh * nearest_spd(C)
  U <- chol(S)

  X <- matrix(0, n_iter, d)
  D <- matrix(0, n_iter, d)
  logp <- numeric(n_iter)
  accept <- integer(n_iter)
  X[1L, ] <- x0
  D[1L, ] <- score_function(matrix(x0, 1L, d))
  logp[1L] <- log_p(matrix(x0, 1L, d))

  for (i in 2:n_iter) {
    x_old <- X[i - 1L, ]
    d_old <- D[i - 1L, ]
    logp_old <- logp[i - 1L]

    mx <- x_old + (hh / 2) * as.numeric(d_old %*% C)
    y <- mx + as.numeric(stats::rnorm(d) %*% U)
    logp_y <- log_p(matrix(y, 1L, d))

    if (is.finite(logp_y)) {
      d_y <- score_function(matrix(y, 1L, d))
      my <- y + (hh / 2) * as.numeric(d_y %*% C)
      q_old_given_y <- sum(forwardsolve(t(U), x_old - my)^2)
      q_y_given_old <- sum(forwardsolve(t(U), y - mx)^2)
      log_accept <- (logp_y - logp_old) + 0.5 * (q_y_given_old - q_old_given_y)
    } else {
      d_y <- d_old
      log_accept <- -Inf
    }

    if (is.finite(log_accept) &&
      (log_accept >= 0 || log(stats::runif(1L)) < log_accept)) {
      X[i, ] <- y
      D[i, ] <- d_y
      logp[i] <- logp_y
      accept[i] <- 1L
    } else {
      X[i, ] <- x_old
      D[i, ] <- d_old
      logp[i] <- logp_old
    }
  }

  list(X = X, D = D, log_p = logp, accept = accept)
}

rwm_reference <- function(log_p, x0, S, n_iter) {
  d <- length(x0)
  U <- chol(nearest_spd(S))
  X <- matrix(0, n_iter, d)
  logp <- numeric(n_iter)
  accept <- integer(n_iter)
  X[1L, ] <- x0
  logp[1L] <- log_p(matrix(x0, 1L, d))

  for (i in 2:n_iter) {
    x_old <- X[i - 1L, ]
    y <- x_old + as.numeric(stats::rnorm(d) %*% U)
    logp_y <- log_p(matrix(y, 1L, d))
    log_accept <- logp_y - logp[i - 1L]

    if (is.finite(log_accept) &&
      (log_accept >= 0 || log(stats::runif(1L)) < log_accept)) {
      X[i, ] <- y
      logp[i] <- logp_y
      accept[i] <- 1L
    } else {
      X[i, ] <- x_old
      logp[i] <- logp[i - 1L]
    }
  }

  list(X = X, log_p = logp, accept = accept)
}

adaptive_mala_reference <- function(log_p, score_function, x0, h0, C0,
                                    alpha, epoch) {
  X <- vector("list", length(epoch))
  accept <- vector("list", length(epoch))
  h <- h0
  C <- C0

  fit <- mala_reference(log_p, score_function, x0, h, C0, epoch[1L])
  X[[1L]] <- fit$X
  accept[[1L]] <- fit$accept

  for (i in 2:length(epoch)) {
    C <- if (alpha[i] == 1) {
      C0
    } else {
      nearest_spd(alpha[i] * C0 + (1 - alpha[i]) * stats::cov(X[[i - 1L]]))
    }
    h <- h * exp(sum(accept[[i - 1L]]) / epoch[i - 1L] - 0.57)
    fit <- mala_reference(
      log_p, score_function, X[[i - 1L]][epoch[i - 1L], ],
      h, C, epoch[i]
    )
    X[[i]] <- fit$X
    accept[[i]] <- fit$accept
  }

  list(h = h, C = C, X = X, accept = accept)
}


## ------------------------------------------------------------------
## Stein method adapters
## ------------------------------------------------------------------

make_adaptive_mc_optimizer <- function(n_mc, mu0, S0, lambda, alpha,
                                       is_valid = NULL) {
  function(f, X_curr) {
    j <- nrow(X_curr) + 1L
    if (nrow(X_curr) == 0L || stats::runif(1L) <= alpha[j]) {
      X_mc <- rmvn(n_mc, mu0, S0)
    } else {
      X_mc <- draw_adaptive_mc(n_mc, mu0, S0, lambda, X_curr)
    }

    obj <- f(X_mc)
    valid <- if (is.null(is_valid)) rep(TRUE, nrow(X_mc)) else is_valid(X_mc)
    bad <- !valid | !is.finite(obj$f_vec)
    if (all(bad)) stop("All Monte Carlo optimizer candidates were invalid")
    obj$f_vec[bad] <- Inf

    i_min <- which.min(obj$f_vec)
    list(
      x_min = as.numeric(X_mc[i_min, ]),
      d_min = if (is.null(obj$D_new)) NA_real_ else as.numeric(obj$D_new[i_min, ]),
      f_min = as.numeric(obj$f_vec[i_min]),
      n_eval = n_mc
    )
  }
}

figure3_stein_points <- function(score_function, kernel, n_points, d, optimizer) {
  first_obj <- function(X_new, D_new = NULL) {
    if (is.null(D_new)) D_new <- validate_scores(score_function, X_new)
    list(f_vec = .k0_diag(kernel, X_new, D_new), D_new = D_new)
  }

  first <- optimizer(first_obj, matrix(0, 0, d))
  fit <- stein_points(
    score_function, kernel, n_points, d, optimizer,
    method = "greedy", x_init = first$x_min
  )
  fit$D[1L, ] <- as.numeric(first$d_min)
  fit$n_eval[1L] <- first$n_eval
  fit$cum_n_eval <- cumsum(fit$n_eval)
  fit
}

figure3_sp_mcmc <- function(log_p, score_function, kernel, d, x0, n_points,
                            m_seq, criterion, chain, S0, h = NULL,
                            lambda = 1, alpha) {
  chain <- match.arg(chain, c("mala", "grw", "rwm"))
  if (identical(chain, "rwm")) chain <- "grw"

  proposal_fn <- function(j, X_curr, h, Sigma, mcmc) {
    S <- if (nrow(X_curr) < 2L || alpha[j] == 1) {
      S0
    } else {
      A <- if (identical(mcmc, "grw")) lambda * stats::cov(X_curr) else stats::cov(X_curr)
      nearest_spd(alpha[j] * S0 + (1 - alpha[j]) * A)
    }
    list(h = h, Sigma = S)
  }
  transition_fn <- function(log_p, score_function, x0, h, Sigma, m_iter) {
    fit <- if (identical(chain, "mala")) {
      mala_reference(log_p, score_function, x0, h, Sigma, m_iter)
    } else {
      grwmetrop(log_p, x0, Sigma, m_iter)
    }
    fit$X <- fit$X[-1L, , drop = FALSE]
    if (!is.null(fit$D)) fit$D <- fit$D[-1L, , drop = FALSE]
    if (!is.null(fit$log_p)) fit$log_p <- fit$log_p[-1L]
    if (!is.null(fit$accept)) fit$accept <- fit$accept[-1L]
    fit
  }
  n_eval_fn <- function(chain, cand_X, cand_D, m_j, mcmc, j, X_curr) {
    if (identical(mcmc, "mala")) 2L * m_j else m_j + nrow(cand_X)
  }
  select_x1 <- function() {
    m_1 <- if (length(m_seq) == 1L) m_seq else m_seq[1L]
    proposal <- proposal_fn(1L, matrix(0, 0, d), h, NULL, chain)
    chain_fit <- transition_fn(
      log_p, score_function, x0, proposal$h,
      proposal$Sigma, m_1
    )
    path <- sp_mcmc_unique_path(sp_mcmc_validate_chain(chain_fit, d))
    D <- if (is.null(path$D)) validate_scores(score_function, path$X) else path$D
    i <- which.min(.k0_diag(kernel, path$X, D))
    list(
      x = as.numeric(path$X[i, ]),
      D = as.numeric(D[i, ]),
      n_eval = n_eval_fn(
        chain_fit, path$X, D, m_1, chain, 1L,
        matrix(0, 0, d)
      )
    )
  }

  first <- select_x1()

  fit <- sp_mcmc(
    score_function, log_p, kernel, n_points, d,
    mcmc = chain, criterion = criterion, m_seq = m_seq, h = h,
    x_init = first$x, transition_fn = transition_fn,
    proposal_fn = proposal_fn, n_eval_fn = n_eval_fn
  )
  fit$D[1L, ] <- first$D
  fit$n_eval[1L] <- first$n_eval
  fit$cum_n_eval <- cumsum(fit$n_eval)
  fit
}


## ------------------------------------------------------------------
## MED and SVGD paths
## ------------------------------------------------------------------

med_greedy_path <- function(log_p, optimizer, n_points, d) {
  X <- matrix(NA_real_, n_points, d)
  logp <- numeric(n_points)
  n_eval <- integer(n_points)

  first <- optimizer(
    function(X_new) list(f_vec = -log_p(X_new), D_new = NULL),
    matrix(0, 0, d)
  )
  X[1L, ] <- first$x_min
  logp[1L] <- log_p(matrix(first$x_min, 1L, d))
  n_eval[1L] <- first$n_eval

  for (j in 2:n_points) {
    obj_med <- function(X_new) {
      logp_new <- log_p(X_new)
      vals <- vapply(seq_len(nrow(X_new)), function(ii) {
        dst <- sqrt(rowSums(sweep(
          X[seq_len(j - 1L), , drop = FALSE],
          2L, X_new[ii, ], "-"
        )^2))
        min(logp_new[ii] + logp[seq_len(j - 1L)] +
          2 * d * log(pmax(dst, 1e-300)))
      }, numeric(1))
      list(f_vec = -vals, D_new = NULL)
    }
    res <- optimizer(obj_med, X[seq_len(j - 1L), , drop = FALSE])
    X[j, ] <- res$x_min
    logp[j] <- log_p(matrix(res$x_min, 1L, d))
    n_eval[j] <- res$n_eval + 1L
  }

  list(X = X, n_eval = n_eval, cum_n_eval = cumsum(n_eval))
}

svgd_path <- function(score_function, kernel, x0, n_iter, stepsize, alpha) {
  if (n_iter <= 1L) {
    trace <- list(as.matrix(x0))
  } else {
    fit <- svgd(kernel)$update(
      x0 = x0,
      lnprob = score_function,
      n_iter = n_iter - 1L,
      stepsize = stepsize,
      alpha = alpha,
      trace_iters = seq_len(n_iter - 1L)
    )
    trace <- c(list(as.matrix(x0)), unname(fit$trace))
  }
  n_eval <- c(0L, rep(nrow(x0), n_iter - 1L))
  list(
    X = trace[[n_iter]], trace = trace,
    n_eval = n_eval, cum_n_eval = cumsum(n_eval)
  )
}


## ------------------------------------------------------------------
## Energy-distance helpers
## ------------------------------------------------------------------

sum_dist_to_ref <- function(x, Y, block = 5000L) {
  total <- 0
  for (lo in seq(1L, nrow(Y), by = block)) {
    hi <- min(nrow(Y), lo + block - 1L)
    total <- total + sum(sqrt(rowSums(sweep(
      Y[lo:hi, , drop = FALSE],
      2L, x, "-"
    )^2)))
  }
  total
}

sum_all_dist <- function(Y, block = 1200L) {
  total <- 0
  for (lo in seq(1L, nrow(Y), by = block)) {
    hi <- min(nrow(Y), lo + block - 1L)
    A <- Y[lo:hi, , drop = FALSE]
    for (jlo in seq(1L, nrow(Y), by = block)) {
      jhi <- min(nrow(Y), jlo + block - 1L)
      B <- Y[jlo:jhi, , drop = FALSE]
      d2 <- outer(rowSums(A^2), rowSums(B^2), "+") - 2 * tcrossprod(A, B)
      total <- total + sum(sqrt(pmax(d2, 0)))
    }
  }
  total
}

energy_final <- function(X, Y, yy) {
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  m <- nrow(Y)
  n <- nrow(X)
  yx_sum <- sum(vapply(
    seq_len(n), function(i) sum_dist_to_ref(X[i, ], Y),
    numeric(1)
  ))
  xx_sum <- 0
  if (n >= 2L) {
    for (i in 2:n) {
      xx_sum <- xx_sum + 2 * sum(sqrt(rowSums(
        sweep(X[seq_len(i - 1L), , drop = FALSE], 2L, X[i, ], "-")^2
      )))
    }
  }
  pmax(2 * yx_sum / (m * n) - yy - xx_sum / (n^2), .Machine$double.eps)
}

energy_cumulative <- function(X, Y, yy) {
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  m <- nrow(Y)
  yx_sum <- 0
  xx_sum <- 0
  out <- numeric(nrow(X))

  for (n in seq_len(nrow(X))) {
    yx_sum <- yx_sum + sum_dist_to_ref(X[n, ], Y)
    if (n > 1L) {
      dst <- sqrt(rowSums(sweep(
        X[seq_len(n - 1L), , drop = FALSE],
        2L, X[n, ], "-"
      )^2))
      xx_sum <- xx_sum + 2 * sum(dst)
    }
    out[n] <- 2 * yx_sum / (m * n) - yy - xx_sum / (n^2)
  }

  pmax(out, .Machine$double.eps)
}

method_energy <- function(result, reference_chain, reference_constant) {
  if (!is.null(result$trace)) {
    vapply(result$trace, function(X) {
      energy_final(X, reference_chain, reference_constant)
    }, numeric(1))
  } else {
    energy_cumulative(result$X, reference_chain, reference_constant)
  }
}


## ------------------------------------------------------------------
## Run methods
## ------------------------------------------------------------------

run_method <- function(label, expr) {
  cat(sprintf("Running %-16s", label))
  t0 <- Sys.time()
  value <- force(expr)
  cat(sprintf(
    " [%.1fs]\n",
    as.numeric(difftime(Sys.time(), t0, units = "secs"))
  ))
  value
}

ensure_spx_data(SPX_DATA)
returns_all <- as.numeric(read_mat5_numeric(SPX_DATA, "r"))
returns <- returns_all[2501:4500]
initial_variance <- stats::var(returns)

log_p <- function(X) igarch_logp(X, returns, initial_variance)
score_fn <- function(X) igarch_score(X, returns, initial_variance)

cat("Building reference chain and preconditioner...\n")
alpha_adapt <- 1 - logistic(seq(-0.1, 1.9, length.out = length(ADAPT_EPOCH)), 20)
adapted <- adaptive_mala_reference(
  log_p, score_fn, X0, 0.004, diag(N_DIM),
  alpha_adapt, ADAPT_EPOCH
)
reference_full <- adapted$X[[length(adapted$X)]]
reference_chain <- reference_full[
  round(seq(1L, nrow(reference_full), length.out = N_REF_KEEP)), ,
  drop = FALSE
]

preconditioner <- stats::cov(reference_chain)
kernel <- stein_kernel(
  type = "imq", c = 1, beta = -0.5,
  precon = solve(nearest_spd(preconditioner))
)

results <- list()

results$MALA <- run_method("MALA", {
  n_iter <- N_POINTS * N_THIN
  fit <- mala_reference(log_p, score_fn, X0, 0.13, V_MCMC, n_iter)
  idx <- seq(1L, n_iter, by = N_THIN)[seq_len(N_POINTS)]
  list(
    X = fit$X[idx, , drop = FALSE],
    n_eval = rep(2L * N_THIN, N_POINTS)
  )
})

results$RWM <- run_method("RWM", {
  n_iter <- N_POINTS * N_THIN
  fit <- rwm_reference(log_p, X0, 0.06 * V_MCMC, n_iter)
  idx <- seq(1L, n_iter, by = N_THIN)[seq_len(N_POINTS)]
  list(
    X = fit$X[idx, , drop = FALSE],
    n_eval = rep(N_THIN, N_POINTS)
  )
})

results$SVGD <- run_method("SVGD", {
  x_init <- cbind(
    stats::runif(N_POINTS, LB[1], UB[1]),
    stats::runif(N_POINTS, LB[2], UB[2])
  )
  svgd_path(score_fn, kernel, x_init, N_SVGD_ITER, stepsize = 0.001, alpha = 0.9)
})

mc_alpha <- 1 - logistic(seq(-0.15, 1.85, length.out = N_POINTS), 50)
mc_alpha[1L] <- 1
mc_optimizer <- make_adaptive_mc_optimizer(
  n_mc = M_SPMCMC, mu0 = X0, S0 = 0.5 * V_MCMC,
  lambda = 5e-6, alpha = mc_alpha, is_valid = igarch_valid_theta
)

results$MED <- run_method("MED", {
  med_greedy_path(log_p, mc_optimizer, N_POINTS, N_DIM)
})

results$SP <- run_method("SP", {
  figure3_stein_points(score_fn, kernel, N_POINTS, N_DIM, mc_optimizer)
})

sp_alpha <- 1 - logistic(seq(-0.1, 1.9, length.out = N_POINTS), 100)
sp_alpha[1:10] <- 1

results$`SP-MALA LAST` <- run_method("SP-MALA LAST", {
  figure3_sp_mcmc(
    log_p, score_fn, kernel, N_DIM, X0, N_POINTS, M_SPMCMC, "last", "mala",
    S0 = V_MCMC / 0.5^2 * 0.02, h = 0.5, alpha = sp_alpha
  )
})

results$`SP-MALA INFL` <- run_method("SP-MALA INFL", {
  figure3_sp_mcmc(
    log_p, score_fn, kernel, N_DIM, X0, N_POINTS, M_SPMCMC, "infl", "mala",
    S0 = V_MCMC / 0.8^2 * 0.02, h = 0.8, alpha = sp_alpha
  )
})

results$`SP-RWM LAST` <- run_method("SP-RWM LAST", {
  figure3_sp_mcmc(
    log_p, score_fn, kernel, N_DIM, X0, N_POINTS, M_SPMCMC, "last", "rwm",
    S0 = 0.2 * V_MCMC, lambda = 2.38^2 / N_DIM, alpha = sp_alpha
  )
})

results$`SP-RWM INFL` <- run_method("SP-RWM INFL", {
  figure3_sp_mcmc(
    log_p, score_fn, kernel, N_DIM, X0, N_POINTS, M_SPMCMC, "infl", "rwm",
    S0 = 0.2 * V_MCMC, lambda = 2.38^2 / N_DIM, alpha = sp_alpha
  )
})


## ------------------------------------------------------------------
## Build plot data
## ------------------------------------------------------------------

cat("Computing energy-distance trajectories...\n")
reference_constant <- sum_all_dist(reference_chain) / nrow(reference_chain)^2

figure3 <- lapply(results, function(result) {
  if (is.null(result$cum_n_eval)) result$cum_n_eval <- cumsum(result$n_eval)
  data.frame(
    log_neval = log(pmax(result$cum_n_eval, 1)),
    log_ep = log(method_energy(result, reference_chain, reference_constant))
  )
})


## ------------------------------------------------------------------
## Plot Figure 3
## ------------------------------------------------------------------

labels <- names(figure3)
cols <- c(
  "#0072BD", "#7E2F8E", "#D95319", "#EDB120", "#77AC30",
  "#0072BD", "#4DBEEE", "#7E2F8E", "#EE82EE"
)
ltys <- c(3, 3, rep(1, 7))
lwds <- c(1.7, 1.7, rep(1.0, 7))

pdf(FIGURE3_PDF, width = 8, height = 5, bg = "white")
plot(NULL,
  xlim = c(1.6, 12.3),
  ylim = c(-11.8, -3.4),
  xlab = expression(log(n[eval])),
  ylab = expression(log(E[P]))
)
for (i in seq_along(labels)) {
  lines(figure3[[i]]$log_neval, figure3[[i]]$log_ep,
    col = cols[i], lty = ltys[i], lwd = lwds[i]
  )
}
legend("topright",
  legend = labels, col = cols, lty = ltys, lwd = lwds,
  bty = "n", cex = 0.78
)
box()
dev.off()

cat(sprintf("Saved Figure 3: %s\n", normalizePath(FIGURE3_PDF)))
