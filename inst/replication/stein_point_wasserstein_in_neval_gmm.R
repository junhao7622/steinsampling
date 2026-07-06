## Wasserstein comparison on a Gaussian mixture
##
## Compares Stein Greedy, Stein Herding, MED, and SVGD on a two-component GMM.
## The output plots Wasserstein error against model evaluations.

replication_dir <- normalizePath(
  if (dir.exists(file.path("inst", "replication"))) file.path("inst", "replication") else ".",
  mustWork = FALSE
)
suppressPackageStartupMessages(library(steinsampling))

if (!requireNamespace("transport", quietly = TRUE)) {
  stop("Package 'transport' is required for Wasserstein distances.", call. = FALSE)
}

options("transport-CPLEX_no_warn" = TRUE)

script_file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_file_arg) > 0L) {
  normalizePath(dirname(sub("^--file=", "", script_file_arg[1L])),
    mustWork = FALSE
  )
} else {
  getwd()
}

with_timer <- function(label, expr) {
  cat(sprintf("  %-34s", label))
  t0 <- Sys.time()
  value <- force(expr)
  cat(sprintf(
    " [%.1fs]\n",
    as.numeric(difftime(Sys.time(), t0, units = "secs"))
  ))
  value
}

## ------------------------------------------------------------------
## Runtime settings
## ------------------------------------------------------------------

n_points <- 100L
n_ref <- 800L
n_checkpoints <- 20L
n_mc <- 20L
n_nm <- 3L
n_grid0 <- 100L
svgd_iter <- 100L

panels_to_run <- c("greedy", "herding", "med", "svgd")
optimizers_to_run <- c("NM", "MC", "GS")
kernels_to_run <- c("k1", "k2", "k3")

## ------------------------------------------------------------------
## Target distribution
## ------------------------------------------------------------------
gmm_mod <- gmm(
  nComp = 2,
  mu = matrix(c(-1.5, 0, 1.5, 0), 2, 2),
  sigma = array(diag(2), c(2, 2, 2)),
  weights = c(0.5, 0.5),
  d = 2
)

p_density <- function(X) {
  X <- if (is.null(dim(X))) matrix(X, ncol = 2) else as.matrix(X)
  likelihoodgmm(gmm_mod, X)
}

log_p_density <- function(X) log(pmax(p_density(X), .Machine$double.eps))
score_p <- get_score_evaluator(gmm_mod)

gmm_hessian_evaluator <- function(model) {
  precision_cache <- build_precision_cache(model)
  function(X) {
    X <- if (is.null(dim(X))) matrix(X, nrow = 1L) else as.matrix(X)
    n <- nrow(X)
    d <- model$d
    k <- model$nComp
    post <- posteriorgmm(model = model, X = X)
    comp_scores <- array(0, dim = c(n, d, k))
    score <- matrix(0, n, d)

    for (component_idx in seq_len(k)) {
      mean_k <- get_component_mean(model, component_idx)
      diff <- sweep(X, 2L, mean_k, "-")
      sk <- -diff %*% precision_cache[[component_idx]]
      comp_scores[, , component_idx] <- sk
      score <- score + sk * post[, component_idx]
    }

    hess <- array(0, dim = c(n, d, d))
    for (i in seq_len(n)) {
      Hi <- matrix(0, d, d)
      for (component_idx in seq_len(k)) {
        sk <- matrix(comp_scores[i, , component_idx], ncol = 1L)
        Hi <- Hi + post[i, component_idx] *
          (-precision_cache[[component_idx]] + sk %*% t(sk))
      }
      si <- matrix(score[i, ], ncol = 1L)
      hess[i, , ] <- Hi - si %*% t(si)
    }
    hess
  }
}

hess_log_p <- gmm_hessian_evaluator(gmm_mod)

## ------------------------------------------------------------------
## Kernels and optimizers
## ------------------------------------------------------------------
kernel_settings <- list(
  greedy = list(
    k1 = list(alpha = 1.0, beta = -0.5),
    k2 = list(alpha = 4.0, beta = -1.0),
    k3 = list(alpha = 2.0, beta = -0.1)
  ),
  herding = list(
    k1 = list(alpha = 0.5, beta = -0.7),
    k2 = list(alpha = 0.5, beta = -1.0),
    k3 = list(alpha = 0.5, beta = -0.7)
  ),
  svgd = list(
    k1 = list(alpha = 0.5, beta = -0.9),
    k2 = list(alpha = 2.0, beta = -1.0),
    k3 = list(alpha = 0.1, beta = -0.7)
  )
)

make_stein_kernel <- function(name, settings) {
  if (identical(name, "k1")) {
    return(stein_kernel(
      type = "imq", c = sqrt(settings$alpha),
      beta = settings$beta
    ))
  }
  if (identical(name, "k2")) {
    return(stein_kernel_inverse_log(
      alpha = settings$alpha,
      beta = settings$beta
    ))
  }
  if (identical(name, "k3")) {
    return(stein_kernel_imq_score(
      alpha = settings$alpha,
      beta = settings$beta,
      hess_log_p = hess_log_p
    ))
  }
  stop("Unknown kernel: ", name, call. = FALSE)
}

make_svgd_k3_kernel <- function(settings) {
  eval_fn <- function(X, Y = NULL, precon = NULL) {
    X <- as.matrix(X)
    Y <- if (is.null(Y)) X else as.matrix(Y)
    SX <- score_p(X)
    SY <- if (identical(X, Y)) SX else score_p(Y)
    (settings$alpha + compute_cross_squared_distance(SX, SY))^settings$beta
  }

  grad_x_fn <- function(X, Y = NULL, precon = NULL) {
    X <- as.matrix(X)
    Y <- if (is.null(Y)) X else as.matrix(Y)
    SX <- score_p(X)
    SY <- if (identical(X, Y)) SX else score_p(Y)
    HX <- hess_log_p(X)
    base <- settings$alpha + compute_cross_squared_distance(SX, SY)
    coef <- 2 * settings$beta * base^(settings$beta - 1)
    arr <- array(0, dim = c(nrow(X), nrow(Y), ncol(X)))
    for (i in seq_len(nrow(X))) {
      for (j in seq_len(nrow(Y))) {
        delta <- matrix(SX[i, ] - SY[j, ], ncol = 1L)
        arr[i, j, ] <- as.numeric(coef[i, j] * HX[i, , ] %*% delta)
      }
    }
    arr
  }

  trace_fn <- function(X, Y = NULL, precon = NULL) {
    X <- as.matrix(X)
    Y <- if (is.null(Y)) X else as.matrix(Y)
    matrix(0, nrow(X), nrow(Y))
  }

  custom_stein_kernel(eval_fn, grad_x_fn, trace_fn)
}

make_svgd_kernel <- function(name, settings) {
  if (identical(name, "k3")) {
    return(make_svgd_k3_kernel(settings))
  }
  make_stein_kernel(name, settings)
}

## ------------------------------------------------------------------
## MED helper
## ------------------------------------------------------------------
lb <- c(-5, -5)
ub <- c(5, 5)
mu0 <- c(0, 0)
Sigma0 <- 25 * diag(2)
delay <- 20L

stein_optimizers <- list(
  MC = fmin_mc(lb, ub,
    n_mc = n_mc, mu0 = mu0, Sigma0 = Sigma0,
    sigsq = 1, delay = delay
  ),
  NM = fmin_nm(lb, ub,
    n_res = n_nm, mu0 = mu0, Sigma0 = Sigma0,
    sigsq = 1, delay = delay,
    control = list(reltol = 1e-3, maxit = 200)
  ),
  GS = fmin_grid(lb, ub, n0 = n_grid0, grow = TRUE)
)

make_scalar_optimizer <- function(kind) {
  kind <- toupper(kind)
  if (identical(kind, "MC")) {
    return(function(f, X_curr) {
      X_mc <- sample_proposal_box(
        n_mc, lb, ub, mu0, Sigma0, 1,
        X_curr, delay
      )
      res <- f(X_mc)
      i <- which.min(res$f_vec)
      list(
        x_min = X_mc[i, ], f_min = res$f_vec[i],
        aux_min = res$aux_vec[i], n_eval = n_mc
      )
    })
  }

  if (identical(kind, "GS")) {
    return(function(f, X_curr) {
      d <- length(lb)
      n_g <- rep(n_grid0 + as.integer(round(sqrt(nrow(X_curr) + 1))), d)
      grid <- as.matrix(do.call(expand.grid, lapply(seq_len(d), function(j) {
        seq(lb[j], ub[j], length.out = n_g[j])
      })))
      dimnames(grid) <- NULL
      res <- f(grid)
      i <- which.min(res$f_vec)
      list(
        x_min = grid[i, ], f_min = res$f_vec[i],
        aux_min = res$aux_vec[i], n_eval = nrow(grid)
      )
    })
  }

  if (identical(kind, "NM")) {
    span <- ub - lb
    to_x <- function(th) lb + span * sin(th)^2
    to_theta <- function(x) asin(sqrt(pmin(pmax((x - lb) / span, 0), 1)))

    return(function(f, X_curr) {
      X0 <- sample_proposal_box(
        n_nm, lb, ub, mu0, Sigma0, 1,
        X_curr, delay
      )
      best_x <- X0[1L, ]
      best_val <- Inf
      n_eval <- 0L
      f_th <- function(th) f(matrix(to_x(th), nrow = 1L))$f_vec[1L]

      for (i in seq_len(nrow(X0))) {
        opt <- stats::optim(to_theta(X0[i, ]), f_th,
          method = "Nelder-Mead",
          control = list(reltol = 1e-3, maxit = 200)
        )
        n_eval <- n_eval + as.integer(opt$counts["function"])
        if (opt$value < best_val) {
          best_val <- opt$value
          best_x <- to_x(opt$par)
        }
      }

      final <- f(matrix(best_x, nrow = 1L))
      list(
        x_min = as.numeric(best_x), f_min = best_val,
        aux_min = final$aux_vec[1L], n_eval = n_eval + 1L
      )
    })
  }

  stop("Unknown optimizer: ", kind, call. = FALSE)
}

run_med <- function(n_total, delta, optimizer, seed = NULL) {
  run <- function() {
    X <- matrix(NA_real_, n_total, 2L)
    y <- numeric(n_total)
    n_eval <- integer(n_total)

    for (j in seq_len(n_total)) {
      X_curr <- X[seq_len(j - 1L), , drop = FALSE]
      obj <- function(X_new) {
        X_new <- as.matrix(X_new)
        fp <- pmax(p_density(X_new), .Machine$double.eps)
        if (j == 1L) {
          y_new <- fp^(1 / 4)
          return(list(f_vec = 1 / y_new, aux_vec = y_new))
        }
        y_new <- fp^(delta / 4)
        inv <- vapply(seq_len(j - 1L), function(i) {
          d_ij <- pmax(
            sqrt(rowSums(sweep(X_new, 2L, X[i, ], "-")^2)),
            .Machine$double.eps
          )
          1 / (y[i] * d_ij^delta)
        }, numeric(nrow(X_new)))
        if (is.null(dim(inv))) inv <- matrix(inv, nrow = nrow(X_new))
        list(f_vec = rowSums(inv) / y_new, aux_vec = y_new)
      }

      res <- optimizer(obj, X_curr)
      X[j, ] <- res$x_min
      y[j] <- res$aux_min
      n_eval[j] <- res$n_eval
    }

    list(
      X = X, n_eval = n_eval, cum_n_eval = cumsum(n_eval),
      method = "MED", delta = delta
    )
  }
  if (is.null(seed)) run() else with_local_seed(seed, run())
}

## ------------------------------------------------------------------
## Wasserstein reference and curve helpers
## ------------------------------------------------------------------
set.seed(2026L)
X_ref <- rgmm(gmm_mod, n = n_ref)
ref_measure <- transport::wpp(X_ref, rep(1 / n_ref, n_ref))

wass_to_ref <- function(X) {
  X <- as.matrix(X)
  if (anyDuplicated(X)) {
    X <- X + matrix(stats::rnorm(length(X), sd = 1e-10), nrow(X), ncol(X))
  }
  transport::wasserstein(
    transport::wpp(X, rep(1 / nrow(X), nrow(X))),
    ref_measure,
    p = 1,
    method = "networkflow"
  )
}

checkpoint_index <- function(n) {
  unique(pmax(1L, pmin(n, as.integer(round(seq(1, n,
    length.out =
      min(n, n_checkpoints)
  ))))))
}

prefix_curve <- function(result, panel, label) {
  idx <- checkpoint_index(nrow(result$X))
  w <- vapply(
    idx, function(i) wass_to_ref(result$X[seq_len(i), , drop = FALSE]),
    numeric(1)
  )
  data.frame(
    panel = panel,
    label = label,
    x = log(pmax(result$cum_n_eval[idx], 1)),
    y = log(pmax(w, .Machine$double.eps)),
    stringsAsFactors = FALSE
  )
}

grid_initial_points <- function(n, lb, ub) {
  side <- ceiling(sqrt(n))
  grid <- as.matrix(expand.grid(
    seq(lb[1L], ub[1L], length.out = side),
    seq(lb[2L], ub[2L], length.out = side)
  ))
  dimnames(grid) <- NULL
  grid[seq_len(n), , drop = FALSE]
}

svgd_curve <- function(kernel_name) {
  settings <- kernel_settings$svgd[[kernel_name]]
  kernel <- make_svgd_kernel(kernel_name, settings)
  obj <- svgd(kernel = kernel)
  X0 <- grid_initial_points(n_points, lb, ub)
  iter_idx <- unique(pmax(1L, as.integer(round(seq(1, svgd_iter,
    length.out =
      min(
        svgd_iter,
        n_checkpoints
      )
  )))))
  fit <- update_svgd(obj,
    x0 = X0, lnprob = score_p, n_iter = svgd_iter,
    stepsize = 0.1, alpha = 0.9, kernel = kernel,
    trace_iters = iter_idx
  )
  w <- vapply(fit$trace, wass_to_ref, numeric(1))
  data.frame(
    panel = "svgd",
    label = kernel_name,
    x = log(iter_idx * n_points),
    y = log(pmax(w, .Machine$double.eps)),
    stringsAsFactors = FALSE
  )
}

## ------------------------------------------------------------------
## Run the method grid
## ------------------------------------------------------------------
cat("Figure 2 GMM Wasserstein reproduction\n")
cat(sprintf(
  "  n=%d, reference=%d, checkpoints=%d\n",
  n_points, n_ref, n_checkpoints
))
cat(sprintf(
  "  optimizers=%s, kernels=%s, panels=%s\n\n",
  paste(optimizers_to_run, collapse = ","),
  paste(kernels_to_run, collapse = ","),
  paste(panels_to_run, collapse = ",")
))

curves <- list()
curve_id <- 0L

add_curve <- function(df) {
  curve_id <<- curve_id + 1L
  curves[[curve_id]] <<- df
  invisible(NULL)
}

for (panel_name in c("greedy", "herding")) {
  if (!panel_name %in% panels_to_run) next
  method <- if (identical(panel_name, "greedy")) "greedy" else "herding"
  for (opt_name in optimizers_to_run) {
    if (!opt_name %in% names(stein_optimizers)) next
    for (kernel_name in kernels_to_run) {
      settings <- kernel_settings[[panel_name]][[kernel_name]]
      if (is.null(settings)) next
      label <- sprintf("%s %s", opt_name, kernel_name)
      result <- with_timer(sprintf("%s %s", panel_name, label), {
        stein_points(
          score_p,
          make_stein_kernel(kernel_name, settings),
          n_points = n_points,
          d = 2,
          optimizer = stein_optimizers[[opt_name]],
          method = method,
          log_p = log_p_density,
          seed = 1000L + curve_id
        )
      })
      add_curve(prefix_curve(result, panel_name, label))
    }
  }
}

if ("med" %in% panels_to_run) {
  for (opt_name in optimizers_to_run) {
    for (delta in c(4, 8, 16)) {
      label <- sprintf("%s delta=%d", opt_name, delta)
      result <- with_timer(sprintf("med %s", label), {
        run_med(n_points, delta, make_scalar_optimizer(opt_name),
          seed = 2000L + curve_id
        )
      })
      add_curve(prefix_curve(result, "med", label))
    }
  }
}

if ("svgd" %in% panels_to_run) {
  for (kernel_name in kernels_to_run) {
    label <- sprintf("svgd %s", kernel_name)
    result <- with_timer(label, svgd_curve(kernel_name))
    add_curve(result)
  }
}

if (length(curves) == 0L) {
  stop("No curves were generated; check the panel, optimizer, and kernel settings.",
    call. = FALSE
  )
}

curve_df <- do.call(rbind, curves)

## ------------------------------------------------------------------
## Plot the method panels
## ------------------------------------------------------------------
panel_titles <- c(
  greedy = "(a) Stein Points (Greedy)",
  herding = "(b) Stein Points (Herding)",
  med = "(c) MED",
  svgd = "(d) SVGD"
)

plot_panel <- function(df, title) {
  if (nrow(df) == 0L) {
    plot.new()
    title(title)
    return(invisible(NULL))
  }
  labels <- unique(df$label)
  cols <- grDevices::hcl.colors(length(labels), palette = "Dark 3")
  ltys <- rep(1:6, length.out = length(labels))
  xlim <- c(0, max(15, ceiling(max(curve_df$x, na.rm = TRUE))))
  ylim <- range(curve_df$y, finite = TRUE)
  ylim <- range(c(-1.2, 1.5, ylim), finite = TRUE)
  plot(NA,
    xlim = xlim, ylim = ylim,
    xlab = expression(log ~ n[eval]),
    ylab = expression(log ~ W[P]),
    main = title
  )
  grid(col = "grey88")
  for (i in seq_along(labels)) {
    rows <- df[df$label == labels[i], ]
    rows <- rows[order(rows$x), ]
    lines(rows$x, rows$y, col = cols[i], lwd = 1.4, lty = ltys[i])
    points(rows$x, rows$y, col = cols[i], pch = 16, cex = 0.45)
  }
  legend("topright",
    legend = labels, col = cols, lty = ltys, lwd = 1.2,
    bty = "n", cex = 0.58
  )
}

out_pdf <- file.path(script_dir, "stein_point_wasserstein_in_neval_gmm.pdf")
pdf(out_pdf, width = 9.5, height = 7.2, bg = "white")
op <- par(
  mfrow = c(2, 2), mar = c(4.0, 4.2, 2.6, 1.0),
  oma = c(0.5, 0.5, 1.8, 0.2), mgp = c(2.4, 0.7, 0)
)
for (panel_name in names(panel_titles)) {
  plot_panel(
    curve_df[curve_df$panel == panel_name, , drop = FALSE],
    panel_titles[[panel_name]]
  )
}
mtext("Figure 2 reproduction: Gaussian mixture Wasserstein vs model evaluations",
  outer = TRUE, side = 3, line = 0.4, cex = 1.0
)
par(op)
dev.off()

cat(sprintf("\nSaved figure: %s\n", normalizePath(out_pdf)))
