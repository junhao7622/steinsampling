## Stein Points comparison on a Gaussian mixture
##
## Compares six point-generation methods on a two-component GMM and plots the
## point sets at three evaluation budgets.

replication_dir <- normalizePath(
  if (dir.exists(file.path("inst", "replication"))) file.path("inst", "replication") else ".",
  mustWork = FALSE
)
suppressPackageStartupMessages(library(steinsampling))

set.seed(2026L)

## ------------------------------------------------------------------
## Target and optimizer setup
## ------------------------------------------------------------------

gmm_mod <- gmm(
  nComp = 2,
  mu = matrix(c(-1.5, 0, 1.5, 0), 2, 2),
  sigma = array(diag(2), c(2, 2, 2)),
  weights = c(0.5, 0.5), d = 2
)
p_density <- function(X) {
  likelihoodgmm(
    gmm_mod,
    if (is.null(dim(X))) matrix(X, ncol = 2) else X
  )
}
log_p_density <- function(X) log(pmax(p_density(X), .Machine$double.eps))
score_p <- get_score_evaluator(gmm_mod)

lb <- c(-5, -5)
ub <- c(5, 5)
mu0 <- c(0, 0)
Sigma0 <- 25 * diag(2)
n_mc <- 20L
delay <- 20L
opt <- fmin_mc(lb, ub,
  n_mc = n_mc, mu0 = mu0, Sigma0 = Sigma0,
  sigsq = 1, delay = delay
)


## ------------------------------------------------------------------
## MED helper
## ------------------------------------------------------------------

med_greedy <- function(n_total, delta, p_fun) {
  d <- 2
  X <- matrix(NA_real_, n_total, d)
  y <- numeric(n_total)
  draw <- function(j) {
    sample_proposal_box(
      n_mc, lb, ub, mu0, Sigma0, 1,
      X[seq_len(j - 1L), , drop = FALSE], delay
    )
  }

  X_mc <- draw(1L)
  fp <- pmax(p_fun(X_mc), .Machine$double.eps)
  i <- which.min(1 / fp^(1 / (2 * d)))
  X[1, ] <- X_mc[i, ]
  y[1] <- fp[i]^(1 / (2 * d))

  for (j in 2:n_total) {
    X_mc <- draw(j)
    fp <- pmax(p_fun(X_mc), .Machine$double.eps)
    yn <- fp^(delta / (2 * d))
    inv <- vapply(seq_len(j - 1L), function(i) {
      d_ij <- pmax(
        sqrt(rowSums(sweep(X_mc, 2, X[i, ], "-")^2)),
        .Machine$double.eps
      )
      1 / (y[i] * d_ij^delta)
    }, numeric(n_mc))
    i <- which.min((1 / yn) * rowSums(inv))
    X[j, ] <- X_mc[i, ]
    y[j] <- yn[i]
  }
  list(X = X, cum_n_eval = cumsum(rep(n_mc, n_total)))
}


## ------------------------------------------------------------------
## Snapshot settings
## ------------------------------------------------------------------

ref_neval <- c(6, 7.5, 9) # MATLAB ref = [6, 7.5, 9]

side <- 10L
g_box <- seq(lb[1], ub[1], length.out = side)
X0_box <- as.matrix(expand.grid(g_box, g_box))
dimnames(X0_box) <- NULL
nPart_box <- nrow(X0_box)
# Shuffle the grid before coordinate descent snapshots.
X0_box <- X0_box[with_local_seed(2026L, sample.int(nPart_box)), ]

pick_idx <- function(log_cum) {
  vapply(
    ref_neval,
    function(r) which.min(abs(log_cum - r)), integer(1)
  )
}


## ------------------------------------------------------------------
## Run all six methods
## ------------------------------------------------------------------

run_method <- function(label, fn) {
  cat(sprintf("  %-30s ", label))
  t0 <- Sys.time()
  out <- fn()
  cat(sprintf(
    "[%.1fs]\n",
    as.numeric(difftime(Sys.time(), t0, units = "secs"))
  ))
  out
}

cat("Generating point sequences ...\n")
results <- list()

results$mc <- run_method("(1) Monte Carlo iid", function() {
  X <- rgmm(gmm_mod, n = 8000L)
  ne <- log(seq_len(nrow(X)))
  list(mode = "extensible", X = X, neval = ne, idx = pick_idx(ne))
})

results$greedy <- run_method("(2) Stein Greedy [k2 alpha=4]", function() {
  res <- stein_points(score_p, stein_kernel_inverse_log(4),
    n_points = 500L, d = 2, optimizer = opt,
    method = "greedy", log_p = log_p_density, seed = 11L
  )
  ne <- log(res$cum_n_eval)
  list(mode = "extensible", X = res$X, neval = ne, idx = pick_idx(ne))
})

results$herd <- run_method("(3) Stein Herd. [k1 0.5,-0.7]", function() {
  res <- stein_points(score_p, stein_kernel(type = "imq", c = sqrt(0.5), beta = -0.7),
    n_points = 500L, d = 2, optimizer = opt,
    method = "herding", log_p = log_p_density
  )
  ne <- log(res$cum_n_eval)
  list(mode = "extensible", X = res$X, neval = ne, idx = pick_idx(ne))
})

results$med <- run_method("(4) MED [delta = 4]", function() {
  res <- med_greedy(500L, 4, p_density)
  ne <- log(res$cum_n_eval)
  list(mode = "extensible", X = res$X, neval = ne, idx = pick_idx(ne))
})

results$codes <- run_method("(5) Stein Co-Des. [k2 alpha=2]", function() {
  k_codes <- stein_kernel_inverse_log(2)
  it <- pmax(1L, as.integer(round((exp(ref_neval) - nPart_box) / n_mc)))
  cols <- lapply(it, function(n_it) {
    stein_codescent(X0_box, score_p, k_codes, n_it, opt, seed = 13L)
  })
  list(
    mode = "iterative",
    X_cols = lapply(cols, `[[`, "X"),
    neval = sapply(cols, function(r) log(tail(r$cum_n_eval, 1L))),
    iter = it
  )
})

results$svgd <- run_method("(6) SVGD [k1 0.5,-0.9]", function() {
  k_svgd <- stein_kernel(type = "imq", c = sqrt(0.5), beta = -0.9)
  obj <- svgd(kernel = k_svgd)
  it <- pmax(1L, as.integer(round(exp(ref_neval) / nPart_box)))
  cols <- lapply(it, function(n_it) {
    update_svgd(
      obj,
      x0 = X0_box, lnprob = score_p,
      n_iter = n_it, stepsize = 0.1, alpha = 0.9, kernel = k_svgd
    )
  })
  list(
    mode = "iterative",
    X_cols = cols, neval = log(it * nPart_box), iter = it
  )
})


## ------------------------------------------------------------------
## Report snapshots and plot
## ------------------------------------------------------------------
labels <- c(
  mc = "Monte Carlo", greedy = "Stein Greedy", herd = "Stein Herd.",
  med = "MED", codes = "Stein Co-Des.", svgd = "SVGD"
)

cat("\nColumn snapshots (achieved log n_eval):\n")
for (key in names(labels)) {
  r <- results[[key]]
  if (r$mode == "extensible") {
    keys <- sprintf("%5d", r$idx)
    vals <- r$neval[r$idx]
    cat(sprintf(
      "  %-15s idx  = %s   log n_eval = %s\n", labels[key],
      paste(keys, collapse = " "),
      paste(sprintf("%.2f", vals), collapse = " ")
    ))
  } else {
    cat(sprintf(
      "  %-15s iter = %s   log n_eval = %s\n", labels[key],
      paste(sprintf("%5d", r$iter), collapse = " "),
      paste(sprintf("%.2f", r$neval), collapse = " ")
    ))
  }
}

panel <- function(key, c) {
  r <- results[[key]]
  if (r$mode == "extensible") {
    list(pts = r$X[seq_len(r$idx[c]), , drop = FALSE], lne = r$neval[r$idx[c]])
  } else {
    list(pts = r$X_cols[[c]], lne = r$neval[c])
  }
}

out_pdf <- "stein_point_comparisonGMM.pdf"
pdf(out_pdf, width = 7.5, height = 12)
par(
  mfrow = c(6, 3), mar = c(0.7, 0.7, 1.4, 0.5),
  oma = c(3.5, 5.5, 1.5, 0.5)
)

g1 <- seq(lb[1], ub[1], length.out = 100)
g2 <- g1
Z <- matrix(p_density(as.matrix(expand.grid(g1, g2))), 100, 100)
zlev <- pretty(range(Z), 12)
zlev <- zlev[zlev > 0]

for (key in names(labels)) {
  for (c in 1:3) {
    p <- panel(key, c)
    pcex <- if (nrow(p$pts) <= 100) {
      0.6
    } else if (nrow(p$pts) <= 500) 0.45 else 0.3
    contour(g1, g2, Z,
      levels = zlev, drawlabels = FALSE,
      xlim = c(lb[1], ub[1]), ylim = c(lb[2], ub[2]),
      col = "grey50", lwd = 0.5,
      xlab = "", ylab = "", xaxt = "n", yaxt = "n",
      main = sprintf("log n_eval = %.2f  (n = %d)", p$lne, nrow(p$pts)),
      cex.main = 0.85
    )
    points(p$pts[, 1], p$pts[, 2], pch = 19, col = "red", cex = pcex)
    box(col = "grey30", lwd = 0.6)
    if (c == 1L) mtext(labels[key], side = 2, line = 1.2, cex = 0.95)
  }
}
mtext("Figure 1 reproduction: 2-component Gaussian mixture (Chen et al. 2018)",
  side = 3, outer = TRUE, line = 0.1, cex = 1.05
)
mtext("log n_eval increases left -> right",
  side = 1, outer = TRUE, line = 1.2, cex = 0.95
)
dev.off()

cat(sprintf("\nSaved figure: %s\n", normalizePath(out_pdf)))
