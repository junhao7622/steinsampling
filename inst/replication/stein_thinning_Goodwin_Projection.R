## Stein thinning projection example
##
## Builds a small Gaussian-mixture thinning figure and a Goodwin projection
## figure comparing standard thinning, Support Points, and Stein thinning.
##
## Data note:
## The package ships 20,000-row Goodwin/RW theta.csv and grad.csv excerpts for a
## CRAN-sized replication run. The complete Harvard Dataverse sources are:
## theta.csv: https://dataverse.harvard.edu/api/access/datafile/4807844
## grad.csv:  https://dataverse.harvard.edu/api/access/datafile/4807838

replication_dir <- normalizePath(
  if (dir.exists(file.path("inst", "replication"))) file.path("inst", "replication") else ".",
  mustWork = FALSE
)
suppressPackageStartupMessages(library(steinsampling))

require_or_stop <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Package '%s' is required but not installed.", pkg), call. = FALSE)
  }
}

required_packages <- c(
  "data.table",
  "parallel",
  "randtoolbox",
  "rngWELL",
  "Rcpp",
  "RcppArmadillo",
  "BH"
)
invisible(lapply(required_packages, require_or_stop))
require(data.table)
require(parallel)


## ------------------------------------------------------------------
## Gaussian mixture helpers
## ------------------------------------------------------------------

make_mixture <- function(means, covs, weights = NULL) {
  means <- as.matrix(means)
  k <- nrow(means)
  d <- ncol(means)
  if (is.null(weights)) weights <- rep(1 / k, k)
  covs <- lapply(covs, as.matrix)
  invs <- lapply(covs, solve)
  log_dets <- vapply(covs, function(S) as.numeric(determinant(S, logarithm = TRUE)$modulus), numeric(1))

  log_density <- function(X) {
    X <- as.matrix(X)
    comp <- matrix(NA_real_, nrow(X), k)
    for (j in seq_len(k)) {
      diff <- sweep(X, 2, means[j, ], "-")
      quad <- rowSums((diff %*% invs[[j]]) * diff)
      comp[, j] <- log(weights[j]) - 0.5 * (d * log(2 * pi) + log_dets[j] + quad)
    }
    max_comp <- apply(comp, 1, max)
    max_comp + log(rowSums(exp(sweep(comp, 1, max_comp, "-"))))
  }

  score <- function(X) {
    X <- as.matrix(X)
    log_comp <- matrix(NA_real_, nrow(X), k)
    comp_score <- array(NA_real_, dim = c(nrow(X), d, k))
    for (j in seq_len(k)) {
      diff <- sweep(X, 2, means[j, ], "-")
      quad <- rowSums((diff %*% invs[[j]]) * diff)
      log_comp[, j] <- log(weights[j]) - 0.5 * (d * log(2 * pi) + log_dets[j] + quad)
      comp_score[, , j] <- -diff %*% invs[[j]]
    }
    max_comp <- apply(log_comp, 1, max)
    resp <- exp(sweep(log_comp, 1, max_comp, "-"))
    resp <- resp / rowSums(resp)

    out <- matrix(0, nrow(X), d)
    for (j in seq_len(k)) {
      out <- out + comp_score[, , j] * resp[, j]
    }
    out
  }

  list(log_density = log_density, score = score)
}

rw_metropolis <- function(n, init, step_cov, log_density) {
  d <- length(init)
  chol_step <- chol(step_cov)
  X <- matrix(NA_real_, n, d)
  X[1, ] <- init
  log_curr <- log_density(matrix(init, 1, d))

  for (i in 2:n) {
    proposal <- X[i - 1, ] + as.numeric(rnorm(d) %*% chol_step)
    log_prop <- log_density(matrix(proposal, 1, d))
    if (log(runif(1)) < log_prop - log_curr) {
      X[i, ] <- proposal
      log_curr <- log_prop
    } else {
      X[i, ] <- X[i - 1, ]
    }
  }
  X
}


## ------------------------------------------------------------------
## Goodwin data and baseline helpers
## ------------------------------------------------------------------

standard_thinning_indices <- function(n, m, burn_in) {
  lag <- floor((n - burn_in) / m)
  if (lag < 1L) stop("burn_in leaves fewer than m states to thin.")
  as.integer(burn_in + seq_len(m) * lag)
}

support_points_selection <- function(X, m) {
  if (!requireNamespace("support", quietly = TRUE)) {
    warning(
      "Package 'support' is unavailable; skipping the Support Points baseline.",
      call. = FALSE
    )
    return(NULL)
  }
  utils::capture.output({
    out <- support::sp(
      n = m,
      p = ncol(X),
      dist.samp = X,
      # Full replication: num.subsamp = min(10000L, nrow(X)),
      num.subsamp = min(1000L, nrow(X)),
      rnd.flg = nrow(X) > 10000L,
      # Full replication: par.flg = TRUE
      par.flg = FALSE
    )
  })
  out$sp
}

dataverse_file_url <- function(file_id) {
  sprintf("https://dataverse.harvard.edu/api/access/datafile/%s", file_id)
}

read_numeric_csv_matrix <- function(path) {
  as.matrix(data.table::fread(path, header = FALSE))
}

load_goodwin_rw_output <- function(data_dir = file.path(replication_dir, "data", "stein_thinning", "Goodwin", "RW")) {
  theta_path <- file.path(data_dir, "theta.csv")
  grad_path <- file.path(data_dir, "grad.csv")
  missing_paths <- c(theta_path, grad_path)[!file.exists(c(theta_path, grad_path))]

  if (length(missing_paths) > 0L) {
    stop(
      paste(
        "Missing Goodwin/RW replication data from Riabiz et al. (2022).",
        "Download the two Dataverse files used for Figure 3 with:",
        sprintf("curl -L -o %s '%s'", theta_path, dataverse_file_url(4807844)),
        sprintf("curl -L -o %s '%s'", grad_path, dataverse_file_url(4807838)),
        sep = "\n"
      ),
      call. = FALSE
    )
  }

  X <- read_numeric_csv_matrix(theta_path)
  S <- read_numeric_csv_matrix(grad_path)
  if (!identical(dim(X), dim(S))) {
    stop("Goodwin/RW theta.csv and grad.csv must have identical dimensions.")
  }
  if (ncol(X) != 4L) stop("Goodwin/RW data should contain four parameters.")
  list(X = X, S = S)
}

standardize_samples_scores <- function(X, S) {
  loc <- colMeans(X)
  scl <- colMeans(abs(sweep(X, 2, loc, "-")))
  if (min(scl) == 0) stop("Too few unique samples to standardize.")
  list(
    X = sweep(X, 2, scl, "/"),
    S = sweep(S, 2, scl, "*")
  )
}

stein_thinning_imq_large <- function(X, S, m, pre,
                                     standardize = TRUE,
                                     allow_repeats = TRUE,
                                     pre_subsample_method = "first",
                                     sclmed_log = "m") {
  if (isTRUE(standardize)) {
    standardized <- standardize_samples_scores(X, S)
    X <- standardized$X
    S <- standardized$S
  }

  precon <- .build_thinning_precon(
    # Full replication: X, m, pre, 1000L, pre_subsample_method, sclmed_log
    X, m, pre, 200L, pre_subsample_method, sclmed_log
  )
  kernel <- stein_kernel(type = "imq", c = 1, beta = -0.5, precon = precon)
  M <- kernel_precon(kernel, ncol(X))
  tr_M <- kernel_precon_trace(M, ncol(X))
  beta <- kernel$beta
  c2 <- kernel$c^2

  objective <- 0.5 * (
    rowSums(S * S) * c2^beta -
      2 * beta * tr_M * c2^(beta - 1)
  )
  selected <- integer(m)

  for (j in seq_len(m)) {
    idx <- which.min(objective)
    selected[j] <- idx
    kp_row <- imq_kp_row(X, S, idx, M, tr_M, beta, c2)
    objective <- objective + kp_row
    if (!isTRUE(allow_repeats)) objective[selected[seq_len(j)]] <- Inf
    rm(kp_row)
    if (j %% 5L == 0L) gc(verbose = FALSE)
  }

  selected
}

imq_kp_row <- function(X, S, idx, M, tr_M, beta, c2) {
  x <- X[idx, ]
  sx <- S[idx, ]
  if (ncol(X) == 2L) {
    dx1 <- x[1] - X[, 1]
    dx2 <- x[2] - X[, 2]
    md1 <- dx1 * M[1, 1] + dx2 * M[2, 1]
    md2 <- dx1 * M[1, 2] + dx2 * M[2, 2]
    sq <- dx1 * md1 + dx2 * md2
    m2d1 <- md1 * M[1, 1] + md2 * M[2, 1]
    m2d2 <- md1 * M[1, 2] + md2 * M[2, 2]
    sq_m2 <- dx1 * m2d1 + dx2 * m2d2
    q <- (sx[1] - S[, 1]) * md1 + (sx[2] - S[, 2]) * md2
    score_dot <- S[, 1] * sx[1] + S[, 2] * sx[2]
    s <- c2 + sq
    return(
      score_dot * s^beta -
        2 * beta * s^(beta - 1) * q +
        (-4 * beta * (beta - 1) * sq_m2 * s^(beta - 2) -
          2 * beta * tr_M * s^(beta - 1))
    )
  }

  delta <- matrix(x, nrow(X), ncol(X), byrow = TRUE) - X
  M_delta <- delta %*% M
  sq <- rowSums(M_delta * delta)
  sq_m2 <- rowSums((M_delta %*% M) * delta)
  q <- rowSums((matrix(sx, nrow(X), ncol(X), byrow = TRUE) - S) * M_delta)
  s <- c2 + sq

  as.numeric(S %*% sx) * s^beta -
    2 * beta * s^(beta - 1) * q +
    (-4 * beta * (beta - 1) * sq_m2 * s^(beta - 2) -
      2 * beta * tr_M * s^(beta - 1))
}


## ------------------------------------------------------------------
## Plot helpers
## ------------------------------------------------------------------

plot_target_contours <- function(target, xlim, ylim, main = "") {
  xs <- seq(xlim[1], xlim[2], length.out = 180)
  ys <- seq(ylim[1], ylim[2], length.out = 180)
  grid <- as.matrix(expand.grid(xs, ys))
  z <- matrix(exp(target$log_density(grid)), length(xs), length(ys))
  image(xs, ys, z,
    col = hcl.colors(50, "Spectral", rev = TRUE),
    xlab = expression(x[1]), ylab = expression(x[2]), main = main,
    asp = 1, useRaster = TRUE
  )
  contour(xs, ys, z,
    add = TRUE, drawlabels = FALSE,
    col = adjustcolor("white", 0.35), lwd = 0.6
  )
}

plot_chain <- function(X, main = "", selected = NULL, show_lines = TRUE,
                       selected_col = "red", pch_selected = 19) {
  plot(X,
    type = "n", asp = 1, xlab = expression(x[1]), ylab = expression(x[2]),
    main = main
  )
  if (show_lines) lines(X, col = "gray55", lwd = 0.8)
  points(X, pch = 16, cex = 0.45, col = "gray70")
  if (!is.null(selected)) {
    points(X[selected, , drop = FALSE],
      pch = pch_selected, cex = 0.85,
      col = selected_col
    )
  }
}

panel_label <- function(label) {
  mtext(label, side = 1, line = 2.3, cex = 1.05)
}

plot_selection_panel <- function(X, idx, main, col = "red", bg_cex = 0.10,
                                 bg_col = "gray50") {
  plot(X,
    pch = 16, cex = bg_cex, col = bg_col,
    xlab = expression(x[1]), ylab = expression(x[2]), main = main
  )
  points(X[idx, , drop = FALSE], pch = 16, cex = 0.85, col = col)
}

plot_limits <- function(X, pad = 0.04) {
  xr <- range(X[, 1])
  yr <- range(X[, 2])
  list(
    xlim = xr + c(-1, 1) * diff(xr) * pad,
    ylim = yr + c(-1, 1) * diff(yr) * pad
  )
}

rescale_to_window <- function(z, from, to) {
  to[1] + (z - from[1]) / diff(from) * diff(to)
}

selection_xy <- function(X, selection) {
  if (is.matrix(selection) || is.data.frame(selection)) {
    selection <- as.matrix(selection)
    return(selection[, 1:2, drop = FALSE])
  }
  X[selection, 1:2, drop = FALSE]
}

draw_zoom_inset <- function(X, selections, colors,
                            zoom_xlim = c(-0.035, 0.045),
                            zoom_ylim = c(0.98, 1.16),
                            inset_xlim = c(-0.66, -0.28),
                            inset_ylim = c(-0.08, 0.36),
                            bg_col = "gray55") {
  in_zoom <- X[, 1] >= zoom_xlim[1] & X[, 1] <= zoom_xlim[2] &
    X[, 2] >= zoom_ylim[1] & X[, 2] <= zoom_ylim[2]
  inset_x <- rescale_to_window(X[in_zoom, 1], zoom_xlim, inset_xlim)
  inset_y <- rescale_to_window(X[in_zoom, 2], zoom_ylim, inset_ylim)

  rect(inset_xlim[1], inset_ylim[1], inset_xlim[2], inset_ylim[2],
    col = "white", border = NA
  )
  points(inset_x, inset_y, pch = ".", col = bg_col)

  for (name in names(selections)) {
    pts <- selection_xy(X, selections[[name]])
    keep <- pts[, 1] >= zoom_xlim[1] & pts[, 1] <= zoom_xlim[2] &
      pts[, 2] >= zoom_ylim[1] & pts[, 2] <= zoom_ylim[2]
    if (any(keep)) {
      points(
        rescale_to_window(pts[keep, 1], zoom_xlim, inset_xlim),
        rescale_to_window(pts[keep, 2], zoom_ylim, inset_ylim),
        pch = 16, cex = 0.75, col = colors[[name]]
      )
    }
  }

  rect(inset_xlim[1], inset_ylim[1], inset_xlim[2], inset_ylim[2], border = "gray35", lwd = 0.7)
  rect(zoom_xlim[1], zoom_ylim[1], zoom_xlim[2], zoom_ylim[2], border = "gray35", lwd = 0.7)
  segments(inset_xlim[1], inset_ylim[2], zoom_xlim[1], zoom_ylim[2], col = "gray35", lwd = 0.7)
  segments(inset_xlim[2], inset_ylim[1], zoom_xlim[1], zoom_ylim[1], col = "gray35", lwd = 0.7)
}

plot_goodwin_figure3_panel <- function(X, selections, colors, main, xlim, ylim,
                                       bg_col = "gray55") {
  plot(NA,
    xlim = xlim, ylim = ylim, xlab = expression(x[1]),
    ylab = expression(x[2]), main = main
  )
  points(X[, 1], X[, 2], pch = ".", col = bg_col)
  draw_zoom_inset(X, selections, colors, bg_col = bg_col)
  for (name in names(selections)) {
    pts <- selection_xy(X, selections[[name]])
    points(pts[, 1], pts[, 2], pch = 16, cex = 0.75, col = colors[[name]])
  }
  box()
}


## ------------------------------------------------------------------
## Figure 1: mixture thinning
## ------------------------------------------------------------------

run_figure1 <- function() {
  target <- make_mixture(
    means = rbind(c(-1.25, -1.1), c(1.1, 1.15)),
    covs = list(
      matrix(c(0.42, 0.18, 0.18, 0.34), 2),
      matrix(c(0.38, 0.16, 0.16, 0.42), 2)
    ),
    weights = c(0.5, 0.5)
  )
  X <- rw_metropolis(
    # Full replication: n = 500,
    n = 200,
    init = c(-3.2, 3.0),
    step_cov = matrix(c(0.32, 0.05, 0.05, 0.32), 2),
    log_density = target$log_density
  )
  S <- target$score(X)
  # Full replication: idx <- stein_thinning(X, S, m = 40, pre = "sclmed", kernel = "imq")
  idx <- stein_thinning(X, S, m = 20, pre = "sclmed", kernel = "imq")

  old <- par(
    mfrow = c(1, 3), mar = c(4.2, 4.2, 2.2, 1),
    oma = c(0, 0, 2, 0), pty = "s"
  )
  on.exit(par(old), add = TRUE)
  plot_target_contours(target, c(-3.2, 3.2), c(-3.2, 3.2), "")
  panel_label("(a)")
  plot_chain(X, "", show_lines = TRUE)
  panel_label("(b)")
  plot_chain(X, "", selected = idx, show_lines = FALSE)
  panel_label("(c)")
  mtext("Figure 1: Stein thinning on a two-component Gaussian mixture", outer = TRUE, cex = 1.15)
}


## ------------------------------------------------------------------
## Figure 3: Goodwin projection
## ------------------------------------------------------------------

run_figure3 <- function() {
  message("Figure 3: loading Goodwin/RW MCMC output from Riabiz et al. replication data...")
  dat <- load_goodwin_rw_output()
  X <- dat$X
  S <- dat$S
  m <- 20

  # Full replication: idx_standard_low <- standard_thinning_indices(nrow(X), m, burn_in = 70000)
  idx_standard_low <- standard_thinning_indices(nrow(X), m, burn_in = 2000)
  # Full replication: idx_standard_high <- standard_thinning_indices(nrow(X), m, burn_in = 820000)
  idx_standard_high <- standard_thinning_indices(nrow(X), m, burn_in = 10000)
  figure3_pre_subsample_method <- "even"
  figure3_sclmed_log <- "pre_subsample"
  message("Figure 3: computing Support Points baseline...")
  pts_support <- support_points_selection(X, m)
  message("Figure 3: Stein Thinning (med)...")
  idx_med <- stein_thinning_imq_large(
    X, S, m,
    pre = "med", allow_repeats = TRUE,
    pre_subsample_method = figure3_pre_subsample_method,
    sclmed_log = figure3_sclmed_log
  )
  message("Figure 3: Stein Thinning (sclmed)...")
  idx_sclmed <- stein_thinning_imq_large(
    X, S, m,
    pre = "sclmed", allow_repeats = TRUE,
    pre_subsample_method = figure3_pre_subsample_method,
    sclmed_log = figure3_sclmed_log
  )
  message("Figure 3: Stein Thinning (smpcov)...")
  idx_smpcov <- stein_thinning_imq_large(X, S, m, pre = "smpcov", allow_repeats = TRUE)

  old <- par(mar = c(4.0, 4.0, 2.4, 1), oma = c(0, 0, 2, 0))
  on.exit(par(old), add = TRUE)
  layout(matrix(c(1, 2, 3, 4, 5, 0), nrow = 3, byrow = TRUE))
  lims <- plot_limits(X)
  message("Figure 3: plotting panels...")

  plot_goodwin_figure3_panel(
    X,
    selections = list(high = idx_standard_high, low = idx_standard_low),
    colors = list(high = "red", low = "black"),
    main = "Standard Thinning",
    xlim = lims$xlim,
    ylim = lims$ylim
  )
  legend("topleft",
    bty = "n", cex = 0.85, pch = 16,
    col = c("red", "black"),
    # Full replication: legend = c(expression(hat(b) == 820000), expression(hat(b) == 70000))
    legend = c(expression(hat(b) == 10000), expression(hat(b) == 2000))
  )

  if (is.null(pts_support)) {
    plot.new()
    box()
    title("Support Points unavailable")
  } else {
    plot_goodwin_figure3_panel(
      X, list(sel = pts_support), list(sel = "red"),
      "Support Points", lims$xlim, lims$ylim
    )
  }
  plot_goodwin_figure3_panel(
    X, list(sel = idx_med), list(sel = "red"),
    "Stein Thinning (med)", lims$xlim, lims$ylim
  )
  plot_goodwin_figure3_panel(
    X, list(sel = idx_sclmed), list(sel = "red"),
    "Stein Thinning (sclmed)", lims$xlim, lims$ylim
  )
  plot_goodwin_figure3_panel(
    X, list(sel = idx_smpcov), list(sel = "red"),
    "Stein Thinning (smpcov)", lims$xlim, lims$ylim
  )
  # Full replication: mtext("Figure 3: first two coordinates of Goodwin/RW MCMC output, n = 2,000,000, m = 20", outer = TRUE, cex = 1.15)
  mtext("Figure 3: first two coordinates of Goodwin/RW MCMC output, n = 20,000, m = 20", outer = TRUE, cex = 1.15)
}


## ------------------------------------------------------------------
## Run figures
## ------------------------------------------------------------------

set.seed(20260617)
pdf(file.path(replication_dir, "stein_thinning_demo_figures.pdf"), width = 11, height = 7, onefile = TRUE)
run_figure1()
run_figure3()
dev.off()

message("Wrote stein_thinning_demo_figures.pdf")
