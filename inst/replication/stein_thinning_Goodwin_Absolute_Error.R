## Goodwin Stein thinning absolute-error example
##
## Compares standard thinning, Support Points, and Stein thinning variants on
## the Goodwin data, then saves the four-parameter error plot.

replication_dir <- normalizePath(
  if (dir.exists(file.path("inst", "replication"))) file.path("inst", "replication") else ".",
  mustWork = FALSE
)
suppressPackageStartupMessages(library(steinsampling))

suppressPackageStartupMessages({
  library(data.table)
  library(support)
})

set.seed(2022)

## ------------------------------------------------------------------
## Data and settings
## ------------------------------------------------------------------

data_dir <- file.path(replication_dir, "data", "stein_thinning", "Goodwin", "RW")
pdf_file <- file.path(replication_dir, "stein_thinning_Goodwin_Absolute_Error.pdf")

X <- as.matrix(fread(file.path(data_dir, "theta.csv"), header = FALSE))
S <- as.matrix(fread(file.path(data_dir, "grad.csv"), header = FALSE))

# Full replication: m_grid <- 1:200
m_grid <- 1:20
# Full replication: burn_low <- 70000L
burn_low <- 2000L
# Full replication: burn_high <- 820000L
burn_high <- 10000L
truth <- colMeans(X[(burn_high + 1L):nrow(X), , drop = FALSE])

## ------------------------------------------------------------------
## Baselines and Stein thinning helpers
## ------------------------------------------------------------------

standard_thinning_indices <- function(n, m, burn_in) {
  lag <- floor((n - burn_in) / m)
  as.integer(burn_in + seq_len(m) * lag)
}

standard_means <- function(burn_in) {
  t(vapply(m_grid, function(m) {
    idx <- standard_thinning_indices(nrow(X), m, burn_in)
    colMeans(X[idx, , drop = FALSE])
  }, numeric(ncol(X))))
}

# Silence Support Points progress output.
quiet_sp <- function(...) {
  output <- file(nullfile(), open = "wt")
  sink(output)
  sink(output, type = "message")
  on.exit(
    {
      sink(type = "message")
      sink()
      close(output)
    },
    add = TRUE
  )
  support::sp(...)
}

support_means <- t(vapply(m_grid, function(m) {
  out <- quiet_sp(
    n = m,
    p = ncol(X),
    dist.samp = X,
    # Full replication: num.subsamp = min(10000L, nrow(X)),
    num.subsamp = min(1000L, nrow(X)),
    rnd.flg = nrow(X) > 10000L,
    # Full replication: par.flg = TRUE
    par.flg = FALSE
  )
  colMeans(out$sp)
}, numeric(ncol(X))))

# Standardize samples and scores before Stein thinning.
standardize_samples_scores <- function(X, S) {
  loc <- colMeans(X)
  scl <- colMeans(abs(sweep(X, 2L, loc, "-")))
  list(
    X = sweep(X, 2L, scl, "/"),
    S = sweep(S, 2L, scl, "*")
  )
}

stein_means <- function(pre) {
  standardized <- standardize_samples_scores(X, S)
  idx <- stein_thinning(
    standardized$X,
    standardized$S,
    m = max(m_grid),
    pre = pre,
    kernel = "imq",
    c = 1,
    beta = -0.5,
    # Full replication: pre_subsample = 1000L,
    pre_subsample = 200L,
    pre_subsample_method = "even",
    sclmed_log = "pre_subsample"
  )
  cumulative <- apply(X[idx, , drop = FALSE], 2L, cumsum)
  t(vapply(m_grid, function(m) cumulative[m, ] / m, numeric(ncol(X))))
}

abs_error <- function(means) abs(sweep(means, 2L, truth, "-"))

## ------------------------------------------------------------------
## Compute errors
## ------------------------------------------------------------------

errors <- list(
  standard_high = abs_error(standard_means(burn_high)),
  standard_low = abs_error(standard_means(burn_low)),
  support = abs_error(support_means),
  stein_med = abs_error(stein_means("med")),
  stein_sclmed = abs_error(stein_means("sclmed")),
  stein_smpcov = abs_error(stein_means("smpcov"))
)

cols <- c(
  standard_high = "gray55",
  standard_low = "gray82",
  support = "black",
  stein_med = "blue",
  stein_sclmed = "green3",
  stein_smpcov = "red"
)
lwd <- setNames(c(1.1, 1.1, 1.25, 1.25, 1.25, 1.25), names(cols))
legend_labels <- c(
  standard_high = "Standard Thinning (high burn-in)",
  standard_low = "Standard Thinning (low burn-in)",
  support = "Support Points",
  stein_med = "Stein Thinning (med)",
  stein_sclmed = "Stein Thinning (sclmed)",
  stein_smpcov = "Stein Thinning (smpcov)"
)

## ------------------------------------------------------------------
## Plot Figure 4-style panels
## ------------------------------------------------------------------

plot_parameter <- function(param, show_xlab, show_ylab) {
  ylim <- c(1e-6, 1e-2)

  plot(
    NA,
    xlim = range(m_grid),
    ylim = ylim,
    log = "xy",
    axes = FALSE,
    xlab = if (show_xlab) expression(m) else "",
    ylab = if (show_ylab) "Absolute Error First Moment" else "",
    main = sprintf("Parameter %d", param)
  )
  box()
  axis(1L, labels = show_xlab)
  axis(2L, labels = show_ylab)
  for (name in names(errors)) {
    lines(m_grid, pmax(errors[[name]][, param], ylim[1L]), col = cols[[name]], lwd = lwd[[name]])
  }
}

pdf(pdf_file, width = 10.0, height = 6.2, onefile = FALSE)
layout(matrix(c(1, 2, 5, 3, 4, 5), nrow = 2L, byrow = TRUE), widths = c(1, 1, 1))
par(mar = c(2.8, 4.2, 2.0, 0.9), mgp = c(1.8, 0.55, 0), tcl = -0.25)
plot_parameter(1L, show_xlab = FALSE, show_ylab = TRUE)
plot_parameter(2L, show_xlab = FALSE, show_ylab = FALSE)
plot_parameter(3L, show_xlab = TRUE, show_ylab = TRUE)
plot_parameter(4L, show_xlab = TRUE, show_ylab = FALSE)
par(mar = c(0, 0, 0, 0))
plot.new()
legend("center", legend = legend_labels, col = cols, lwd = lwd, bty = "o", cex = 0.88)
invisible(dev.off())
