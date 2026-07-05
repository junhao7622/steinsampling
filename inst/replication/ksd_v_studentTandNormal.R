## KSD V test on Student-t chains
##
## Generates Markov chains from Student-t targets and checks them against a
## standard normal score. The output is a boxplot of p-values by degrees of
## freedom.

replication_dir <- normalizePath(
  if (dir.exists(file.path("inst", "replication"))) file.path("inst", "replication") else ".",
  mustWork = FALSE
)
suppressPackageStartupMessages(library(steinsampling))

library(ggplot2)

## ------------------------------------------------------------------
## MCMC helper
## ------------------------------------------------------------------

score_norm <- function(x) {
  return(-x)
}

rmcmc_t <- function(n, df, proposal_var = 0.5, init_val = 0) {
  samples <- numeric(n)
  current <- init_val

  target_log_pdf <- function(x) {
    if (is.infinite(df)) {
      return(dnorm(x, log = TRUE))
    }
    return(dt(x, df = df, log = TRUE))
  }

  for (i in 1:n) {
    proposal <- current + rnorm(1, mean = 0, sd = sqrt(proposal_var))
    prob <- exp(target_log_pdf(proposal) - target_log_pdf(current))
    if (runif(1) < prob) {
      current <- proposal
    }
    samples[i] <- current
  }
  return(samples)
}

## ------------------------------------------------------------------
## Run the p-value experiment
## ------------------------------------------------------------------

# Other figure settings used in the source experiment:
#   Figure 1: thinning = 1,  change_prob = 0.5
#   Figure 2: thinning = 1,  change_prob = 0.02
#   Figure 3: thinning = 20, change_prob = 0.1
dfs <- c(1, 5, 10, Inf)
n_samples <- 1400
n_reps <- 100
thinning <- 20
a_n_val <- 0.1

results <- list()

for (v in dfs) {
  p_values <- numeric(n_reps)
  cat(sprintf("Testing degrees of freedom (df) = %s at thinning level = %d and change_prob (a_n) = %.2f ... \n", as.character(v), thinning, a_n_val))

  current_val <- 0
  burn_in_samples <- rmcmc_t(5000, df = v, proposal_var = 0.5, init_val = current_val)
  current_val <- burn_in_samples[5000]

  for (r in 1:n_reps) {
    raw_samples <- rmcmc_t(n_samples * thinning, df = v, proposal_var = 0.5, init_val = current_val)
    current_val <- raw_samples[length(raw_samples)]

    thinned_samples <- matrix(raw_samples[seq(1, length(raw_samples), by = thinning)], ncol = 1)

    test_res <- ksd_v_test(
      X = thinned_samples,
      score_function = score_norm,
      boot_method = "markov",
      change_prob = a_n_val,
      nboot = 1000
    )

    p_values[r] <- test_res$p.value
  }

  results[[as.character(v)]] <- p_values
}

## ------------------------------------------------------------------
## Plot p-values
## ------------------------------------------------------------------

for (v in names(results)) {
  rejection_rate <- mean(results[[v]] < 0.05)
  cat(sprintf("degree of freedom %s: rejection = %.2f\n", v, rejection_rate))
}

df_plot <- stack(results)
colnames(df_plot) <- c("p_value", "df")

df_plot$df <- factor(df_plot$df, levels = c("1", "5", "10", "Inf"), labels = c("1.0", "5.0", "10.0", "inf"))

p <- ggplot(df_plot, aes(x = df, y = p_value)) +
  geom_boxplot(fill = "gray70", color = "black", outlier.shape = 18, outlier.color = "blue", outlier.size = 3) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
  theme_bw() +
  labs(
    x = "degrees of freedom",
    y = "p values"
  ) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 14)
  )

figure_path <- "ksd_vStudentTandNormal.pdf"
ggsave(figure_path, plot = p, width = 6, height = 4, dpi = 150, device = grDevices::pdf)
cat(sprintf("Saved figure: %s\n", normalizePath(figure_path, mustWork = FALSE)))
