# gmm_model.R

#' Returns a Gaussian Mixture Model
#'
#' @param nComp (scalar) : number of components
#' @param mu (d by k): mean of each component
#' @param sigma (d by d by k): covariance of each component
#' @param weights (1 by k) : mixing weight of each proportion (optional)
#' @param d : number of dimensions of vector (optional)
#'
#' @return model : A Gaussian Mixture Model generated from the given parameters
#'
#'
#' @export
#'
#' @examples
#' # Default 1-d gaussian mixture model
#' model <- gmm()
#'
#' # 1-d Gaussian mixture model with 3 components
#' model <- gmm(nComp = 3)
#'
#' # 3-d Gaussian mixture model with 3 components, with specified mu,sigma and weights
#' mu <- matrix(c(1,2,3,2,3,4,5,6,7),ncol=3)
#' sigma <- array(diag(3),c(3,3,3))
#' model <- gmm(nComp = 3, mu = mu, sigma=sigma, weights = c(0.2,0.4,0.4), d = 3)

gmm <- function(nComp=NULL, mu=NULL, sigma=NULL, weights=NULL, d=NULL){
  # NOTE: set.seed(0) intentionally removed.
  # Hardcoding set.seed() inside a function that is called from parallel
  # workers (foreach / doRNG) resets every worker's RNG state on every call,
  # which corrupts doRNG's reproducibility and causes all parallel trials to
  # silently fail or return garbage.  Callers are responsible for seeding.
  #
  # Generate a default Gaussian Mixture Model
  if(is.null(nComp)){
    d <- 1; k <- 5
    mu <- 10 * runif(k)
    mu <- mu - mean(mu)

    sigma <- array(diag(d),c(d,d,k));
  }else{
    k <- nComp
  }
  
  # Case when mean is not specified
  if(is.null(mu)){
    if(is.null(d)) d <- 1
    
    mu <- 10 * replicate(k,runif(d))
    mu <- mu - mean(mu)
    
  }
  
  # Case when sigma is not specified
  if(is.null(sigma)){
    if(is.null(d)){
      d <- dim(mu)[1]
    }
    
    sigma <- array(diag(d),c(d,d,k));
  }
  
  #If sigma is one-dimensional
  if(is.null(dim(sigma))){
    sigma <- array(sigma, c(1,1,k))
  }
  
  # Handle the cases for the weights
  if(is.null(weights)){
    weights = rep(1,k) / k
  }else if(sum(weights < 0) > 1){
    stop('Non-positive weights')
  }else if(sum(weights) != 1){
    weights <- weights / sum(weights)
  }
  
  model <- list("nComp"=k, "mu" = mu, "sigma" = sigma,
                "weights" = weights, "d"=d)
  
  return(model)
}

#' Generates dataset from Gaussian Mixture Model
#' @note Requires library mvtnorm
#'
#' @param model : Gaussian Mixture Model defined by gmm()
#' @param n : number of samples desired
#'
#' @return data (n by d): Random dataset generated from given the Gaussian Mixture Model
#' @export
#'
#' @examples
#' #Generate 100 samples from default gaussian mixture model
#' model <- gmm()
#' X <- rgmm(model)
#'
#' #Generate 300 samples from 3-d gaussian mixture model
#' model <- gmm(d=3)
#' X <- rgmm(model,n=300)


rgmm <- function(model = NULL,n=100){
  if(is.null(model)){
    stop('Supply GMM Model')
  }else{
    k <- model$nComp
    mu <- model$mu
    sigma <- model$sigma
    weights <- model$weights
    d <- model$d
    
    components <- sample(1:k,prob=weights,size=n,replace=TRUE)
    
    # 1 Dimensional case
    if(d == 1){
      stdev <- rep(0,k)
      for(i in 1:k){
        stdev[i] <- sqrt(sigma[,,i])
      }
      data <- rnorm(n=n,mean=mu[components],sd=stdev[components])
    }else{
      # Multidimensional Case
      if (!requireNamespace("mvtnorm", quietly = TRUE)) {
        stop("mvtnorm needed for this demo to work. Please install it.",
             call. = FALSE)
      }
      # Place each row at the slot corresponding to its component label so the row
      # order follows the random `components` vector and prefixes remain iid.
      data <- matrix(0, nrow = n, ncol = d)
      for (i in seq_len(k)) {
        idx <- which(components == i)
        if (length(idx) > 0L) {
          data[idx, ] <- mvtnorm::rmvnorm(length(idx),
                                          mean = mu[, i],
                                          sigma = sigma[, , i])
        }
      }
    }
    
    return(data)
    # hcum <- h <- hist(data,breaks=30,plot=FALSE)
    # hcum$counts <- cumsum(hcum$counts)
  }
}


#' Returns a perturbed model of given GMM
#'
#' @param model : The base Gaussian Mixture Model
#'
#' @return perturbedModel : Perturbed model with added noise to the supplied GMM
#'
#'
#' @export
#'
#' @examples
#' #Add noise to default 1-d gaussian mixture model
#' model <- gmm()
#' noisymodel <- perturbgmm(model)

perturbgmm <- function(model = NULL){
  if(is.null(model)){
    stop('Supply GMM Model')
  }
  # NOTE: set.seed(0) intentionally removed — same reason as gmm().
  k <- model$nComp
  d <- model$d
  noise <- replicate(k,rnorm(d))
  perturbed_mu <- model$mu + noise
  perturbed_model <- gmm(k,perturbed_mu,model$sigma, model$weights,d)
  
  return(perturbed_model)
}

#' Row-wise log-sum-exp.
#'
#' @param log_mat Numeric matrix of log values.
#'
#' @return Numeric vector of row-wise log-sum-exp values.
#' @examples
#' row_logsumexp(matrix(c(0, 1, 2, 3), nrow = 2))
#' @export
row_logsumexp <- function(log_mat) {
  row_max <- apply(log_mat, 1, max)
  shifted <- log_mat - row_max
  row_max + log(rowSums(exp(shifted)))
}

#' Extract a Gaussian mixture component mean
#'
#' @param model Gaussian mixture model returned by [gmm()].
#' @param component_idx Component index.
#'
#' @return Numeric mean vector for the requested component.
#' @export
get_component_mean <- function(model, component_idx) {
  if (model$d == 1) {
    if (is.null(dim(model$mu))) {
      return(as.numeric(model$mu[component_idx]))
    }
    return(as.numeric(model$mu[1, component_idx]))
  }
  as.numeric(model$mu[, component_idx])
}

#' @keywords internal
safe_precision_matrix <- function(sigma, ridge = 1e-6, cond_threshold = 1e10) {
  if (is.null(dim(sigma))) {
    val <- as.numeric(sigma)
    if (!is.finite(val) || val <= ridge) {
      val <- ridge
    }
    return(matrix(1 / val, nrow = 1, ncol = 1))
  }

  sigma <- as.matrix(sigma)
  d <- nrow(sigma)
  if (ncol(sigma) != d) {
    stop("Covariance matrix must be square")
  }

  sigma <- (sigma + t(sigma)) / 2
  diag_jitter <- ridge

  cond_num <- suppressWarnings(kappa(sigma, exact = FALSE))
  if (!is.finite(cond_num) || cond_num > cond_threshold) {
    sigma <- sigma + diag(diag_jitter, d)
  }

  chol_attempt <- try(chol(sigma), silent = TRUE)
  if (!inherits(chol_attempt, "try-error")) {
    return(chol2inv(chol_attempt))
  }

  sigma_reg <- sigma + diag(diag_jitter, d)
  chol_attempt_reg <- try(chol(sigma_reg), silent = TRUE)
  if (!inherits(chol_attempt_reg, "try-error")) {
    return(chol2inv(chol_attempt_reg))
  }

  if (requireNamespace("MASS", quietly = TRUE)) {
    return(MASS::ginv(sigma_reg))
  }

  qr.solve(sigma_reg, diag(d))
}

#' Build precision matrices for Gaussian mixture components
#'
#' Computes one inverse covariance matrix per component, with ridge
#' regularization for ill-conditioned covariance matrices.
#'
#' @param model Gaussian mixture model returned by [gmm()].
#' @param ridge Positive ridge value used when regularization is needed.
#' @param cond_threshold Condition-number threshold above which ridge
#'   regularization is applied.
#'
#' @return A list of precision matrices, one per mixture component.
#' @export
build_precision_cache <- function(model, ridge = 1e-6, cond_threshold = 1e10) {
  lapply(seq_len(model$nComp), function(component_idx) {
    safe_precision_matrix(
      model$sigma[, , component_idx],
      ridge = ridge,
      cond_threshold = cond_threshold
    )
  })
}

#' @keywords internal
gmm_log_component_densities <- function(model, X) {
  d <- model$d
  k <- model$nComp
  mu <- model$mu
  sigma <- model$sigma

  if (is.null(dim(X))) {
    if (d == 1) {
      x_mat <- matrix(as.numeric(X), ncol = 1)
    } else {
      x_mat <- matrix(as.numeric(X), nrow = 1)
    }
  } else {
    x_mat <- as.matrix(X)
  }

  n <- nrow(x_mat)
  if (ncol(x_mat) != d) {
    stop("X dimension does not match model dimension")
  }

  log_comp <- matrix(NA_real_, nrow = n, ncol = k)

  if (d == 1) {
    x_vec <- as.numeric(x_mat[, 1])
    for (i in seq_len(k)) {
      stdev <- sqrt(sigma[, , i])
      log_comp[, i] <- stats::dnorm(x_vec, mean = mu[i], sd = stdev, log = TRUE)
    }
  } else {
    if (!requireNamespace("mvtnorm", quietly = TRUE)) {
      stop("mvtnorm needed for this demo to work. Please install it.",
           call. = FALSE)
    }
    for (i in seq_len(k)) {
      log_comp[, i] <- mvtnorm::dmvnorm(x_mat, mean = mu[, i], sigma = sigma[, , i], log = TRUE)
    }
  }

  log_comp
}

#' Calculate posterior component probabilities for a GMM
#'
#' @param model Gaussian mixture model created by `gmm()`.
#' @param X Numeric vector or matrix of samples.
#'
#' @return Matrix of posterior component probabilities.
#' @examples
#' model <- gmm()
#' X <- rgmm(model, n = 5)
#' posteriorgmm(model, X)
#' @export
posteriorgmm <- function(model=NULL, X=NULL){
  if(is.null(model) || is.null(X)){
    stop('Supply Model and Data')
  }
  
  weights <- model$weights
  log_comp <- gmm_log_component_densities(model, X)
  log_w <- log(as.numeric(weights))
  log_joint <- sweep(log_comp, 2, log_w, "+")
  
  row_max <- apply(log_joint, 1, max, na.rm = TRUE)
  is_inf_row <- is.infinite(row_max) & row_max < 0
  post <- matrix(0, nrow=nrow(log_joint), ncol=ncol(log_joint))
  valid <- !is_inf_row
  
  if(any(valid)) {
    shifted <- log_joint[valid, , drop=FALSE] - row_max[valid]
    post[valid, ] <- exp(shifted) / rowSums(exp(shifted))
  }
  if(any(is_inf_row)) {
    post[is_inf_row, ] <- 1 / model$nComp
  }
  return(post)
}

#' Calculates the likelihood for a given dataset for a GMM
#'
#' @param model : The Gaussian Mixture Model
#' @param X (n by d): The dataset of interest,
#' where n is the number of samples and d is the dimension
#'
#' @return P (n by k) : The likelihood of each dataset belonging to each of the k component
#'
#'
#' @export
#'
#' @examples
#' # compute likelihood for a default 1-d gaussian mixture model
#' # and dataset generated from it
#' model <- gmm()
#' X <- rgmm(model)
#' p <- likelihoodgmm(model=model, X=X)
likelihoodgmm <- function(model=NULL, X=NULL){
  if(is.null(model) || is.null(X)){
    stop('Supply Model and Data')
  }else{
    weights <- model$weights

    log_comp <- gmm_log_component_densities(model, X)
    log_w <- log(as.numeric(weights))
    log_joint <- sweep(log_comp, 2, log_w, "+")
    sum_prob <- exp(row_logsumexp(log_joint))
  }
  
  return(sum_prob)
}


#' Score function for given GMM : calculates score function dlogp(x)/dx
#' for a given Gaussian Mixture Model
#'
#' @param model : The Gaussian Mixture Model
#' @param X (n by d): The dataset of interest,
#' where n is the number of samples and d is the dimension
#'
#' @return y : The score computed by the given function
#'
#'
#' @export
#'
#' @examples
#' # Compute score for a given gaussianmixture model and dataset
#' model <- gmm()
#' X <- rgmm(model)
#' score <- scorefunctiongmm(model=model, X=X)
scorefunctiongmm <- function(model=NULL, X=NULL){
  if(is.null(model) || is.null(X)){
    stop('Supply Model and Data')
  }else{
    P <- posteriorgmm(model, X)
    d <- model$d
    precision_cache <- build_precision_cache(model)
    if(d == 1){
      n <- length(X)
      x_mat <- matrix(as.numeric(X), ncol = 1)
    }else{
      n <- dim(X)[1]
      x_mat <- as.matrix(X)
    }
    
    score <- matrix(0, nrow = n, ncol = d)
    
    for(component_idx in 1:model$nComp){
      mean_k <- get_component_mean(model, component_idx)
      if (d == 1) {
        diff <- x_mat[, 1] - mean_k
        comp_score <- -matrix(diff, ncol = 1) %*% precision_cache[[component_idx]]
      } else {
        diff <- sweep(x_mat, 2, mean_k, "-")
        comp_score <- -diff %*% precision_cache[[component_idx]]
      }
      score <- score + comp_score * P[, component_idx]
      
    }
  }
  
  return(score)
}


#' Plots histogram for 1-d GMM given the dataset
#'
#' @param data (n by 1): The dataset of interest,
#' where n is the number of samples.
#' @param mu : True mean of the GMM (optional)
#'
#'
#' @export
#'
#' @examples
#' # Plot pdf histogram for a given dataset
#' model <- gmm()
#' X <- rgmm(model)
#' plotgmm(data=X)
#'
#' # Plot pdf histogram for a given dataset, with lines that indicate the mean
#' model <- gmm()
#' mu <- model$mu
#' X <- rgmm(model)
#' plotgmm(data=X, mu=mu)
plotgmm <- function(data, mu = NULL){
  hcum <- h <- hist(data,breaks=30,plot=FALSE)
  hcum$counts <- cumsum(hcum$counts)
  
  ##----- Plot only pdf----------------
  par(mar=c(1,1,1,1))
  plot(h, xlim = c(-20,20),col="grey")
  d <- density(data)
  lines(x = d$x, y = d$y  * length(data) * diff(h$breaks)[1], lwd = 2)
  if(!is.null(mu)){
    abline(v = mu,col = "royalblue",lwd = 2)
  }
}


#' Returns a score evaluator closure for a fixed GMM
#'
#' This helper exposes a generated GMM model through interfaces that require
#' `grad_log_prob` with signature `f(X)`.
#'
#' @param model : Gaussian Mixture Model defined by gmm()
#'
#' @return A function that takes only `X` and returns gradients of log-density.
#' If `X` is a vector (no `dim`) and `model$d == 1`, it is treated as a batch
#' of 1-D samples and the returned gradient is a vector of the same length.
#' If `X` is a vector and `model$d > 1`, it is treated as a single sample in
#' `d` dimensions and the returned gradient is a vector of length `d`.
#' If `X` is a matrix, the returned gradient is a matrix.
#'
#' @export
#'
#' @examples
#' model <- gmm()
#' grad_log_prob <- get_score_evaluator(model)
#' X <- rgmm(model)
#' G <- grad_log_prob(X)
get_score_evaluator <- function(model){
  precision_cache <- build_precision_cache(model)

  function(X){
    d <- model$d

    if(is.null(dim(X))){
      if(d == 1){
        X_mat <- matrix(X, ncol = 1)
      }else{
        X_mat <- matrix(X, nrow = 1)
      }
    } else {
      X_mat <- as.matrix(X)
    }

    post <- posteriorgmm(model = model, X = X_mat)
    n <- nrow(X_mat)
    grad <- matrix(0, nrow = n, ncol = d)

    for (component_idx in seq_len(model$nComp)) {
      mean_k <- get_component_mean(model, component_idx)
      if (d == 1) {
        diff <- X_mat[, 1] - mean_k
        comp_grad <- -matrix(diff, ncol = 1) %*% precision_cache[[component_idx]]
      } else {
        diff <- sweep(X_mat, 2, mean_k, "-")
        comp_grad <- -diff %*% precision_cache[[component_idx]]
      }
      grad <- grad + comp_grad * post[, component_idx]
    }

    if (is.null(dim(X))) {
      return(as.vector(grad))
    }

    grad
  }
}

# Snake_case aliases
gaussian_mixture_model <- gmm
sample_gmm <- rgmm
perturb_gmm <- perturbgmm
posterior_gmm <- posteriorgmm
likelihood_gmm <- likelihoodgmm
score_function_gmm <- scorefunctiongmm
plot_gmm <- plotgmm
