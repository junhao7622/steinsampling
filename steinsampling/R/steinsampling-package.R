#' @description
#' Score-based goodness-of-fit tests and Stein sampling tools: kernel Stein
#' discrepancy (KSD) and finite set Stein discrepancy (FSSD) tests, Stein
#' thinning, Stein Points, Stein Point Markov chain Monte Carlo (SP-MCMC),
#' Stein variational gradient descent (SVGD), and reusable Stein kernels.
#'
#' @details
#' Many target distributions are specified by a density known up to a
#' normalizing constant, making direct probability calculations and exact
#' simulation difficult. Two related tasks arise: assessing whether observed
#' data agree with the target distribution, and constructing a point set whose
#' empirical distribution approximates it. steinsampling addresses both tasks
#' using
#' \deqn{s_p(x)=\nabla \log p(x),}
#' where \eqn{s_p} is the target score.
#'
#' For goodness-of-fit testing, the observations are compared with a specified
#' target distribution through its score and a kernel. [ksd_u_test()] provides
#' a KSD test for independent observations. [ksd_v_test()] supports independent
#' observations and serially dependent observations under the corresponding
#' assumptions. [fssd_test()] instead evaluates Stein features at a finite set
#' of test locations, with random or optimized locations.
#'
#' For sampling and compression, the choice depends on how points are to be
#' obtained. [svgd()] moves an initial set of particles toward the target
#' distribution. [stein_points()] constructs points by numerical optimization,
#' whereas [sp_mcmc()] constructs points from short Markov chains. When samples
#' are already available, [stein_thinning()] selects a representative sequence,
#' allowing repeated selections.
#'
#' These methods share a score-and-kernel setup. [stein_kernel()] provides the
#' built-in kernels and links to custom kernel construction. Gaussian mixture
#' targets can be constructed with [gmm()], whose help page describes density
#' evaluation, simulation, and score calculation.
#'
#' The methods can be used as complete algorithms or assembled from individual
#' components.
"_PACKAGE"
