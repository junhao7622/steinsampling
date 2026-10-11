# steinsampling

`steinsampling` provides a unified R workflow for kernel Stein
goodness-of-fit tests, Stein sampling, point construction, and sample
compression. It includes KSD U- and V-statistic tests, FSSD tests, SVGD,
Stein Points, SP-MCMC, and Stein Thinning, together with reusable score,
kernel, calibration, optimization, and transition components.

## Installation

Install the released version from CRAN:

```r
install.packages("steinsampling")
```

## Paper

The accompanying paper is available on
[arXiv](https://arxiv.org/abs/2608.26450).

## Citation

If you use `steinsampling` in published work, please cite the accompanying
paper. The recommended citation is also available from R:

```r
citation("steinsampling")
```

```bibtex
@misc{gao2026kernelizedsteindiscrepancygoodnessoffit,
  title         = {Kernelized Stein Discrepancy for Goodness-of-Fit Tests and
                   Stein Sampling in R},
  author        = {Junhao Gao and Ery Arias-Castro},
  year          = {2026},
  eprint        = {2608.26450},
  archivePrefix = {arXiv},
  primaryClass  = {stat.CO},
  url           = {https://arxiv.org/abs/2608.26450}
}
```

After journal publication, the package citation will be updated to the
version of record and its journal DOI.

## Links

- Package source and issue tracker:
  <https://github.com/junhao7622/steinsampling>
- CRAN package:
  <https://CRAN.R-project.org/package=steinsampling>
