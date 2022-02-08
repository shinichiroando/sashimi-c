# Semi-Analytical SubHalo Inference ModelIng for CDM (SASHIMI-C)
[![arXiv](https://img.shields.io/badge/arXiv-1803.07691%20-green.svg)](https://arxiv.org/abs/1803.07961)
[![arXiv](https://img.shields.io/badge/arXiv-1903.11427%20-green.svg)](https://arxiv.org/abs/1903.11427)

The codes allow to calculate various subhalo properties efficiently using semi-analytical models for cold dark matter (CDM). The results are well in agreement with those from numerical N-body simulations.

## Authors

- Shin'ichiro Ando
- Nagisa Hiroshima
- Ariane Dekker

Please send enquiries to Shin'ichiro Ando (s.ando@uva.nl).

## What can we do with SASHIMI?

- SASHIMI provides a full catalog of dark matter subhalos in a host halo with arbitrary mass and redshift, which is calculated with semi-analytical models.
- Each subhalo in this catalog is characterized by its mass and density profile both at accretion and at the redshift of interest, accretion redshift, and effective number (or weight) corresponding to that particular subhalo.
- It can be used to quickly compute the subhalo mass function without making any assumptions such as power-law functional forms, etc. Only power law that we assume here is the one for primordial power spectrum predicted by inflation! Everything else is calculated theoretically.
- SASHIMI is not limited to numerical resoultion which is often the most crucial limiting factor for the numerical simulation. One can easily set the minumum halo mass to be a micro solar mass or even lighter!
- SASHIMI is not limited to Poisson shot noise that inevitably accompanies when one has to count subhalos like in the case of numerical simulations.
- One can calculates the annihilation boost factor.

## What are the future developments for SASHIMI?

- Extention to different dark matter models. The case of warm dark matter (WDM) has finished: https://github.com/shinichiroando/sashimi-w
- Including spatial information.
- Including intrinsic variance that accompanies the host halo evolution.
- Application to various primordial power spectra.
- Including baryonic effects.

## References

When you use the outcome of this package for your scientific output, please cite the following publications.

- N. Hiroshima, S. Ando, T. Ishiyama, Phys. Rev. D 97, 123002 (2018) [https://arxiv.org/abs/1803.07691]
- S. Ando, T. Ishiyama, N. Hiroshima, Galaxies 7, 68 (2019) [https://arxiv.org/abs/1903.11427]

The SASHIMI codes depend on results from various earlier papers. Listed below are some of the most essential papers. Please make sure to cite them too, if your focus is close to theirs!

- (Concentration-mass-redshift relation) https://arxiv.org/abs/1502.00391
- (Evolution of host halo mass) https://arxiv.org/abs/1409.5228
- (Extended Press-Schechter model) https://arxiv.org/abs/1104.1757
- (Power spectrum and rms mass density) https://arxiv.org/abs/1601.02624
