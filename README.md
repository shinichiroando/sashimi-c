# Semi-Analytical SubHalo Inference ModelIng for CDM (SASHIMI-C)
[![arXiv](https://img.shields.io/badge/arXiv-1803.07691%20-green.svg)](https://arxiv.org/abs/1803.07961)
[![arXiv](https://img.shields.io/badge/arXiv-1903.11427%20-green.svg)](https://arxiv.org/abs/1903.11427)

The codes allow to calculate various subhalo properties efficiently using semi-analytical models for cold dark matter (CDM). The results are well in agreement with those from numerical N-body simulations.

## Authors

- Shin'ichiro Ando
- Nagisa Hiroshima
- Ariane Dekker

Special thanks to Tomoaki Ishiyama, who provided data of cosmological N-body simulations that were used for calibration of model output.

Please send enquiries to Shin'ichiro Ando (s.ando@uva.nl). We have checked the codes work with python 3.9 but cannot guarantee for other versions of python. In any case, we cannot help with any technical issues not directly related to the content of SASHIMI (such as installation, sub-packages required, etc.)

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

## Examples

The file 'sashimi_c.py' contains all the variables and functions that are used to compute various subhalo properties. Please read 'sample.ipyb' for more extensive examples.

Here, as a minimal example, here is how you generate a semi-analytical catalog of subhalos:

```
from sashimi_c import *

sh = subhalo_properties()  # call the relevant class
M0 = 1.e12*sh.Msun         # input of host halo mass; here 10^{12} solar masses

ma200,z_acc,rs_acc,rhos_acc,m_z0,rs_z0,rhos_z0,ct_z0,weight,survive \
    = sh.subhalo_properties_calc(M0)
```

For inputs and outputs of this function, see its documentaion. For reference, it is:

```
-----
Input
-----
M0: Mass of the host halo defined as M_{200} (200 times critial density) at *z = 0*.
    Note that this is *not* the host mass at the given redshift! It can be obtained
    via Mzi(M0,redshift). If you want to give this parameter as the mass at the given
    redshift, then turn 'M0_at_redshift' parameter on (see below).
      
(Optional) redshift:       Redshift of interest. (default: 0)
(Optional) dz:             Grid of redshift of halo accretion. (default 0.1)
(Optional) zmax:           Maximum redshift to start the calculation of evolution from. (default: 7.)
(Optional) N_ma:           Number of logarithmic grid of subhalo mass at accretion defined as M_{200}).
                           (default: 500)
(Optional) sigmalogc:      rms scatter of concentration parameter defined for log_{10}(c).
                           (default: 0.128)
(Optional) N_herm:         Number of grid in Gauss-Hermite quadrature for integral over concentration.
                           (default: 5)
(Optional) logmamin:       Minimum value of subhalo mass at accretion defined as log_{10}(m_{min}/Msun)). 
                           (default: -6)
(Optional) logmamax:       Maximum value of subhalo mass at accretion defined as log_{10}(m_{max}/Msun).
                           If None, m_{max}=0.1*M0. (default: None)
(Optional) N_hermNa:       Number of grid in Gauss-Hermite quadrature for integral over host evoluation, 
                           used in Na_calc. (default: 200)
(Optional) Na_model:       Model number of EPS defined in Yang et al. (2011). (default: 3)
(Optional) ct_th:          Threshold value for c_t(=r_t/r_s) parameter, below which a subhalo is assumed to
                           be completely desrupted. Suggested values: 0.77 (default) or 0 (no desruption).
(Optional) profile_change: Whether we implement the evolution of subhalo density profile through tidal
                           mass loss. (default: True)
(Optional) M0_at_redshift: If True, M0 is regarded as the mass at a given redshift, instead of z=0.
        
------
Output
------
List of subhalos that are characterized by the following parameters.
ma200:    Mass m_{200} at accretion.
z_acc:    Redshift at accretion.
rs_acc:   Scale radius r_s at accretion.
rhos_acc: Characteristic density \rho_s at accretion.
m_z0:     Mass up to tidal truncation radius at a given redshift.
rs_z0:    Scale radius r_s at a given redshift.
rhos_z0:  Characteristic density \rho_s at a given redshift.
ct_z0:    Tidal truncation radius in units of r_s at a given redshift.
weight:   Effective number of subhalos that are characterized by the same set of the parameters above.
survive:  If that subhalo survive against tidal disruption or not.
```

These outputs are adopted further in various functions of 'subhalo_observables' class. See 'sample.ipynb' for details. They can be used in https://github.com/shinichiroando/dwarf_params to discuss density profiles of dwarf galaxies, as discussed in a related paper: https://arxiv.org/abs/2002.11956
