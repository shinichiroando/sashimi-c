<p align="center">
  <img src="assets/logo.svg" alt="SASHIMI-C logo" width="440">
</p>

# Semi-Analytical SubHalo Inference ModelIng for CDM (SASHIMI-C)
[![arXiv](https://img.shields.io/badge/arXiv-1803.07691%20-green.svg)](https://arxiv.org/abs/1803.07961)
[![arXiv](https://img.shields.io/badge/arXiv-1903.11427%20-green.svg)](https://arxiv.org/abs/1903.11427)

The codes allow to calculate various subhalo properties efficiently using semi-analytical models for cold dark matter (CDM). The results are well in agreement with those from numerical N-body simulations.

## Authors

- Shin'ichiro Ando
- Nagisa Hiroshima
- Shunichi Horigome 
- Ariane Dekker

Special thanks to Tomoaki Ishiyama, who provided data of cosmological N-body simulations that were used for calibration of model output.

Please send enquiries to Shin'ichiro Ando (s.ando@uva.nl). We have checked that the codes work with python 3.10 but cannot guarantee for other versions of python. In any case, we cannot help with any technical issues not directly related to the content of SASHIMI (such as installation, sub-packages required, etc.)

## What can we do with SASHIMI?

- SASHIMI provides a full catalog of dark matter subhalos in a host halo with arbitrary mass and redshift, which is calculated with semi-analytical models.
- Each subhalo in this catalog is characterized by its mass and density profile both at accretion and at the redshift of interest, accretion redshift, and effective number (or weight) corresponding to that particular subhalo.
- It can be used to quickly compute the subhalo mass function without making any assumptions such as power-law functional forms, etc. Only power law that we assume here is the one for primordial power spectrum predicted by inflation! Everything else is calculated theoretically.
- SASHIMI is not limited to numerical resolution which is often the most crucial limiting factor for the numerical simulation. One can easily set the minimum halo mass to be a micro solar mass or even lighter!
- SASHIMI is not limited to Poisson shot noise that inevitably accompanies when one has to count subhalos like in the case of numerical simulations.
- One can calculate the annihilation boost factor.

## What are the future developments for SASHIMI?

- Extension to different dark matter models. The case of warm dark matter (WDM) and self-interacting dark matter (SIDM) has finished: https://github.com/shinichiroando/sashimi-w, https://github.com/shinichiroando/sashimi-si
- Including spatial information.
- Including intrinsic variance that accompanies the host halo evolution.
- Application to various primordial power spectra.
- Including baryonic effects.

## Prompt cusps (dark matter annihilation)

The smallest dark matter halos form *prompt cusps* with steep central density profiles that dominate the annihilation signal. This is relevant **only when discussing annihilation**, so all prompt-cusp code lives in a separate module, `prompt_cusps.py`, which is imported lazily and is never loaded for standard (non-annihilation) runs. To switch it on, pass `prompt_cusps=True` when constructing a class:

```python
from sashimi_c import *

sh = subhalo_observables(1.e12, prompt_cusps=True)   # M0/Msun; turns the prompt-cusp path on
Bsh, Bcusp_dressed, Bcusp_naked, luminosity_ratio, Ncusp_dressed, Ncusp_naked \
    = sh.annihilation_boost_factor_prompt_cusps(n=0)
```

With `prompt_cusps=True`, sigma(M) is computed from the CAMB linear matter power spectrum (with a free-streaming cutoff set by `k_fs_Mpc` / `filter` / `alpha`) instead of the Ludlow fit. The higher-order (`n>0`) boost tables are pre-computed with `boost_iteration_prompt_cusps.py`. See Ando et al. (arXiv:2601.19863).

## sigma(M, z) from CLASS (opt-in)

By default sigma(M, z) is the Ludlow et al. (2016) fit times the scale-independent growth factor `growthD(z)`. With `sigma_model='CLASS'` the EPS part of the model instead uses sigma(M, z) computed from a table of the linear CDM power spectrum P_cdm(k, z) made with CLASS; `class_table` (the path of the table) is then required. The module `class_sigma.py` is imported lazily, so the default path never loads it and its results are unchanged.

```python
from sashimi_c import *

sh = subhalo_properties(sigma_model='CLASS', class_table='data/class_pcdm.npz')   # Sharp-k, c = 2.5
ma200,z_acc,rs_acc,rhos_acc,m_z0,rs_z0,rhos_z0,ct_z0,weight,survive \
    = sh.subhalo_properties_calc(1.e12*sh.Msun, logmamin=-9)
```

Physics choices:

- **Field and threshold.** The CDM-only density field delta_c (CLASS `d_cdm`, synchronous gauge) with the collapse threshold 1.686. Below the baryon Jeans scale the baryons stay smooth, so delta_c grows more slowly (about a^0.9) than on large scales.
- **Scale-dependent growth.** P_cdm(k, z) is taken from CLASS at 18 redshift nodes (z = 0 to 500), so the growth is the true, scale-dependent one, including radiation, the baryon Jeans scale and Lambda. It is not sigma(M, 0) times D(z). Between nodes, ln sigma - ln growthD is interpolated with a bicubic spline in (log10 M, ln(1+z)).
- **High-k extension and cutoff.** Above the table's k_max (at least 1e4 Mpc^-1, beyond the Jeans transition), P is extended with the small-scale CDM shape k^(n_s-4) ln^2(B k), with ln B fitted at z = 500. The amplitude is matched at k_max for z >= 7 and at 1e4 Mpc^-1 for z < 7: CLASS's tanh reionisation makes c_b^2 < 0 at 6.8 < z < 8.2, which boosts delta_cdm at lower z, steeply with k (about 0.4% at 2e4 Mpc^-1). The free-streaming cutoff exp[-(k/k_fs)^2] uses `k_fs_Mpc` (default 1.06e6 Mpc^-1; `None` for no cutoff).
- **Window.** `ps_filter='Sharp-k'` (default) with mass assignment k_s = c/R_TH and `ps_c` = 2.5 (Benson et al. 2013). The options are `'Smooth-k'` (beta = 4.8, c = 3.3; Leo et al. 2018) and `'TopHat'`. Do not use the top-hat window below the cutoff: it leaves spurious haloes there.
- **EPS in the field at z_acc.** `Na_calc` (Yang et al. 2011, models 1-3) takes the random walk in the linear field at the accretion redshift, S = sigma_c^2(M, z_acc), where the collapse barrier is the constant 1.686. For Sharp-k the walk is Markovian, so Yang's Eqs. (12) and (14) hold despite the scale-dependent growth. The host barrier at z_Max is carried to z_acc with the growth at the host mass, 1.686 sigma_c(M_max, z_acc)/sigma_c(M_max, z_Max); this is exact when the growth is scale independent at the host scale (M_max >~ 1e8 Msun). With the Ludlow sigma this is algebraically the same as the default walk in the z = 0 field (the tests check it). The z = 0 field with the mass-dependent barrier B(M, z) = 1.686 sigma_c(M, 0)/sigma_c(M, z) (`deltac_func(z, M)`), solved as a moving barrier, is a different mapping of the walk between redshifts and is not exact for Sharp-k; compared with it, the z_acc-field accretion rate is up to 7% higher at m_a ~ 1e2-1e4 Msun and within 3% at m_a <= 1 Msun (M0 = 1e12 Msun, zmax = 7, central host-mass node).
- **Host history kept on Ludlow.** The host mass-accretion history (Correa et al. 2015: `Mzi`, `Mzzi`, `dMdz`) is calibrated with the top-hat sigma without a cutoff, so it keeps the Ludlow sigma and `growthD`. `conc200` does not depend on sigma.

Two parts keep their own sigma and growth and do not follow this option: the `prompt_cusps` class in `prompt_cusps.py` (cusp counts from its own Delta^2 at z = 31 times (a/a_31)^1.8) and `TidalStrippingSolver` (always the default `halo_model`). With `prompt_cusps=True` and `sigma_model='CLASS'` together, EPS uses the CLASS sigma, the host history uses the Ludlow sigma (not the CAMB sigma that `prompt_cusps=True` alone uses), and the pre-computed `n>0` prompt-cusp boost tables were made without the CLASS sigma.

**Comparing weights between the two modes.** `subhalo_properties_calc` normalises the weights as weight/sum(weight)*Na_total, a grid sum against a Simpson integral (unchanged from the default code). Where dN/dln m_a rises towards `logmamin` (Ludlow), this lowers all weights by a grid-dependent factor f < 1; in the CLASS mode the low-mass end is cut off and f is close to 1. For M0 = 1e12 Msun, f = 0.962 (Ludlow) and 0.980 (CLASS) at the defaults (N_ma = 500, logmamin = -6, zmax = 7), but 0.77 and 1.00 at N_ma = 81 and logmamin = -9. Compare the two modes with `Na_calc` directly, or on fine grids.

**Building the table.** The table is an `.npz` file with the keys `k_Mpc`, `z`, `pcdm`, `pcb`, `tcdm`, `sigma8`, `runtime_s`, `kmax` and `params`. It needs `classy` (the Python wrapper of CLASS):

```python
from class_sigma import make_class_table
make_class_table('data/class_pcdm.npz')   # Planck 2018, k_max = 2e4 Mpc^-1: about 35 min on one thread
```

Set the number of threads with `OMP_NUM_THREADS` in the shell (or pass `omp_threads`, which sets it during the call; it only takes effect if OpenMP has not started in the process yet). `make_class_table` requires k_max >= 1e4 Mpc^-1, and reading a table checks its h, Omega_m and Omega_b against sashimi's cosmology (a warning if they differ by more than 0.1%).

The tests (`OMP_NUM_THREADS=1 python -B tests/test_class_sigma.py`, or with `pytest`) check that the default results are bitwise unchanged, that the CLASS-mode `Na_calc` reduces to the default one for the Ludlow sigma, the error paths, and sigma(M, z) against an independent implementation. Some of them need files outside this repository, set by `SASHIMI_ORIG` (the original `sashimi_c.py`), `PS_REF_DIR` (the reference implementation) and `CLASS_TABLE`; a test whose inputs are missing is skipped.

## Versions

| Version | Description |
|---------|-------------|
| v1.0 | First public release. |
| v1.1 | Improved computational efficiency (perturbative evaluation of tidal stripping). |
| v1.2 | Prompt cusps for the annihilation signal (opt-in via `prompt_cusps=True`, isolated in `prompt_cusps.py`), plus general fixes. |

**Note (v1.2):** the default of `ct_th` (tidal-disruption threshold) changed from `0.77` to `0.0` (no disruption). This affects results for *all* users, not only prompt-cusp runs; pass `ct_th=0.77` to recover the previous behavior. v1.2 also includes a fix to `dDdz` and other minor corrections.

## References

When you use the outcome of this package for your scientific output, please cite the following publications.

- N. Hiroshima, S. Ando, T. Ishiyama, Phys. Rev. D 97, 123002 (2018) [https://arxiv.org/abs/1803.07691]
- S. Ando, T. Ishiyama, N. Hiroshima, Galaxies 7, 68 (2019) [https://arxiv.org/abs/1903.11427]

The SASHIMI codes depend on results from various earlier papers. Listed below are some of the most essential papers. Please make sure to cite them too, if your focus is close to theirs!

- (Concentration-mass-redshift relation) https://arxiv.org/abs/1502.00391
- (Evolution of host halo mass) https://arxiv.org/abs/1409.5228
- (Extended Press-Schechter model) https://arxiv.org/abs/1104.1757
- (Power spectrum and rms mass density) https://arxiv.org/abs/1601.02624
- (Prompt cusps and dark matter annihilation) https://arxiv.org/abs/2601.19863

## Examples

The file 'sashimi_c.py' contains all the variables and functions that are used to compute various subhalo properties. Please read 'sample.ipyb' for more extensive examples.

Here, as a minimal example, is how you generate a semi-analytical catalog of subhalos:

```
from sashimi_c import *

sh = subhalo_properties()  # call the relevant class
M0 = 1.e12*sh.Msun         # input of host halo mass; here 10^{12} solar masses

ma200,z_acc,rs_acc,rhos_acc,m_z0,rs_z0,rhos_z0,ct_z0,weight,survive \
    = sh.subhalo_properties_calc(M0)
```

For inputs and outputs of this function, see its documentation. For reference, it is:

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
(Optional) N_ma:           Number of logarithmic grid of subhalo mass at accretion defined as M_{200}.
                           (default: 500)
(Optional) sigmalogc:      rms scatter of concentration parameter defined for log_{10}(c).
                           (default: 0.128)
(Optional) N_herm:         Number of grid in Gauss-Hermite quadrature for integral over concentration.
                           (default: 5)
(Optional) logmamin:       Minimum value of subhalo mass at accretion defined as log_{10}(m_{min}/Msun). 
                           (default: -6)
(Optional) logmamax:       Maximum value of subhalo mass at accretion defined as log_{10}(m_{max}/Msun).
                           If None, m_{max}=0.1*M0. (default: None)
(Optional) N_hermNa:       Number of grid in Gauss-Hermite quadrature for integral over host evoluation, 
                           used in Na_calc. (default: 200)
(Optional) Na_model:       Model number of EPS defined in Yang et al. (2011). (default: 3)
(Optional) ct_th:          Threshold value for c_t(=r_t/r_s) parameter, below which a subhalo is assumed to
                           be completely desrupted. Suggested values: 0.77 (disruption) or 0.0
                           (no desruption; default since v1.2).
(Optional) profile_change: Whether we implement the evolution of subhalo density profile through tidal
                           mass loss. (default: True)
(Optional) M0_at_redshift: If True, M0 is regarded as the mass at a given redshift, instead of z=0.
(Optional) method:         Method to calculate the subhalo mass stripping. (default: "pert2_shanks")
                           - "odeint" : use odeint to solve the differential equation.
                           - "pert0" : use perturbative method with zeroth-order correction.
                           - "pert1" : use perturbative method with first-order correction.
                           - "pert2" : use perturbative method with second-order correction.
                           - "pert2_shanks" : use perturbative method with second-order correction 
                             and Shanks transformation.
                           - "pert3" : use perturbative method with third-order correction.
(Optional) kwargs:         Additional arguments for the odeint function.
        
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
