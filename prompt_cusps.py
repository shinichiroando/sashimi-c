"""
Prompt-cusp extension for SASHIMI-C.

This module contains everything specific to *prompt cusps* (relevant only for the
dark-matter annihilation signal). It is imported lazily by ``sashimi_c`` and is
NOT needed for standard SASHIMI-C usage:

  * ``build_ps_interpolators`` is called by ``halo_model.__init__`` only when
    ``prompt_cusps=True`` -- it replaces the Ludlow sigma(M) fit with sigma(M)
    computed from the CAMB linear matter power spectrum (free-streaming cut off).
  * ``prompt_cusps`` is the class used by
    ``subhalo_observables.annihilation_boost_factor_prompt_cusps``.

If ``prompt_cusps=False`` (the default) this module is never imported, so the
non-annihilation pipeline is completely unaffected.

Reference: Ando et al., arXiv:2601.19863.
"""
from contextlib import suppress
from functools import lru_cache
import os
from pathlib import Path
from shutil import copymode
from tempfile import TemporaryDirectory

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.special import erf

from sashimi_c import cosmology


def _file_key(filename):
    """Identify a local table, including edits and atomic replacements."""
    path = Path(os.fsdecode(filename)).resolve()
    stat = path.stat()
    return path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=8)
def _read_table(file_key, delimiter=None):
    table = np.loadtxt(file_key[0], delimiter=delimiter, ndmin=2)
    table.setflags(write=False)
    return table


def build_ps_interpolators(cosmo, k_fs, filter='Sharp-k', alpha=1.8,
                           datafile='./data/Planck2018_CAMB_extrap.dat'):
    """
    Build ``sigma(M)`` and ``dsigma^2/dM`` interpolators for the prompt-cusp mass
    function, from the CAMB linear matter power spectrum with a free-streaming
    cutoff ``exp[-(k/k_fs)^2]``.

    ``cosmo`` supplies ``OmegaM``, ``rhocrit0``, ``Mpc``, ``h``, ``Msun``.
    Returns ``(sigma_interp, dsdm_interp)`` -- cubic-spline ``interp1d`` objects
    over ``log10(M/Msun)``. Lifted out of ``sashimi_c.halo_model`` so the heavy
    power-spectrum code is only loaded when ``prompt_cusps=True``.
    """
    model = (cosmo.Mpc / cosmo.h, cosmo.Msun, 4. * np.pi * cosmo.OmegaM * cosmo.rhocrit0,
             k_fs, filter, alpha)
    if isinstance(datafile, (str, bytes, os.PathLike)) and os.path.isfile(datafile):
        samples = _cached_variance(_file_key(datafile), *model)
    else:
        # Preserve np.loadtxt's other inputs, including streams and iterables of lines.
        samples = _variance_samples(np.loadtxt(datafile, ndmin=2), *model)
    log_mass, sigma, derivative = samples
    options = dict(kind='cubic', bounds_error=False, fill_value=None)
    return interp1d(log_mass, sigma, **options), interp1d(log_mass, derivative, **options)


@lru_cache(maxsize=16)
def _cached_variance(file_key, length_unit, solar_mass, radius_denominator, k_fs, window, alpha):
    return _variance_samples(_read_table(file_key), length_unit, solar_mass, radius_denominator,
                             k_fs, window, alpha)


def _variance_samples(table, length_unit, solar_mass, radius_denominator, k_fs, window, alpha):
    """Keep the original quadratures but limit temporary arrays to a block of masses."""
    k, power = table.T
    k = k * length_unit ** -1
    power = power * length_unit ** 3 * np.exp(-(k / k_fs) ** 2)
    log_k = np.log(k)

    mass = np.logspace(-12, 17, 2000) * solar_mass
    radius = np.cbrt(3.0 * mass / radius_denominator)
    sigma_squared = np.empty_like(mass)
    derivative = np.empty_like(mass)
    normalization = 2.0 * np.pi ** 2

    if window == "Sharp-k":
        k_max = alpha / radius
        log_power = interp1d(log_k, np.log(power))
        for start in range(0, mass.size, 128):
            block = slice(start, start + 128)
            kk = np.logspace(np.log10(k[0]), np.log10(k_max[block]), 1000)
            log_kk = np.log(kk)
            integrand = kk ** 3 * np.exp(log_power(log_kk)) / normalization
            sigma_squared[block] = simpson(integrand, x=log_kk, axis=0)
        derivative = (
            -k_max ** 3 * np.exp(log_power(np.log(k_max))) / mass / (6.0 * np.pi ** 2)
        )
    elif window == "TopHat":
        k_cubed = k ** 3
        derivative_power = k_cubed * power / normalization
        for start in range(0, mass.size, 64):
            block = slice(start, start + 64)
            x = k * radius[block, None]
            sin_x, cos_x = np.sin(x), np.cos(x)
            x_squared = x ** 2
            weight = 3.0 * (sin_x / x - cos_x) / x_squared
            integrand = k_cubed * weight ** 2 * power / normalization
            sigma_squared[block] = simpson(integrand, x=log_k, axis=-1)
            dweight_dx = (
                9.0 * cos_x / x**3 - 9.0 * sin_x / x**4 + 3.0 * sin_x / x_squared
            )
            dx_dm = x / (3.0 * mass[block, None])
            integrand = derivative_power * 2.0 * weight * dweight_dx * dx_dm
            derivative[block] = simpson(integrand, x=log_k, axis=-1)
    else:
        raise ValueError(f"Unknown prompt-cusp variance window: {window!r}.")

    return np.log10(mass / solar_mass), np.sqrt(sigma_squared), derivative


def _ellipsoidal_collapse_factor(shape):
    """Lower root for accepted peak shapes q = e**2 - p*abs(p) in [0, 0.26)."""
    # For f - 1 - C*f**b the lower root is bracketed by 1 and b/(b-1)
    # up to the double root. The shape cut lies below that critical q.
    # 48 bisections give a bracket narrower than 2e-14.
    exponent = 0.615
    power = 2.0 * exponent
    coefficient = 0.47 * (5.0 * shape) ** exponent
    lower = np.ones_like(shape)
    upper = np.full_like(shape, power / (power - 1.0))
    for _ in range(48):
        middle = (lower + upper) * 0.5
        below_root = middle - 1.0 - coefficient * middle ** power <= 0.0
        lower = np.where(below_root, middle, lower)
        upper = np.where(below_root, upper, middle)
    return lower


def _bbks_peak_density(nu, x, R_s, gamma):
    """BBKS peak number density per unit peak height and curvature."""
    f_x = ((x**3 - 3.0 * x) / 2.0) * (
        erf(np.sqrt(5.0 / 2.0) * x) + erf(np.sqrt(5.0 / 8.0) * x)
    ) + np.sqrt(2.0 / (5.0 * np.pi)) * (
        (31.0 / 4.0 * x**2 + 8.0 / 5.0) * np.exp(-(5.0 / 8.0) * x**2)
        + (x**2 / 2.0 - 8.0 / 5.0) * np.exp(-(5.0 / 2.0) * x**2)
    )
    return (np.exp(-0.5 * nu**2) / ((2.0 * np.pi)**2 * R_s**3) * f_x
            * np.exp(-0.5 * (x - gamma * nu)**2 / (1.0 - gamma**2))
            / np.sqrt(2.0 * np.pi * (1.0 - gamma**2)))


@lru_cache(maxsize=16)
def _peak_number_density(R_s, gamma):
    """Absolute density of peaks on the original finite diagnostic grid."""
    nu = np.linspace(1.e-10, 10., 1000)
    x = np.linspace(1.e-10, 10., 1001)
    density = _bbks_peak_density(nu, x[:, None], R_s, gamma)
    return simpson(simpson(density, x=nu, axis=-1), x=x)


class prompt_cusps(cosmology):
    
    
    def __init__(self, k_fs):
        cosmology.__init__(self)
        
        self.GeV      = self.Msun/1.118e57
        self.m_chi    = 100.*self.GeV
        self.T_kd     = 0.03*self.GeV
        self.a_kd     = 5.3e-12
        self.k_fs     = k_fs
        
        
    def powerspectrum(self, k, z=0.):
        _z         = 31.
        _a         = 1./(1.+_z)
        a          = 1./(1.+z)
        g          = 0.901
        _k, _Delta2 = _read_table(_file_key('data/powerspectrum31.txt'), delimiter=',').T
        _k = _k * self.Mpc**-1
        _Delta2 = _Delta2 * (a/_a)**(2.*g)
        fint       = interp1d(np.log(_k),np.log(_Delta2),fill_value='extrapolate')
        Delta2     = np.exp(fint(np.log(k)))
        Delta2     = Delta2*np.exp(-(k/self.k_fs)**2)
        
        return Delta2

    
    def sigma_function(self, k=np.logspace(-1.,7.,num=1000)):
        #Spectral parameters
        k     = k/self.Mpc
        sigma = np.empty(3)
        power = self.powerspectrum(k=k)
        log_k = np.log(k)
        for j in np.arange(3):
            y        = power*k**(2*j)
            sigma_sq = simpson(y,x=log_k)
            sigma[j] = np.sqrt(sigma_sq)

        #Comoving characteristic correlation length
        R_s      = np.sqrt(3.)*sigma[1]/sigma[2]

        #Deviation from pure power-law behaviour
        gamma    = sigma[1]**2/(sigma[0]*sigma[2])

        return np.array(sigma), R_s, gamma
    

    def cusps_Monte_Carlo(self, length=10000, nu_max=10.0, x_max=10.0, N_nu=1001, N_x=1000,
                          rng_seed=None, verbose=False,
    ):
        """
        Monte Carlo sampler for prompt-cusp peak parameters:
        (nu, x) from BBKS peak density d^2n/(dnu dx),
        (e, p) from the DW/BBKS (Doroshkevich-type) conditional ellipticity/prolateness distribution,
        f_ec   solved from the collapse condition equation (as in your current code).

        DW-consistent core (up to normalization):
        f(e,p|nu) ∝ e (e^2 - p^2) exp[-(5/2) nu^2 (3 e^2 + p^2)]
        With u=p/e:
        f(e,u|nu) ∝ e^4 (1-u^2) exp[-alpha e^2],  alpha=(5/2) nu^2 (3+u^2)
        => f(u|nu) ∝ (1-u^2) (3+u^2)^(-5/2)
        => t=e^2 | (u,nu) ~ Gamma(shape=5/2, scale=1/alpha)

        Returns
        -------
        x_random, nu_random, e_random, p_random, fec_random : 1D arrays (length,)
        """

        rng = np.random.default_rng(rng_seed)

        # --- spectral parameters ---
        _, R_s, gamma = self.sigma_function()

        # --- sample (nu, x) on a grid via CDF inversion (stable; avoids p-sum issues) ---
        nu_grid = np.linspace(1e-12, float(nu_max), int(N_nu))
        x_grid = np.linspace(1e-12, float(x_max), int(N_x))

        # weights on mesh: shape (N_nu, N_x)
        w = _bbks_peak_density(nu_grid[:, None], x_grid, R_s, gamma)
        w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)

        wsum = float(np.sum(w))
        if not np.isfinite(wsum) or wsum <= 0.0:
            raise ValueError("cusps_Monte_Carlo: peak weight integral is zero/invalid. Check sigma_function() output.")

        w_flat = w.ravel()
        cdf = np.cumsum(w_flat, dtype=np.float64)
        cdf /= cdf[-1]  # normalize exactly

        u01 = rng.random(int(length))
        idx = np.searchsorted(cdf, u01, side="right")
        idx = np.clip(idx, 0, w_flat.size - 1)

        i_nu = idx // x_grid.size
        i_x = idx % x_grid.size

        nu_random = nu_grid[i_nu].astype(float)
        x_random = x_grid[i_x].astype(float)

        # -------------------------------------------------------------------------
        # --- DW-consistent sampling of (e, p) given nu via (u=p/e, t=e^2) ----------
        # Target: f(u|nu) ∝ (1-u^2) (3+u^2)^(-5/2) on u∈[-1,1]
        # Two-stage rejection:
        #   stage1: Uniform[-1,1] accepted with (1-u^2)  -> proposal ∝ (1-u^2)
        #   stage2: accept with (3/(3+u^2))^(5/2)       -> adds (3+u^2)^(-5/2)
        def draw_u(n):
            out = np.empty(n, dtype=float)
            filled = 0
            while filled < n:
                m = (n - filled) * 2 + 64
                uu = rng.uniform(-1.0, 1.0, size=m)

                # stage 1: accept with (1-u^2)
                a1 = rng.random(m) < (1.0 - uu**2)
                uu = uu[a1]
                if uu.size == 0:
                    continue

                # stage 2: accept with (3/(3+u^2))^(5/2)
                # This is always <= 1 since 3/(3+u^2) ∈ [3/4, 1]
                a2 = rng.random(uu.size) < (3.0 / (3.0 + uu**2)) ** 2.5
                uu = uu[a2]
                if uu.size == 0:
                    continue

                take = min(uu.size, n - filled)
                out[filled : filled + take] = uu[:take]
                filled += take
            return out

        u_random = draw_u(int(length))

        # Given u and nu:
        # f(e|u,nu) ∝ e^4 exp(-alpha e^2), alpha=(5/2) nu^2 (3+u^2)
        # Let t=e^2 => f(t|u,nu) ∝ t^(3/2) exp(-alpha t)  => Gamma(k=5/2, theta=1/alpha)
        alpha = 2.5 * (nu_random**2) * (3.0 + u_random**2)
        if np.any(~np.isfinite(alpha)) or np.any(alpha <= 0.0):
            raise ValueError("cusps_Monte_Carlo: alpha invalid (non-finite or non-positive).")

        t = rng.gamma(shape=2.5, scale=1.0 / alpha, size=int(length))
        e_random = np.sqrt(t)
        p_random = e_random * u_random

        # -------------------------------------------------------------------------
        shape = e_random**2 - p_random * np.abs(p_random)
        accepted = shape < 0.26
        fec_random = np.full(int(length), np.nan, dtype=float)
        if verbose:
            print(f"cusps_Monte_Carlo: solving f_ec for {accepted.sum()}/{length} samples")
        fec_random[accepted] = _ellipsoidal_collapse_factor(shape[accepted])

        return x_random, nu_random, e_random, p_random, fec_random
        
        
    def _load_or_generate_prompt_cusp_mc(
        self,
        length=10000,
        mc_cache_file="data/prompt_cusps/prompt_cusps_Monte_Carlo.txt",
        regenerate=False,
        rng_seed=None,
        verbose=False,
    ):
        """Load cached Monte Carlo samples for prompt-cusp peak parameters, or generate and cache them.

        The cached file contains 5 columns in the following order:
            x, nu, e, p, f_ec

        Parameters
        ----------
        length : int
            Number of Monte Carlo samples to generate if cache is missing or regeneration is requested.
        mc_cache_file : str
            Path to the cache file.
        regenerate : bool
            If True, ignore any existing cache and regenerate samples.
        rng_seed : int or None
            RNG seed passed to the generator for reproducibility.
        verbose : bool
            If True, print basic cache/generation info.

        Returns
        -------
        x_random, nu_random, e_random, p_random, fec_random : 1D arrays
        """

        cache_path = Path(os.fsdecode(mc_cache_file))
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if (not regenerate) and cache_path.exists():
            try:
                data = _read_table(_file_key(cache_path))
                if data.shape[1] != 5:
                    raise ValueError(f"Expected 5 columns (x,nu,e,p,f_ec) in {cache_path}, got {data.shape[1]}")
                x_random, nu_random, e_random, p_random, fec_random = data.T.copy()
                if verbose:
                    print(f"prompt_cusps: loaded Monte Carlo cache: {cache_path} (N={x_random.size})")
                return x_random, nu_random, e_random, p_random, fec_random
            except Exception as exc:
                if verbose:
                    print(f"prompt_cusps: failed to load cache {cache_path} ({exc}); regenerating.")

        # generate and cache
        x_random, nu_random, e_random, p_random, fec_random = self.cusps_Monte_Carlo(
            length=int(length),
            rng_seed=rng_seed,
            verbose=verbose,
        )

        header = (
            "x, nu, e, p, f_ec\n"
            "Generated by prompt_cusps.cusps_Monte_Carlo and cached for reproducibility." 
        )
        out = np.column_stack([x_random, nu_random, e_random, p_random, fec_random])
        # Publish the complete table so a concurrent reader sees one complete draw.
        destination = cache_path.resolve()
        with TemporaryDirectory(dir=destination.parent) as tmpdir:
            staged = Path(tmpdir) / cache_path.name
            np.savetxt(staged, out, header=header)
            with suppress(FileNotFoundError):
                copymode(destination, staged)
            staged.replace(destination)
        if verbose:
            print(f"prompt_cusps: saved Monte Carlo cache: {cache_path} (N={out.shape[0]})")

        return x_random, nu_random, e_random, p_random, fec_random

    def cusp_properties(
        self,
        f_surv=1.,
        z=0.,
        mc_length=10000,
        mc_cache_file="data/prompt_cusps/prompt_cusps_Monte_Carlo.txt",
        regenerate_mc=False,
        rng_seed=None,
        verbose=False,
        *,
        return_peak_diagnostics=False,
    ):
        """Return the collapsed fraction and the J factor of each collapsed cusp.

        If no peak collapses, return a zero fraction and a scalar zero J factor.
        With return_peak_diagnostics=True, append a dict with the universal
        peak and surviving cusp counts per unit dark matter mass (N_peaks_M,
        N_cusps_M), the corresponding annihilation factor per unit mass
        (J_cusps_M), and the late-time Delos & White fit (J_cusps_fit, Eq. 2.7).
        Counts have units of inverse mass and both J/M values have units of
        density, in the model's internal units. f_surv scales the surviving
        population. The host boost obtains its cusp count from subhalo weights.
        """
        x_random, nu_random, e_random, p_random, fec_random = self._load_or_generate_prompt_cusp_mc(
            length=mc_length,
            mc_cache_file=mc_cache_file,
            regenerate=regenerate_mc,
            rng_seed=rng_seed,
            verbose=verbose,
        )

        (sigma_0, _, sigma_2), R_s, gamma = self.sigma_function()

        #Scale factor
        a  = 1./(1.+z)

        #Growth index at small scales
        g  = 0.901

        #Critical density contrast for a spherical collapse
        delta_ec = 1.686

        delta      = nu_random*sigma_0
        dd_delta   = -x_random*sigma_2

        collapsed  = (delta*a**g > delta_ec*fec_random) & (e_random**2-p_random*np.abs(p_random) < 0.26)
        f_coll     = np.sum(collapsed)/len(collapsed)
        delta      = delta[collapsed]
        dd_delta   = dd_delta[collapsed]
        fec_random = fec_random[collapsed]

        #Cusp properties
        R       = np.sqrt(np.abs(delta/dd_delta))
        delta_a = delta*a**g
        a_coll  = (fec_random*delta_ec/delta_a)**(1./g)*a
        Aec     = 24.*(self.OmegaC*self.rhocrit0)*a_coll**-1.5*R**1.5
        rcusp   = 0.11*a_coll*R
        f_max   = (2.*np.pi)**(-3./2.)*(self.m_chi/self.T_kd)**(3./2.)*self.OmegaC*self.rhocrit0*self.a_kd**-3
        rcore   = (3.e-5*self.G**-3*f_max**-2*Aec**-1*self.c**6)**(1./4.5)
        # The cusp mass for the comparison in Ando et al., Sec. 2.4, is
        # Mcusp = 8.*np.pi/3.*Aec*rcusp**1.5.
        J_cusp = 4.*np.pi*Aec**2*(0.531+np.log(rcusp/rcore)) if delta.size else 0.

        if return_peak_diagnostics:
            rho_dm = self.OmegaC * self.rhocrit0
            n_peaks = _peak_number_density(R_s, gamma)
            N_cusps_M = f_surv * f_coll * n_peaks / rho_dm
            log_mT = np.log(self.m_chi * self.T_kd / self.GeV**2)
            diagnostics = {
                'N_peaks_M': n_peaks / rho_dm,
                'N_cusps_M': N_cusps_M,
                'J_cusps_M': np.mean(J_cusp) * N_cusps_M,
                'J_cusps_fit': 0.08 * f_surv * (log_mT + 36.)**5 * rho_dm,
            }
            return f_coll, J_cusp, diagnostics

        return f_coll, J_cusp
