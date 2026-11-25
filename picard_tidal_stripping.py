"""Picard integration of the native SASHIMI tidal stripping equation.

Adapted from gomeshun/sashimi-c at 2ffefc8f02ff1db2f634aae73c1b2a394914a98e.
The caller supplies the redshift-dependent host mass, stripping rate and exponent.
"""

# MIT License
#
# Copyright (c) 2026 Shin'ichiro Ando
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from __future__ import annotations

import hashlib
import time
import warnings

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline


class PicardFallbackWarning(UserWarning):
    """Report replacement of a Picard estimate by direct log-mass integration.

    Notes
    -----
    The warning records why the request exceeded the numerical domain or
    iteration tolerance. The solver also records the affected mass count.
    """


def _real(value, name):
    """Read mass or redshift coordinates as finite real numbers.

    Parameters
    ----------
    value : array_like
        Scalar or array in the caller's physical unit.
    name : str
        Coordinate name used to identify invalid input.

    Returns
    -------
    numpy.ndarray
        Floating values with the input shape and unit.

    Raises
    ------
    ValueError
        If any element is non-real or non-finite.
    """
    a = np.asarray(value)
    if a.dtype.kind not in "iuf" or np.any(~np.isfinite(a)):
        raise ValueError(f"{name} must contain finite real numbers.")
    return a.astype(float, copy=False)


def _integer(value, name, minimum):
    """Validate a grid size or iteration count.

    Parameters
    ----------
    value : scalar
        Requested dimensionless count.
    name : str
        Numerical control named in an error.
    minimum : int
        Smallest count supported by the operation.

    Returns
    -------
    int
        The validated count.

    Raises
    ------
    ValueError
        If the input is fractional, non-finite, boolean or below the minimum.
    """
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
        or int(value) != value
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def physics_key(solver):
    """Identify the host and rate inputs that determine a stripped mass.

    Parameters
    ----------
    solver : object
        Native host model providing Mzvir, Phi and zetaMz in consistent mass units.

    Returns
    -------
    tuple
        Immutable key for reusing results on this solver.

    Notes
    -----
    Scalar settings, numerical arrays and rate callables enter the key.
    A provider with additional internal state can supply picard_physics_key().
    """
    result = []
    for name, value in sorted(vars(solver).items()):
        if name.startswith(("_picard", "_eps", "eps_")):
            continue
        if isinstance(value, (float, int, str, bool, np.number)):
            result.append((name, repr(value)))
        elif isinstance(value, np.ndarray) and value.dtype.kind in "iuf":
            result.append((name, value.shape, hashlib.sha256(value.tobytes()).hexdigest()))
        elif callable(value):
            result.append((name, id(value)))
    if hasattr(solver, "picard_physics_key"):
        result.append(("provider", solver.picard_physics_key()))
    for name in ("Phi", "zetaMz", "Mzvir"):
        value = getattr(solver, name)
        result.append((name, id(getattr(value, "__func__", value))))
    return tuple(result)


def _state(solver):
    """Invalidate cached tidal solutions when their host or rates change.

    Parameters
    ----------
    solver : object
        Native host model providing Mzvir, Phi and zetaMz in consistent mass units.

    Returns
    -------
    tuple
        Current physical-state key for the solver-owned cache.
    """
    key = physics_key(solver)
    if getattr(solver, "_picard_physics_key", None) != key:
        solver._picard_tables = {}
        solver._picard_host_cache = {}
        solver._picard_physics_key = key
    if not hasattr(solver, "_picard_events"):
        solver._picard_events = []
    return key


def cached_host_mass(solver, z):
    """Reuse the native virial conversion across stripping coefficients.

    Parameters
    ----------
    solver : object
        Native solver with an uncached host history anchored at M200(z=0).
    z : array_like
        Dimensionless redshift grid.

    Returns
    -------
    float or numpy.ndarray
        Virial host mass in the native mass unit and input shape. A caller
        may modify its returned array without changing the cached history.
    """
    _state(solver)
    epoch = np.asarray(z, dtype=float)
    key = (epoch.shape, epoch.tobytes())
    cache = solver._picard_host_cache
    if key not in cache:
        if len(cache) >= 64:
            cache.pop(next(iter(cache)))
        cache[key] = np.asarray(solver._Mzvir_uncached(z)).copy()
    value = cache[key]
    return float(value) if value.ndim == 0 else value.copy()


def _iterate(log_ma, coefficients, n_iterations):
    """Integrate the nonlinear log-mass equation on one descending redshift grid.

    Parameters
    ----------
    log_ma : 1-D array_like
        Natural log of each accreted mass in the unit of the host mass.
    coefficients : tuple of numpy.ndarray
        Redshift, Phi, zeta and log host mass on the same 1-D path.
    n_iterations : int
        Number of Picard updates before estimating convergence.

    Returns
    -------
    z : numpy.ndarray
        Redshifts from accretion toward observation.
    delta : numpy.ndarray
        Log bound-to-accreted mass ratio, shape (len(z), len(log_ma)).
    estimate : numpy.ndarray
        Estimated remaining iteration error for each mass. It is infinite
        where the local contraction estimate reaches unity.

    Notes
    -----
    Each update integrates Phi * exp(zeta * (ln(ma/Mhost) + delta))
    by the cumulative trapezoidal rule, with delta(z_acc) = 0. A further
    update estimates the iteration error on this fixed grid.
    """
    z, phi, zeta, log_host = coefficients
    log_ma = np.asarray(log_ma)
    initial = cumulative_trapezoid(phi, x=z, initial=0.0)
    delta = np.broadcast_to(initial[:, None], (len(z), log_ma.size)).copy()
    log_ratio = log_ma[None, :] - log_host[:, None]

    def update(current):
        """Evaluate one Picard update over the mass grid.

        Parameters
        ----------
        current : numpy.ndarray
            Previous log mass change, shape (n_redshift, n_mass).

        Returns
        -------
        numpy.ndarray
            Updated log mass change anchored at zero at accretion.
        """
        rhs = phi[:, None] * np.exp(zeta[:, None] * (log_ratio + current))
        return cumulative_trapezoid(rhs, x=z, axis=0, initial=0.0)

    for _ in range(n_iterations):
        delta = update(delta)
    next_delta = update(delta)
    residual = np.max(np.abs(next_delta - delta), axis=0)
    # A conservative local contraction estimate from the integrated rate.
    contraction = np.max(np.abs(zeta)) * np.max(np.abs(next_delta), axis=0)
    estimate = np.full(residual.shape, np.inf)
    valid = contraction < 1.0
    estimate[valid] = residual[valid] / (1 - contraction[valid])
    return z, delta, estimate


def direct_log_mass(solver, ma, za, z, *, rtol=1e-11, atol=1e-12):
    """Integrate the native stripping law directly for a guarded Picard fallback.

    Parameters
    ----------
    solver : object
        Native host model providing Mzvir, Phi and zetaMz in consistent mass units.
    ma : array_like
        Positive virial mass at accretion, in the same unit as Mzvir.
    za : float
        Accretion redshift.
    z : float or 1-D array_like
        Output redshift(s) in non-increasing order from za toward the present.
    rtol, atol : float, optional
        Local relative and absolute DOP853 tolerances in natural log mass.

    Returns
    -------
    float or numpy.ndarray
        Bound mass in the input mass unit. A redshift vector prepends a time
        axis to the input mass shape.

    Raises
    ------
    ValueError
        If coordinates are non-finite, masses nonpositive, or time runs backward.
    RuntimeError
        If the integrator fails or returns non-finite log masses.
    """
    mass = _real(ma, "ma")
    redshifts = _real(z, "z")
    za = float(_real(za, "za"))
    if np.any(mass <= 0) or za <= -1 or np.any(redshifts <= -1) or np.any(redshifts > za):
        raise ValueError("Require ma>0 and -1<z<=za.")
    scalar_z = redshifts.ndim == 0
    times = np.atleast_1d(redshifts)
    if times.ndim != 1 or np.any(np.diff(times) > 0):
        raise ValueError("History redshifts must be one-dimensional and non-increasing.")
    if mass.size == 0 or times.size == 0:
        return np.empty(mass.shape if scalar_z else times.shape + mass.shape)
    if times[-1] == za:
        return mass.copy() if scalar_z else np.broadcast_to(mass, times.shape + mass.shape).copy()

    def rhs(redshift, log_mass):
        """Evaluate the derivative of log bound mass with respect to redshift.

        Parameters
        ----------
        redshift : float
            Dimensionless redshift at the ODE stage.
        log_mass : numpy.ndarray
            Log masses in the same unit as the native host mass.

        Returns
        -------
        numpy.ndarray
            Phi times (m/Mvir)**zeta for each mass.
        """
        return solver.Phi(redshift) * np.exp(
            solver.zetaMz(redshift) * (log_mass - np.log(solver.Mzvir(redshift)))
        )

    # Integrate from the declared accretion time even if it is not in t_eval.
    unique, reverse = np.unique(times, return_inverse=True)
    result = solve_ivp(
        rhs,
        (za, float(times[-1])),
        np.log(mass.ravel()),
        t_eval=unique[::-1],
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not result.success or np.any(~np.isfinite(result.y)):
        raise RuntimeError("Log-mass ODE failed: " + result.message)
    answer = np.exp(result.y.T[::-1][reverse]).reshape(times.shape + mass.shape)
    answer = answer[0] if scalar_z else answer
    return float(answer) if answer.ndim == 0 else answer


def _fallback(solver, ma, za, z, reason):
    """Record why Picard requests a direct integration and return its mass.

    Parameters
    ----------
    solver : object
        Native host model providing Mzvir, Phi and zetaMz in consistent mass units.
    ma : array_like
        Positive virial mass at accretion, in the same unit as Mzvir.
    za : float
        Accretion redshift.
    z : float or 1-D array_like
        Output redshift(s) in non-increasing order from za toward the present.
    reason : str
        Domain or iteration condition triggering the alternative integration.

    Returns
    -------
    float or numpy.ndarray
        Bound mass in the input mass unit. A redshift vector prepends a time
        axis to the input mass shape.
    """
    _state(solver)
    event = dict(
        reason=reason,
        mass_count=int(np.size(ma)),
        z_acc=float(za),
        z_final=float(np.min(np.atleast_1d(z))),
    )
    solver._picard_events.append(event)
    warnings.warn(
        f"Picard fallback to log-mass DOP853: {reason}; {event['mass_count']} masses.",
        PicardFallbackWarning,
        stacklevel=3,
    )
    return direct_log_mass(solver, ma, za, z)


class PicardTidalStrippingTable:
    """Tabulate the log mass loss to one observation redshift.

    Parameters
    ----------
    solver : object
        Native host model providing Mzvir, Phi and zetaMz in consistent mass units.
    z_final, z_acc_max : float, optional
        Observation and maximum accretion redshift, defaulting to the solver limits.
    n_z_acc, n_log_ratio : int, optional
        Numbers of nodes along accretion redshift and log mass ratio.
    log10_ratio_min, log10_ratio_max : float, optional
        Base-10 limits of accreted mass divided by host mass at accretion.
    n_integration : int, optional
        Number of uniformly spaced redshift samples, including both ends.
    n_iterations : int, optional
        Number of nonlinear updates after the mass-independent first guess.
    convergence_tolerance : float, optional
        Threshold on the estimated remaining change in natural log mass.
        Quadrature and interpolation errors must be checked independently.
    interpolation : {"linear", "cubic"}, optional
        Default to cubic when both axes have at least four nodes, else linear.

    Attributes
    ----------
    delta_ln_mass, iteration_error : numpy.ndarray
        Log mass change and iteration estimate on the accretion-redshift by
        log-mass-ratio grid.

    Notes
    -----
    For repeated endpoint queries, endpoint_mass checks the iteration estimate
    and calls the direct solver where needed. Raw table lookups require points
    inside the table and the same host state used when constructing it.
    """

    def __init__(
        self,
        solver,
        z_final=None,
        z_acc_max=None,
        n_z_acc=48,
        n_log_ratio=32,
        log10_ratio_min=-24.0,
        log10_ratio_max=3.0,
        n_integration=129,
        n_iterations=3,
        convergence_tolerance=8e-4,
        interpolation=None,
    ):
        """Build endpoint corrections for a fixed host history and observing epoch.

        Parameters
        ----------
        solver : object
            Native host model providing Mzvir, Phi and zetaMz in consistent mass units.
        z_final, z_acc_max : float, optional
            Dimensionless observation and maximum accretion redshift. Defaults
            are solver.z_min and solver.z_max.
        n_z_acc, n_log_ratio : int, optional
            Numbers of nodes along accretion redshift and log mass ratio.
        log10_ratio_min, log10_ratio_max : float, optional
            Base-10 limits of accreted mass divided by host mass at accretion.
        n_integration : int, optional
            Number of uniformly spaced redshift samples, including both ends.
        n_iterations : int, optional
            Number of nonlinear updates after the mass-independent first guess.
        convergence_tolerance : float, optional
            Threshold on the estimated remaining change in natural log mass.
            Quadrature and interpolation errors must be checked independently.
        interpolation : {"linear", "cubic"}, optional
            Default to cubic when both axes have at least four nodes, else linear.

        Raises
        ------
        ValueError
            For invalid bounds, grid controls, or nonphysical tidal coefficients.
        """
        self.solver = solver
        self.z_final = float(_real(solver.z_min if z_final is None else z_final, "z_final"))
        self.z_acc_max = float(_real(solver.z_max if z_acc_max is None else z_acc_max, "z_acc_max"))
        if self.z_final <= -1 or self.z_acc_max < self.z_final:
            raise ValueError("Require -1 < z_final <= z_acc_max.")
        self.n_z_acc = _integer(n_z_acc, "n_z_acc", 2)
        self.n_log_ratio = _integer(n_log_ratio, "n_log_ratio", 2)
        self.n_integration = _integer(n_integration, "n_integration", 2)
        self.n_iterations = _integer(n_iterations, "n_iterations", 1)
        self.log10_ratio_min = float(_real(log10_ratio_min, "log10_ratio_min"))
        self.log10_ratio_max = float(_real(log10_ratio_max, "log10_ratio_max"))
        if self.log10_ratio_max <= self.log10_ratio_min:
            raise ValueError("log10_ratio_max must exceed log10_ratio_min.")
        self.convergence_tolerance = float(_real(convergence_tolerance, "convergence_tolerance"))
        if self.convergence_tolerance <= 0:
            raise ValueError("convergence_tolerance must be positive.")
        if interpolation is None:
            interpolation = "cubic" if min(self.n_z_acc, self.n_log_ratio) >= 4 else "linear"
        if interpolation not in ("linear", "cubic"):
            raise ValueError("interpolation must be linear or cubic.")
        if interpolation == "cubic" and min(self.n_z_acc, self.n_log_ratio) < 4:
            raise ValueError("Cubic tables need at least four nodes on each axis.")
        self.interpolation = interpolation
        self.z_acc_grid = np.linspace(self.z_final, self.z_acc_max, self.n_z_acc)
        self.log_ratio_grid = np.linspace(
            self.log10_ratio_min, self.log10_ratio_max, self.n_log_ratio
        )
        self._ln_ratio_grid = self.log_ratio_grid * np.log(10.0)
        self.delta_ln_mass = np.zeros((self.n_z_acc, self.n_log_ratio))
        self.iteration_error = np.zeros_like(self.delta_ln_mass)
        self._physics_key = _state(solver)
        start = time.perf_counter()
        # Evaluate each host/background array once for the complete table.
        # Flattening retains the one-dimensional provider interface.
        paths = np.linspace(self.z_acc_grid, self.z_final, self.n_integration, axis=-1)
        flat = paths.ravel()
        phi = np.broadcast_to(np.asarray(solver.Phi(flat)), flat.shape).reshape(paths.shape)
        zeta = np.broadcast_to(np.asarray(solver.zetaMz(flat)), flat.shape).reshape(paths.shape)
        host = np.broadcast_to(np.asarray(solver.Mzvir(flat)), flat.shape).reshape(paths.shape)
        if (
            np.any(~np.isfinite(host))
            or np.any(host <= 0)
            or np.any(~np.isfinite(phi))
            or np.any(phi < 0)
            or np.any(~np.isfinite(zeta))
        ):
            raise ValueError(
                "Tidal table requires finite, positive host mass and nonnegative rate."
            )
        log_host = np.log(host)
        for i, za in enumerate(self.z_acc_grid):
            if za == self.z_final:
                continue
            log_ma = log_host[i, 0] + self._ln_ratio_grid
            _, history, error = _iterate(
                log_ma, (paths[i], phi[i], zeta[i], log_host[i]), self.n_iterations
            )
            self.delta_ln_mass[i] = history[-1]
            self.iteration_error[i] = error
        self.build_seconds = time.perf_counter() - start
        if self.z_final == self.z_acc_max:
            self._interpolator = None
        elif interpolation == "linear":
            self._interpolator = RegularGridInterpolator(
                (self.z_acc_grid, self._ln_ratio_grid), self.delta_ln_mass, bounds_error=True
            )
        else:
            self._interpolator = RectBivariateSpline(
                self.z_acc_grid, self._ln_ratio_grid, self.delta_ln_mass, kx=3, ky=3, s=0
            )

    def _points(self, ma, z_acc):
        """Locate accreted subhalos in the fixed endpoint table.

        Parameters
        ----------
        ma, z_acc : array_like
            Positive accreted virial mass and dimensionless accretion redshift,
            broadcast to a common shape. Mass uses the host model's unit.

        Returns
        -------
        mass : numpy.ndarray
            Broadcast mass array in host units.
        points : numpy.ndarray
            Flattened pairs of accretion redshift and natural-log mass ratio.

        Raises
        ------
        ValueError
            For stale host state, invalid mass or coordinates outside the table.
        """
        if physics_key(self.solver) != self._physics_key:
            raise ValueError("Picard table is stale after a host or background change.")
        mass, za = np.broadcast_arrays(_real(ma, "ma"), _real(z_acc, "z_acc"))
        if np.any(mass <= 0):
            raise ValueError("Accretion masses must be positive.")
        if np.any(za < self.z_final) or np.any(za > self.z_acc_max):
            raise ValueError("z_acc lies outside the precomputed redshift range.")
        if mass.size == 0:
            return mass, np.empty((0, 2))
        host = np.asarray(self.solver.Mzvir(za))
        ratio = np.log(mass / host)
        if (
            np.any(~np.isfinite(ratio))
            or np.any(ratio < self._ln_ratio_grid[0])
            or np.any(ratio > self._ln_ratio_grid[-1])
        ):
            raise ValueError("ma/Mvir(z_acc) lies outside the precomputed mass-ratio range.")
        return mass, np.column_stack((za.ravel(), ratio.ravel()))

    def delta_log_mass(self, ma, z_acc):
        """Interpolate the natural log of final mass divided by accreted mass.

        Parameters
        ----------
        ma, z_acc : array_like
            Positive accreted virial mass and dimensionless accretion redshift,
            broadcast to a common shape. Mass uses the host model's unit.

        Returns
        -------
        float or numpy.ndarray
            Dimensionless log ratio in the broadcast input shape.
        """
        mass, points = self._points(ma, z_acc)
        if self._interpolator is None or mass.size == 0:
            values = np.zeros_like(mass)
        else:
            if self.interpolation == "linear":
                values = self._interpolator(points).reshape(mass.shape)
            else:
                values = self._interpolator.ev(points[:, 0], points[:, 1]).reshape(mass.shape)
        return float(values) if values.ndim == 0 else values

    def mass(self, ma, z_acc):
        """Evaluate bound mass at the observation redshift of this table.

        Parameters
        ----------
        ma, z_acc : array_like
            Positive accreted virial mass and dimensionless accretion redshift,
            broadcast to a common shape. Mass uses the host model's unit.

        Returns
        -------
        float or numpy.ndarray
            Interpolated mass in the input mass unit and broadcast shape.
        """
        mass = np.asarray(ma)
        values = mass * np.exp(self.delta_log_mass(ma, z_acc))
        return float(values) if values.ndim == 0 else values

    def converged(self, ma, z_acc):
        """Check the iteration estimate around each requested interpolation point.

        Parameters
        ----------
        ma, z_acc : array_like
            Positive accreted virial mass and dimensionless accretion redshift,
            broadcast to a common shape. Mass uses the host model's unit.

        Returns
        -------
        numpy.ndarray
            Boolean mask in the broadcast input shape. True means the adjacent
            table estimates satisfy the configured iteration threshold.
        """
        mass, points = self._points(ma, z_acc)
        if mass.size == 0 or self._interpolator is None:
            return np.ones(mass.shape, dtype=bool)
        i = np.clip(np.searchsorted(self.z_acc_grid, points[:, 0]) - 1, 0, self.n_z_acc - 2)
        j = np.clip(np.searchsorted(self._ln_ratio_grid, points[:, 1]) - 1, 0, self.n_log_ratio - 2)
        offsets = (-1, 0, 1, 2) if self.interpolation == "cubic" else (0, 1)
        error = np.maximum.reduce(
            [
                self.iteration_error[
                    np.clip(i + di, 0, self.n_z_acc - 1), np.clip(j + dj, 0, self.n_log_ratio - 1)
                ]
                for di in offsets
                for dj in offsets
            ]
        )
        return (error <= self.convergence_tolerance).reshape(mass.shape)


def endpoint_mass(
    solver,
    ma,
    za,
    z,
    *,
    n_z_acc=48,
    n_log_ratio=32,
    log10_ratio_min=-24.0,
    log10_ratio_max=3.0,
    n_integration=129,
    n_iterations=3,
    convergence_tolerance=8e-4,
    interpolation=None,
):
    """Evaluate the stripped mass at one or more observation redshifts.

    Parameters
    ----------
    solver : object
        Native host model providing Mzvir, Phi and zetaMz in consistent mass units.
    ma, za, z : array_like
        Accreted virial mass, accretion redshift and observation redshift.
        Inputs broadcast together. Mass uses the host model's unit.
    n_z_acc, n_log_ratio : int, optional
        Numbers of nodes along accretion redshift and log mass ratio.
    log10_ratio_min, log10_ratio_max : float, optional
        Base-10 limits of accreted mass divided by host mass at accretion.
    n_integration : int, optional
        Number of uniformly spaced redshift samples, including both ends.
    n_iterations : int, optional
        Number of nonlinear updates after the mass-independent first guess.
    convergence_tolerance : float, optional
        Threshold on the estimated remaining change in natural log mass.
        Quadrature and interpolation errors must be checked independently.
    interpolation : {"linear", "cubic"}, optional
        Default to cubic when both axes have at least four nodes, else linear.

    Returns
    -------
    float or numpy.ndarray
        Bound mass in the input mass unit and broadcast shape.

    Raises
    ------
    ValueError
        If a mass, redshift, coefficient or numerical control is invalid.

    Notes
    -----
    The table covers 0 <= z <= za <= 7 and -24 <= log10(ma/Mvir(za)) <= 3.
    Requests outside this numerical domain or above the iteration tolerance
    use DOP853 and emit PicardFallbackWarning.
    """
    n_z_acc = _integer(n_z_acc, "n_z_acc", 2)
    n_log_ratio = _integer(n_log_ratio, "n_log_ratio", 2)
    n_integration = _integer(n_integration, "n_integration", 2)
    n_iterations = _integer(n_iterations, "n_iterations", 1)
    log10_ratio_min = float(_real(log10_ratio_min, "log10_ratio_min"))
    log10_ratio_max = float(_real(log10_ratio_max, "log10_ratio_max"))
    if log10_ratio_max <= log10_ratio_min:
        raise ValueError("log10_ratio_max must exceed log10_ratio_min.")
    convergence_tolerance = float(_real(convergence_tolerance, "convergence_tolerance"))
    if convergence_tolerance <= 0:
        raise ValueError("convergence_tolerance must be positive.")
    if interpolation is None:
        interpolation = "cubic" if min(n_z_acc, n_log_ratio) >= 4 else "linear"
    if interpolation not in ("linear", "cubic"):
        raise ValueError("interpolation must be linear or cubic.")
    if interpolation == "cubic" and min(n_z_acc, n_log_ratio) < 4:
        raise ValueError("Cubic tables need at least four nodes on each axis.")
    mass, acc, final = np.broadcast_arrays(_real(ma, "ma"), _real(za, "za"), _real(z, "z"))
    if np.any(mass <= 0) or np.any(acc <= -1) or np.any(final <= -1) or np.any(final > acc):
        raise ValueError("Require ma>0 and -1<z<=za.")
    _state(solver)
    output = mass.copy()
    if mass.size == 0:
        return output
    # Group distinct final redshifts while preserving the usual scalar-target
    # table reuse. SI uses history_mass instead of creating endpoint tables.
    for target in np.unique(final):
        selected = (final == target) & (acc > target)
        if not np.any(selected):
            continue
        m, a = mass[selected], acc[selected]
        host = np.asarray(solver.Mzvir(a))
        if np.any(~np.isfinite(host)) or np.any(host <= 0):
            raise ValueError("Tidal stripping requires finite positive host mass.")
        ratio = np.log10(m / host)
        outside = (
            (a > 7.0 + 32 * np.finfo(float).eps)
            | (target < 0.0)
            | (ratio < -24.0)
            | (ratio > 3.0 + 32 * np.finfo(float).eps)
        )
        selected_output = np.empty_like(m)
        for start in np.unique(a[outside]):
            use = outside & (a == start)
            selected_output[use] = _fallback(
                solver, m[use], start, target, "outside validated redshift or mass-ratio domain"
            )
        inside = ~outside
        if np.any(inside):
            # Required coordinates are included even when custom table limits
            # are narrower. Expansion is explicit in the cached table metadata.
            lo = min(float(log10_ratio_min), float(np.floor(np.min(ratio[inside])))) - 1e-10
            hi = max(float(log10_ratio_max), float(np.ceil(np.max(ratio[inside])))) + 1e-10
            zmax = max(
                min(float(solver.z_max), 7.0) + 32 * np.finfo(float).eps, float(np.max(a[inside]))
            )
            key = (
                float(target),
                zmax,
                n_z_acc,
                n_log_ratio,
                lo,
                hi,
                n_integration,
                n_iterations,
                convergence_tolerance,
                interpolation,
            )
            table = solver._picard_tables.get(key)
            if table is None:
                table = PicardTidalStrippingTable(
                    solver,
                    target,
                    zmax,
                    n_z_acc,
                    n_log_ratio,
                    lo,
                    hi,
                    n_integration,
                    n_iterations,
                    convergence_tolerance,
                    interpolation,
                )
                solver._picard_tables[key] = table
            ok = table.converged(m[inside], a[inside])
            values = table.mass(m[inside], a[inside])
            for start in np.unique(a[inside][~ok]):
                use = (~ok) & (a[inside] == start)
                values[use] = _fallback(
                    solver,
                    m[inside][use],
                    start,
                    target,
                    "Picard iteration convergence estimate exceeded tolerance",
                )
            selected_output[inside] = values
        output[selected] = selected_output
    return float(output) if output.ndim == 0 else output
