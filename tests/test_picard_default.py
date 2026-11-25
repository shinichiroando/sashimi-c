"""Regression tests for the native catalogue and the nonlinear stripping law."""

import inspect
import warnings

import numpy as np
import pytest

import sashimi_c as model
from picard_tidal_stripping import PicardFallbackWarning


def test_default_mass_against_independent_integration():
    """Resolve the unexpanded tidal law for an off-grid accretion epoch.

    Notes
    -----
    The native LSODA solver evolves mass directly with tight tolerances.
    Picard must agree to 0.1 percent in mass for every sampled mass and
    epoch, and return the initial masses at accretion. This comparison
    checks numerical integration for one host below z=4.
    """
    solver = model.TidalStrippingSolver(1e12, z_max=3.0)
    mass = np.geomspace(1e7, 1e10, 5) * solver.Msun
    za = 2.73
    epochs = np.linspace(za, 0.17, 31)
    reference = solver.subhalo_mass_stripped(
        mass, za, epochs[-1], method="odeint", rtol=1e-10, atol=1e-11
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", PicardFallbackWarning)
        result = solver.subhalo_mass_stripped(mass, za, epochs[-1])
    np.testing.assert_allclose(result, reference, rtol=1e-3, atol=0.0)
    np.testing.assert_array_equal(solver.subhalo_mass_stripped(mass, za, za), mass)


def test_guarded_fallback_and_input_domain():
    """Exercise an unresolved Picard iteration and an invalid redshift.

    Notes
    -----
    A single update with a tight residual threshold must warn and agree
    with a separate mass-space ODE solve to 1e-7 relative accuracy. An
    observation before accretion must raise ValueError. The controls do
    not bound interpolation errors on the default grid.
    """
    solver = model.TidalStrippingSolver(1e12, z_max=3.0)
    mass = np.array([1e7, 1e10]) * solver.Msun
    ref = solver.subhalo_mass_stripped(mass, 2.8, 0.3, method="odeint", rtol=1e-10, atol=1e-11)
    with pytest.warns(PicardFallbackWarning, match="convergence"):
        result = solver.subhalo_mass_stripped(
            mass, 2.8, 0.3, n_iterations=1, convergence_tolerance=1e-12
        )
    np.testing.assert_allclose(result, ref, rtol=1e-7, atol=0.0)
    with pytest.raises(ValueError):
        solver.subhalo_mass_stripped(mass, 0.3, 2.8)


def test_catalogue_default_and_legacy_method():
    """Check the stripping default through the catalogue entry point.

    Notes
    -----
    A small deterministic grid compares its bound-mass column to LSODA
    with a 0.1 percent numerical tolerance. Explicit Shanks must remain
    callable and produce the same accretion coordinates. The catalogue
    comparison covers this host and grid only.
    """
    catalogue = model.subhalo_properties().subhalo_properties_calc
    options = dict(
        M0=1e12,
        redshift=0.0,
        dz=0.5,
        zmax=1.5,
        N_ma=12,
        N_herm=3,
        N_hermNa=8,
        logmamin=8.0,
        logmamax=10.0,
    )
    default = catalogue(**options)
    legacy = catalogue(**options, method="pert2_shanks")
    reference = catalogue(**options, method="odeint", rtol=1e-10, atol=1e-11)
    assert inspect.signature(catalogue).parameters["method"].default == "picard_table"
    np.testing.assert_allclose(default[4], reference[4], rtol=1e-3, atol=0.0)
    np.testing.assert_array_equal(default[0], legacy[0])
    np.testing.assert_array_equal(default[1], legacy[1])
    assert (
        inspect.signature(model.subhalo_observables).parameters["method"].default == "picard_table"
    )


def test_host_change_rebuilds_endpoint_table():
    """Require separate endpoint tables for different host histories.

    Notes
    -----
    After changing M0 on a warm solver, its mass must reproduce a fresh
    solver at the new M0 within floating-point precision and differ from
    the old solution. This probes cache validity at one mass and epoch.
    """
    solver = model.TidalStrippingSolver(1e12, z_max=3.0)
    old_mass = solver.subhalo_mass_stripped(1e8, 2.6, 0.3)
    solver.M0 = 1e13
    actual = solver.subhalo_mass_stripped(1e8, 2.6, 0.3)
    expected = model.TidalStrippingSolver(1e13, z_max=3.0).subhalo_mass_stripped(1e8, 2.6, 0.3)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0)
    assert not np.isclose(actual, old_mass)
