import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import brentq

import prompt_cusps as pc
from sashimi_c import cosmology


class PromptCuspTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.model = pc.prompt_cusps(k_fs=1.e6)

    def test_lower_collapse_root(self):
        # Include the spherical limit and the last float below the shape cut.
        shapes = np.r_[0., np.geomspace(1.e-12, 0.25, 30), np.nextafter(0.26, 0.)]

        def residual(f, q):
            return f - 1. - 0.47 * (5. * q * f**2)**0.615

        scalar_roots = [brentq(residual, 1., 5., args=(q,), xtol=1.e-14) for q in shapes]
        factors = pc._ellipsoidal_collapse_factor(shapes)
        self.assertEqual(factors[0], 1.)
        np.testing.assert_allclose(factors, scalar_roots, rtol=1.e-13, atol=0.)

    def test_seeded_bbks_draw(self):
        # Reference grid indices and scalar roots from e07ef8f.
        moments = (np.array([2., np.sqrt(3.), 3.]), 1., 0.5)
        with patch.object(self.model, 'sigma_function', return_value=moments):
            x, nu, e, p, factors = self.model.cusps_Monte_Carlo(
                length=10, rng_seed=17, N_nu=33, N_x=31)
        x_grid = np.linspace(1.e-12, 10., 31)
        nu_grid = np.linspace(1.e-12, 10., 33)
        np.testing.assert_array_equal(x, x_grid[[7, 10, 9, 7, 6, 8, 12, 6, 6, 5]])
        np.testing.assert_array_equal(nu, nu_grid[[7, 1, 4, 3, 2, 3, 3, 5, 6, 0]])
        np.testing.assert_array_equal(np.isfinite(factors), e**2 - p*abs(p) < 0.26)
        np.testing.assert_allclose(factors, [1.3916807014328125, np.nan, 1.3007062931542885,
                                            np.nan, np.nan, 2.717710530934844, np.nan,
                                            1.8214373874987104, 1.5056114830391443, np.nan],
                                   rtol=5.e-13, atol=0.)

    def test_peak_spectrum_across_epochs_and_edits(self):
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(self.directory)
        Path('data').mkdir()
        path = Path('data/powerspectrum31.txt')
        k = np.geomspace(0.1, 1.e6, 100)
        power = 1. + np.log1p(k)
        np.savetxt(path, np.column_stack((k, power)), delimiter=',')
        initial = self.model.powerspectrum(k, z=31.)
        today = self.model.powerspectrum(k)
        np.testing.assert_allclose(today, initial * 32.**(2.*0.901), rtol=1.e-13)
        np.testing.assert_array_equal(self.model.powerspectrum(k, z=31.), initial)
        np.savetxt(path, np.column_stack((k, 4.*power)), delimiter=',')
        np.testing.assert_allclose(self.model.powerspectrum(k), 4.*today, rtol=1.e-13)
        with patch.object(self.model, 'powerspectrum', wraps=self.model.powerspectrum) as evaluate:
            self.model.sigma_function(k)
        evaluate.assert_called_once()

    def test_variance_reuse_and_file_replacement(self):
        table = self.directory / 'camb.dat'
        k = np.geomspace(1.e-10, 1.e9, 256)
        power = 1.e3 / (1. + k**2)
        masses = [-12., -6., 0., 12.]
        for window in ('Sharp-k', 'TopHat'):
            with self.subTest(window=window):
                np.savetxt(table, np.column_stack((k, power)))

                def build(source=table, cutoff=1.e9, alpha=1.8):
                    return pc.build_ps_interpolators(cosmology(), cutoff, window, alpha, source)

                sigma, derivative = build()
                with patch.object(pc, '_variance_samples', side_effect=AssertionError('cache miss')):
                    again, again_derivative = build()
                sigma.fill_value = 0.
                self.assertTrue(np.isnan(again(18.)))
                np.testing.assert_array_equal(sigma(masses), again(masses))
                np.testing.assert_array_equal(derivative(masses), again_derivative(masses))
                self.assertLess(build(cutoff=1.e8)[0](-12.), again(-12.))
                if window == 'Sharp-k':
                    self.assertLess(build(alpha=1.2)[0](-6.), again(-6.))

                # Streams have no stable file identity and are read at each call.
                streamed, _ = build(io.StringIO(table.read_text()))
                np.testing.assert_array_equal(streamed(masses), again(masses))
                new = self.directory / 'replacement'
                np.savetxt(new, np.column_stack((k, 4. * power)))
                new.replace(table)
                updated, d_updated = build()
                np.testing.assert_allclose(updated(masses), 2. * again(masses), rtol=1.e-13, atol=0.)
                np.testing.assert_allclose(d_updated(masses), 4. * again_derivative(masses), rtol=1.e-13, atol=0.)

    def test_five_column_mc_cache(self):
        original = np.array([[1., 1.5, 0.2, 0.1, np.nan]])
        new = np.array([[2., 2.5, 0.3, 0.1, 1.15]])
        for name in ('peaks.txt', 'peaks.txt.gz'):
            with self.subTest(name=name):
                path = self.directory / name
                np.savetxt(path, original)
                path.chmod(0o640)
                mode, contents = path.stat().st_mode, path.read_bytes()

                def read(**kwargs):
                    return self.model._load_or_generate_prompt_cusp_mc(mc_cache_file=path, **kwargs)

                columns = read()
                np.testing.assert_array_equal(np.column_stack(columns), original)
                columns[0][0] = -1.
                np.testing.assert_array_equal(np.column_stack(read()), original)
                with patch.object(self.model, 'cusps_Monte_Carlo', return_value=tuple(new.T)):
                    with patch.object(np, 'savetxt', side_effect=OSError('interrupted')):
                        with self.assertRaises(OSError):
                            read(regenerate=True)
                    self.assertEqual(path.read_bytes(), contents)
                    read(regenerate=True)
                self.assertEqual(path.stat().st_mode, mode)
                np.testing.assert_array_equal(np.column_stack(read()), new)

    def test_universal_peak_diagnostics(self):
        # Frozen from the four intermediate quantities in e07ef8f, with
        # sigma0=2, sigma1=sqrt(3), sigma2=3 and the following four peaks.
        cache = self.directory / 'peaks.txt'
        np.savetxt(cache, [[1., 2., 0.1, 0., 1.1], [2., 1.5, 0.1, 0., 1.1],
                           [2., 0.2, 0.1, 0., 1.1], [2., 1.3, 0.8, 0., np.nan]])
        moments = (np.array([2., np.sqrt(3.), 3.]), 1., 0.5)
        with patch.object(self.model, 'sigma_function', return_value=moments):
            def properties(**options):
                return self.model.cusp_properties(mc_cache_file=cache, **options)

            with patch.object(pc, '_peak_number_density', side_effect=AssertionError('unexpected integral')):
                fraction, light = properties()
            f, j, diag = properties(f_surv=0.35, return_peak_diagnostics=True)
            self.assertEqual(f, 0.5)
            self.assertEqual(fraction, f)
            np.testing.assert_array_equal(light, j)
            expected = {'N_peaks_M': 4.333327387026309e-13, 'N_cusps_M': 7.58332292729604e-14,
                        'J_cusps_M': 1.7545385944076116e14, 'J_cusps_fit': 6.600996531649125e16}
            for key, value in expected.items():
                np.testing.assert_allclose(diag[key], value, rtol=2.e-12, atol=0.)

            f, j, empty = properties(z=1.e5, f_surv=0.35, return_peak_diagnostics=True)
            self.assertEqual((f, j), (0., 0.))
            self.assertEqual(empty['N_cusps_M'], 0.)
            self.assertEqual(empty['J_cusps_M'], 0.)
            self.assertEqual(empty['N_peaks_M'], diag['N_peaks_M'])
            self.assertEqual(empty['J_cusps_fit'], diag['J_cusps_fit'])
