"""
Tests of the opt-in CLASS sigma(M, z) of SASHIMI-C (class_sigma.py, sigma_model='CLASS').

Run from the repository root, as a script (prints the numbers) or with pytest:
    OMP_NUM_THREADS=1 python -B tests/test_class_sigma.py
    OMP_NUM_THREADS=1 python -B -m pytest -p no:cacheprovider tests/test_class_sigma.py

(a) Default behaviour unchanged: the original sashimi_c.py (imported under another module name) and
    this one give bitwise-identical sigmaMz, dsdm, deltac_func, Mzi, Mzzi, dMdz, Na_calc and
    subhalo_properties_calc, for the default and for prompt_cusps=True, and the same warnings.
(b) ClassSigma vs the orbit-sashimi prototype (ps_sigma.sigma applied to ps_compare_class.class_pk,
    with its z < 7 anchor moved to 1e4 Mpc^-1) for Sharp-k (c = 2.5), TopHat and Smooth-k, at
    z = 0, 3, 10, 30, 100: agreement within 0.5 %.
(c) CLASS mode (Sharp-k): Na_calc in the z_acc field reduces exactly to the default Na_calc when the
    CLASS sigma is replaced by the Ludlow sigma times growthD (models 3, 1, 2); accretion-mass
    function CLASS/Ludlow for z_acc <= 3 and <= 7 and its suppression below the cutoff;
    Na_calc with Smooth-k and TopHat.
(e) Error paths: missing or wrong table, unknown options, invalid ps_c/ps_beta/k_fs, k_fs_Mpc=None,
    cosmology mismatch, CLASS-only options in the Ludlow mode, make_class_table arguments and its
    OMP_NUM_THREADS handling (with a stand-in for classy, so CLASS is not run).
(d) Timings (script mode).

Local inputs outside this repository (environment overrides): SASHIMI_ORIG (original sashimi_c.py,
default: the main checkout ../../../LMC/sashimi-c), PS_REF_DIR (prototype directory, default
../power_spectrum) and CLASS_TABLE (default PS_REF_DIR/tables/class_pcdm.npz). A test whose inputs
are missing is skipped: (a) needs SASHIMI_ORIG, (b) needs PS_REF_DIR, and (b), (c), (e) need the
table (or build one with class_sigma.make_class_table). Each test runs in TEST_WORKDIR, or in a new
temporary directory that is removed afterwards, so nothing is written next to either sashimi_c.py.
"""
import contextlib
import importlib.util
import os
import shutil
import sys
import tempfile
import time
import traceback
import types
import unittest
import warnings
import numpy as np

sys.dont_write_bytecode = True
REPO       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG       = os.environ.get('SASHIMI_ORIG',
                            os.path.normpath(os.path.join(REPO,'..','..','..','LMC','sashimi-c','sashimi_c.py')))
PS_REF_DIR = os.environ.get('PS_REF_DIR',os.path.normpath(os.path.join(REPO,'..','power_spectrum')))
TABLE      = os.environ.get('CLASS_TABLE',os.path.join(PS_REF_DIR,'tables','class_pcdm.npz'))
if REPO not in sys.path:
    sys.path.insert(0,REPO)

import sashimi_c as sc  # noqa: E402
import class_sigma  # noqa: E402
from class_sigma import ClassSigma  # noqa: E402

Msun     = sc.units_and_constants().Msun
Mpc      = sc.units_and_constants().Mpc
FAILURES = []
TIMES    = {}
SMALL    = dict(redshift=0.0, dz=0.01, zmax=3.0, N_ma=60, logmamin=-6)       # (a)
SMALL_C  = dict(redshift=0.0, dz=0.01, zmax=3.0, N_ma=81, logmamin=-9)       # (c): reaches below the cutoff
SMALL_C7 = dict(SMALL_C, zmax=7.0)                                           # (c): sashimi's default zmax
OUT_KEYS = ('ma200','z_acc','rs_acc','rhos_acc','m_z0','rs_z0','rhos_z0','ct_z0','weight','survive')


def check(name, ok, info=''):
    print('  [%s] %s %s'%('PASS' if ok else 'FAIL',name,info))
    if not ok:
        FAILURES.append(name)


def assert_checks(n0):
    """ End of a test: raise if a check since FAILURES[n0] failed (pytest then reports the test as failed). """
    assert len(FAILURES)==n0, 'failed checks: %s'%(FAILURES[n0:],)


def require(path, what):
    if not os.path.exists(path):
        raise unittest.SkipTest('%s not found: %s'%(what,path))


@contextlib.contextmanager
def workdir():
    """ chdir into TEST_WORKDIR or into a new temporary directory, which is removed on exit. """
    given = os.environ.get('TEST_WORKDIR')
    path  = given or tempfile.mkdtemp(prefix='class_sigma_test_')
    old   = os.getcwd()
    os.makedirs(path,exist_ok=True)
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(old)
        if not given:
            shutil.rmtree(path,ignore_errors=True)


def maxdiff(a, b):
    a = np.asarray(a,dtype=float)
    b = np.asarray(b,dtype=float)
    if a.shape!=b.shape:
        return np.inf
    d = np.abs(a-b)
    both_nan = np.isnan(a)&np.isnan(b)
    return float(np.max(np.where(both_nan,0.,d))) if d.size else 0.


def raises(exc, func, *args, **kwargs):
    """ (raised exc?, message) """
    try:
        func(*args,**kwargs)
    except exc as err:
        return True, str(err)
    except Exception as err:
        return False, 'raised %r instead'%(err,)
    return False, 'nothing raised'


def caught_warnings(func, *args, **kwargs):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        func(*args,**kwargs)
    return [str(x.message) for x in w]


def load_original():
    spec = importlib.util.spec_from_file_location('sashimi_c_orig',ORIG)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def link_camb_data():
    """ prompt_cusps=True reads ./data/Planck2018_CAMB_extrap.dat: symlink it into the work dir. """
    src = os.path.join(os.path.dirname(ORIG),'data','Planck2018_CAMB_extrap.dat')
    if not os.path.exists(src):
        return False
    os.makedirs('data',exist_ok=True)
    dst = os.path.join('data','Planck2018_CAMB_extrap.dat')
    if not os.path.exists(dst):
        os.symlink(src,dst)
    return True


def load_table():
    with np.load(TABLE) as f:
        return {key: f[key] for key in f.files}


# ----------------------------------------------------------------------------------------- (a)
def test_default_unchanged():
    print('\n(a) default behaviour bitwise unchanged (original: %s)'%ORIG)
    require(ORIG,'original sashimi_c.py')
    n0 = len(FAILURES)
    with workdir():
        orig  = load_original()
        modes = [('default',{})]
        if link_camb_data():
            modes.append(('prompt_cusps=True',{'prompt_cusps': True}))
        else:
            print('  (CAMB file not found: prompt_cusps=True comparison skipped)')
        M     = np.logspace(-6,15,85)*Msun
        z     = np.array([0.,0.5,1.,3.,7.]).reshape(-1,1)
        M0    = np.array([1.e6,1.e9,1.e12,1.e14]).reshape(-1,1)*Msun
        zacc  = np.arange(0.01,3.01,0.05)
        ma    = np.logspace(-6,10,33)*Msun
        for label, kw in modes:
            ho    = orig.subhalo_properties(**kw)
            warns = caught_warnings(sc.subhalo_properties,**kw)
            check('%s: the same warnings on construction as the original'%label,
                  warns==caught_warnings(orig.subhalo_properties,**kw),str(warns))
            hn    = sc.subhalo_properties(**kw)
            pairs = [('sigmaMz',ho.sigmaMz(M,z),hn.sigmaMz(M,z)),
                     ('dsdm',ho.dsdm(M,z),hn.dsdm(M,z)),
                     ('deltac_func',ho.deltac_func(z),hn.deltac_func(z)),
                     ('s_func',ho.s_func(M),hn.s_func(M)),
                     ('Mzi',ho.Mzi(M0,z.T),hn.Mzi(M0,z.T)),
                     ('Mzzi',ho.Mzzi(M0,z.T+0.3,0.3),hn.Mzzi(M0,z.T+0.3,0.3)),
                     ('dMdz',ho.dMdz(M0,z.T+0.3,0.3),hn.dMdz(M0,z.T+0.3,0.3))]
            for Na_model in (3,1,2):
                pairs.append(('Na_calc model %d'%Na_model,
                              ho.Na_calc(ma,zacc,1.e12*Msun,Na_model=Na_model),
                              hn.Na_calc(ma,zacc,1.e12*Msun,Na_model=Na_model)))
            t0    = time.time()
            out_o = ho.subhalo_properties_calc(1.e12*Msun,**SMALL)
            t1    = time.time()
            out_n = hn.subhalo_properties_calc(1.e12*Msun,**SMALL)
            TIMES['subhalo_properties_calc (a) %s, original'%label] = t1-t0
            TIMES['subhalo_properties_calc (a) %s, this'%label]     = time.time()-t1
            pairs += [('subhalo_properties_calc:'+k,a,b) for k, a, b in zip(OUT_KEYS,out_o,out_n)]
            for name, a, b in pairs:
                d = maxdiff(a,b)
                check('%s %s'%(label,name),d==0.,'max|diff| = %.3g, shape %s'%(d,np.shape(b)))
    assert_checks(n0)


# ----------------------------------------------------------------------------------------- (c)
class LudlowStandIn:
    """ The Ludlow sigma times growthD with the ClassSigma interface (scale-independent growth). """

    def __init__(self, hm):
        self.hm = hm

    def sigma(self, M, z):
        return self.hm.sigma_Ludlow(M)*self.hm.growthD(z)

    def dsdm(self, M, z):
        return self.hm.dsdm_Ludlow(M)*self.hm.growthD(z)**2

    def growth_eff(self, M, z):
        return self.hm.growthD(z)/self.hm.growthD(0.)


def mass_function(out):
    """ dN/dln m_a from the weights (sum over z_acc and concentration), on the ma200 grid. """
    ma200, weight = out[0], out[8]
    m   = np.unique(ma200)
    dNb = np.array([weight[ma200==mi].sum() for mi in m])
    return m, dNb/np.log(m[1]/m[0])


def test_zacc_frame_reduces_to_default():
    print('\n(c1) CLASS-mode Na_calc (z_acc field) with the Ludlow sigma x growthD == default Na_calc')
    require(TABLE,'CLASS table')
    n0   = len(FAILURES)
    shs  = sc.subhalo_properties(sigma_model='CLASS',class_table=TABLE)
    shs.class_sigma = LudlowStandIn(shs)
    shl  = sc.subhalo_properties()
    zacc = np.arange(0.01,7.01,0.05)
    ma   = np.logspace(-6,11,35)*Msun
    for Na_model in (3,1,2):
        a   = shl.Na_calc(ma,zacc,1.e12*Msun,Na_model=Na_model)
        b   = shs.Na_calc(ma,zacc,1.e12*Msun,Na_model=Na_model)
        ok  = np.all(np.isfinite(b))&np.all((a==0.)==(b==0.))
        rel = float(np.max(np.abs(b[a>0.]/a[a>0.]-1.)))
        check('model %d: relative difference < 1e-9 (z_acc <= 7)'%Na_model,ok&(rel<1.e-9),'max %.2e'%rel)
    assert_checks(n0)


def class_vs_ludlow(grid, label):
    """ Na_calc and subhalo_properties_calc in the CLASS (Sharp-k, c = 2.5) and Ludlow modes on grid. """
    t0   = time.time()
    shc  = sc.subhalo_properties(sigma_model='CLASS',class_table=TABLE,ps_filter='Sharp-k',ps_c=2.5)
    TIMES['subhalo_properties(sigma_model=CLASS) construction'] = time.time()-t0
    shl  = sc.subhalo_properties()
    dz   = grid['dz']
    zacc = np.arange(grid['redshift']+dz,grid['zmax']+dz,dz)
    ma   = np.logspace(grid['logmamin'],np.log10(0.1*1.e12),grid['N_ma'])*Msun
    na   = {}
    outs = {}
    for name, sh in (('Ludlow',shl),('CLASS',shc)):
        t0         = time.time()
        na[name]   = sh.Na_calc(ma,zacc,1.e12*Msun)
        TIMES['Na_calc %s %s'%(label,name)] = time.time()-t0
        t0         = time.time()
        outs[name] = sh.subhalo_properties_calc(1.e12*Msun,**grid)
        TIMES['subhalo_properties_calc %s %s'%(label,name)] = time.time()-t0
    return shc, shl, zacc, ma, na, outs


def report_class_mode(grid, label):
    shc, shl, zacc, ma, na, outs = class_vs_ludlow(grid,label)
    dz = grid['dz']
    check('%s: CLASS Na_calc finite and >= 0'%label,np.all(np.isfinite(na['CLASS']))&np.all(na['CLASS']>=0.))
    check('%s: CLASS weights finite and >= 0'%label,
          np.all(np.isfinite(outs['CLASS'][8]))&np.all(outs['CLASS'][8]>=0.))
    check('%s: Mzi/Mzzi/dMdz unchanged in CLASS mode (host history on Ludlow sigma)'%label,
          maxdiff(shc.dMdz(1.e12*Msun,zacc,0.),shl.dMdz(1.e12*Msun,zacc,0.))==0.)
    m, dn_l = mass_function(outs['Ludlow'])
    _, dn_c = mass_function(outs['CLASS'])
    ratio   = dn_c/dn_l
    # the same from Na directly: dN/dln m_a = sum_z Na dz/(1+z). The weights equal f times this, with
    # one factor f per run from the normalisation weight/sum(weight)*Na_total (Simpson vs grid sum)
    w       = dz/(1.+zacc[:,np.newaxis])
    dr      = {name: np.sum(na[name]*w,axis=0) for name in na}
    ratio_d = dr['CLASS']/dr['Ludlow']
    late    = {name: np.sum((na[name]*w)[zacc>3.],axis=0)/dr[name] for name in na}
    fl      = np.median((dn_l/dr['Ludlow'])[m>1.e-5*Msun])
    fc      = np.median((dn_c/dr['CLASS'])[m>1.e-5*Msun])
    print('    m_a [Msun]  dN/dln m_a from the weights: Ludlow     CLASS    CLASS/Ludlow | from Na: CLASS/Ludlow'
          + (' | share z_acc > 3: Ludlow CLASS' if grid['zmax']>3. else ''))
    for lm in range(-9,11):
        i = int(np.argmin(np.abs(np.log10(m/Msun)-lm)))
        print('    1e%+03d                        %12.4e %12.4e  %10.3e |        %10.3e'
              %(lm,dn_l[i],dn_c[i],ratio[i],ratio_d[i])
              + ('   |                   %.3f  %.3f'%(late['Ludlow'][i],late['CLASS'][i]) if grid['zmax']>3. else ''))
    print('    weight normalisation factor f (weights / Na-direct, m_a > 1e-5 Msun): Ludlow %.4f, CLASS %.4f'
          %(fl,fc))
    for name in ('Ludlow','CLASS'):
        wt = outs[name][8]
        print('    total number %-6s: all m_a %.4e,  m_a >= 1e-6 Msun %.4e,  m_a >= 1e-3 Msun %.4e'
              %(name,wt.sum(),wt[outs[name][0]>=1.e-6*Msun].sum(),wt[outs[name][0]>=1.e-3*Msun].sum()))
    below = m<=1.e-7*Msun
    check('%s: CLASS/Ludlow < 1e-2 for m_a <= 1e-7 Msun (below the cutoff, M_hm ~ 2.7e-6 Msun)'%label,
          np.all(ratio_d[below]<1.e-2),'max ratio there %.2e'%ratio_d[below].max())
    deep = m<=1.e-8*Msun
    check('%s: CLASS/Ludlow < 1e-10 for m_a <= 1e-8 Msun'%label,np.all(ratio_d[deep]<1.e-10),
          'max ratio there %.2e'%ratio_d[deep].max())
    above = (m>=1.e3*Msun)&(m<=1.e10*Msun)
    check('%s: CLASS/Ludlow within [0.5, 2] for 1e3 <= m_a <= 1e10 Msun (weights and Na-direct)'%label,
          np.all((ratio[above]>0.5)&(ratio[above]<2.)&(ratio_d[above]>0.5)&(ratio_d[above]<2.)),
          'weights %.3f .. %.3f, Na-direct %.3f .. %.3f'
          %(ratio[above].min(),ratio[above].max(),ratio_d[above].min(),ratio_d[above].max()))
    big = m>=1.e8*Msun
    check('%s: CLASS/Ludlow (Na-direct) within 5%% for m_a >= 1e8 Msun (Jeans scale and cutoff irrelevant)'
          %label,np.all(np.abs(ratio_d[big]-1.)<0.05),'%.3f .. %.3f'%(ratio_d[big].min(),ratio_d[big].max()))


def test_class_mode_zmax3():
    print('\n(c2) CLASS mode (Sharp-k, c = 2.5, k_fs = 1.06e6 Mpc^-1) vs Ludlow; M0 = 1e12 Msun, %s'%SMALL_C)
    require(TABLE,'CLASS table')
    n0 = len(FAILURES)
    with workdir():
        report_class_mode(SMALL_C,'zmax=3')
    assert_checks(n0)


def test_class_mode_zmax7():
    print('\n(c3) the same with zmax = 7 (sashimi default), %s'%SMALL_C7)
    require(TABLE,'CLASS table')
    n0 = len(FAILURES)
    with workdir():
        report_class_mode(SMALL_C7,'zmax=7')
    assert_checks(n0)


def test_class_mode_windows():
    print('\n(c4) Na_calc with the Smooth-k and TopHat windows (models 3, 1, 2), z_acc <= 3')
    require(TABLE,'CLASS table')
    n0   = len(FAILURES)
    zacc = np.arange(0.01,3.01,0.02)
    ma   = np.logspace(-9,11,41)*Msun
    w    = 0.02/(1.+zacc[:,np.newaxis])
    ref  = np.sum(sc.subhalo_properties().Na_calc(ma,zacc,1.e12*Msun)*w,axis=0)
    for window in ('Smooth-k','TopHat'):
        t0 = time.time()
        sh = sc.subhalo_properties(sigma_model='CLASS',class_table=TABLE,ps_filter=window)
        for Na_model in (3,1,2):
            na = sh.Na_calc(ma,zacc,1.e12*Msun,Na_model=Na_model)
            check('%s model %d: Na finite and >= 0'%(window,Na_model),np.all(np.isfinite(na))&np.all(na>=0.))
            if Na_model==3:
                r = np.sum(na*w,axis=0)/ref
                print('    %-8s CLASS/Ludlow at m_a = 1e-9, 1e-6, 1e-3, 1, 1e3, 1e6, 1e9 Msun:'%window,
                      ' '.join('%.3g'%v for v in r[::6]))
                big = ma>=1.e8*Msun
                check('%s: CLASS/Ludlow within 5%% for m_a >= 1e8 Msun'%window,
                      np.all(np.abs(r[big]-1.)<0.05),'%.3f .. %.3f'%(r[big].min(),r[big].max()))
        TIMES['Na_calc x3 %s'%window] = time.time()-t0
    assert_checks(n0)


# ----------------------------------------------------------------------------------------- (e)
def write_table(path, **changes):
    """ Copy of the CLASS table with some entries changed (params: a dict, or None to drop it). """
    t = load_table()
    if 'params' in changes:
        params = changes.pop('params')
        t.pop('params')
        if params is not None:
            t['params'] = np.str_(repr(params))
    t.update(changes)
    np.savez(path,**t)
    return path


def test_errors_and_checks():
    print('\n(e) error paths and input checks')
    require(TABLE,'CLASS table')
    n0 = len(FAILURES)
    with workdir():
        hp = sc.halo_model
        for name, exc, kw, word in (
                ('no class_table',ValueError,dict(sigma_model='CLASS'),'make_class_table'),
                ('missing table file',FileNotFoundError,dict(sigma_model='CLASS',class_table='none.npz'),
                 'make_class_table'),
                ('unknown sigma_model',ValueError,dict(sigma_model='camb'),'sigma_model'),
                ('unknown ps_filter',ValueError,dict(sigma_model='CLASS',class_table=TABLE,ps_filter='Gauss'),
                 'window'),
                ('ps_c <= 0',ValueError,dict(sigma_model='CLASS',class_table=TABLE,ps_c=-1.),'c must'),
                ('ps_beta <= 0',ValueError,dict(sigma_model='CLASS',class_table=TABLE,ps_filter='Smooth-k',
                                                ps_beta=0.),'beta must'),
                ('k_fs_Mpc <= 0',ValueError,dict(sigma_model='CLASS',class_table=TABLE,k_fs_Mpc=-1.),'k_fs must')):
            ok, msg = raises(exc,hp,**kw)
            check('%s -> %s'%(name,exc.__name__),ok and word in msg,msg)
        t     = load_table()
        n_cut = int(np.searchsorted(t['k_Mpc'],1.e3,side='right'))
        short = write_table('short.npz',k_Mpc=t['k_Mpc'][:n_cut],pcdm=t['pcdm'][:,:n_cut])
        ok, msg = raises(ValueError,hp,sigma_model='CLASS',class_table=short)
        check('table with k_max = %.3g Mpc^-1 < 1e4 -> ValueError'%t['k_Mpc'][n_cut-1],ok,msg)
        params = dict(class_sigma.CLASS_PARAMS)
        warns  = caught_warnings(hp,sigma_model='CLASS',class_table=write_table('h07.npz',params=dict(params,h=0.7)))
        check('table with h = 0.7 -> warning',any('differs from sashimi' in x for x in warns),str(warns))
        warns  = caught_warnings(hp,sigma_model='CLASS',class_table=write_table('nopar.npz',params=None))
        check('table without params -> warning',any('not checked' in x for x in warns),str(warns))
        warns  = caught_warnings(hp,sigma_model='CLASS',class_table=TABLE)
        check('the reference table: no warning',not warns,str(warns))
        warns  = caught_warnings(hp,ps_filter='TopHat')
        check('CLASS-only option with sigma_model=Ludlow -> warning',any('ignored' in x for x in warns),str(warns))
        M      = np.logspace(-9,12,8)*Msun
        s_none = hp(sigma_model='CLASS',class_table=TABLE,k_fs_Mpc=None).sigmaMz(M,0.)
        s_inf  = hp(sigma_model='CLASS',class_table=TABLE,k_fs_Mpc=np.inf).sigmaMz(M,0.)
        check('k_fs_Mpc=None (no cutoff) == k_fs_Mpc=inf in the CLASS mode',maxdiff(s_none,s_inf)==0.,
              'sigma(1e-9 Msun) = %.4g'%s_none[0])
        cs = hp(sigma_model='CLASS',class_table=TABLE).class_sigma
        za = cs.z_nodes<class_sigma.Z_ANCHOR_SWITCH
        check('extension anchored at the table k nearest 1e4 Mpc^-1 for z < 7, at k_max for z >= 7',
              np.allclose(np.exp(cs.lnk_anchor[za]),1.e4,rtol=0.13)
              and np.all(cs.lnk_anchor[~za]==cs.lnk_tab[-1]),
              'k_anchor = %.4g, %.4g Mpc^-1'%(np.exp(cs.lnk_anchor[0]),np.exp(cs.lnk_anchor[-1])))
        k_a  = np.exp(cs.lnk_anchor)/Mpc
        jump = max(abs(cs.lnDelta2(k_a[iz]*(1.+1.e-7),iz)-cs.lnDelta2(k_a[iz]*(1.-1.e-7),iz))
                   for iz in range(len(cs.z_nodes)))
        check('P(k) continuous at the anchor',jump<1.e-5,'max |d ln Delta^2| = %.1e'%jump)
        ok, msg = raises(ValueError,class_sigma.make_class_table,'x.npz',kmax=1.e3)
        check('make_class_table(kmax=1e3) -> ValueError (before running CLASS)',ok,msg)
        ok, msg = raises(ValueError,class_sigma.make_class_table,'x.npz',z_grid=(1.,2.))
        check('make_class_table(z_grid not from 0) -> ValueError',ok,msg)
        check_omp_handling()
    assert_checks(n0)


class FakeClass:
    """ Stand-in for classy.Class that records OMP_NUM_THREADS at compute() (CLASS is not run). """
    seen = []

    def set(self, par):
        self.par = par

    def compute(self):
        FakeClass.seen.append(os.environ.get('OMP_NUM_THREADS'))

    def get_transfer(self, z):
        k = np.logspace(-4.,4.5,30)
        return {'k (h/Mpc)': k/self.par['h'], 'd_cdm': -k**0.1, 'd_b': -k**0.1}

    def sigma8(self):
        return 0.8

    def struct_cleanup(self):
        pass


def check_omp_handling():
    saved_mod = sys.modules.get('classy')
    saved_env = os.environ.get('OMP_NUM_THREADS')
    fake      = types.ModuleType('classy')
    fake.Class = FakeClass
    sys.modules['classy'] = fake
    try:
        cases = []
        for shell, arg in (('3',None),('3',1),(None,2)):
            if shell is None:
                os.environ.pop('OMP_NUM_THREADS',None)
            else:
                os.environ['OMP_NUM_THREADS'] = shell
            class_sigma.make_class_table('fake.npz',omp_threads=arg)
            cases.append((shell,arg,FakeClass.seen[-1],os.environ.get('OMP_NUM_THREADS')))
        expect = [('3',None,'3','3'),('3',1,'1','3'),(None,2,'2',None)]
        check('make_class_table: omp_threads=None keeps OMP_NUM_THREADS; an integer is used and then restored',
              cases==expect,'(shell, omp_threads, seen by CLASS, after) = %s'%(cases,))
    finally:
        if saved_mod is None:
            sys.modules.pop('classy',None)
        else:
            sys.modules['classy'] = saved_mod
        if saved_env is None:
            os.environ.pop('OMP_NUM_THREADS',None)
        else:
            os.environ['OMP_NUM_THREADS'] = saved_env


# ----------------------------------------------------------------------------------------- (b)
def test_vs_reference():
    """ Runs last when run as a script: the prototype puts the main checkout on sys.path. """
    print('\n(b) ClassSigma vs the prototype ps_sigma.sigma(ps_compare_class.class_pk) (%s)'%PS_REF_DIR)
    require(os.path.join(PS_REF_DIR,'ps_compare_class.py'),'prototype')
    require(TABLE,'CLASS table')
    n0 = len(FAILURES)
    if PS_REF_DIR not in sys.path:
        sys.path.append(PS_REF_DIR)
    import ps_compare_class as pcc
    import ps_sigma as ps
    hm   = sc.halo_model()
    t    = load_table()
    # the prototype anchors the extension at 3e3 Mpc^-1 for z < 7 and at its last k otherwise: give it
    # the table cut at the node nearest 1e4 Mpc^-1, labelled z >= 7, to move its anchor there
    ja   = int(np.argmin(np.abs(np.log(t['k_Mpc']/class_sigma.K_ANCHOR_LOWZ_MPC))))
    t_lo = dict(t,k_Mpc=t['k_Mpc'][:ja+1],pcdm=t['pcdm'][:,:ja+1],z=np.full_like(t['z'],7.))
    k    = np.logspace(-5,9,1401)*Mpc**-1         # 100 per decade, as the prototype's CAMB grid
    m    = np.logspace(-9,12,64)*Msun             # off the ClassSigma grid on purpose
    wins = [('Sharp-k','sk',2.5,None),('TopHat','th',None,None),('Smooth-k','smk',3.3,4.8)]
    for window, kind, c, beta in wins:
        t0 = time.time()
        cs = ClassSigma(TABLE,hm,k_fs=1.06e6*Mpc**-1,window=window,c=c,beta=4.8)
        TIMES['ClassSigma build %s'%window] = time.time()-t0
        worst = 0.
        for zz in (0.,3.,10.,30.,100.):
            iz  = int(np.argmin(np.abs(t['z']-zz)))
            pk  = pcc.class_pk(iz,t_lo if zz<class_sigma.Z_ANCHOR_SWITCH else t,k)
            ref = ps.sigma(m,k,pk,kind,c,beta)
            r   = cs.sigma(m,zz)/ref
            worst = max(worst,np.max(np.abs(r-1.)))
            print('    %-8s z=%5.0f: sigma/ref at M = 1e-9, 1e-6, 1e-3, 1, 1e3, 1e6, 1e9, 1e12 Msun:'%(window,zz),
                  ' '.join('%.5f'%v for v in r[::9]))
        check('%s within 0.5%% of the prototype'%window,worst<5.e-3,'max |ratio-1| = %.2e'%worst)
    print('    fitted ln(B/Mpc) = %.4f (prototype 2.769)'%cs.lnB)
    # internal consistency of ClassSigma
    cs  = ClassSigma(TABLE,hm,k_fs=1.06e6*Mpc**-1,window='Sharp-k',c=2.5)
    M   = np.logspace(-7,14,43)*Msun
    h   = 1.e-3
    err = 0.
    for zz in (0.,10.,100.):
        num = (cs.sigma(M*np.exp(h),zz)**2-cs.sigma(M*np.exp(-h),zz)**2)/(2.*np.sinh(h)*M)
        err = max(err,np.max(np.abs(num/cs.dsdm(M,zz)-1.)))
    check('Sharp-k dsdm = -Delta^2(k_s)/(3M) vs d sigma^2/dM of the sigma spline (1e-7..1e14 Msun; z = 0, 10, 100)',
          err<1.e-2,'max |ratio-1| = %.2e'%err)
    check('growth_eff(M, 0) == 1',maxdiff(cs.growth_eff(M,0.),np.ones(M.shape))==0.)
    nocut = ClassSigma(TABLE,hm,k_fs=None,window='TopHat')
    M8    = 4.*np.pi/3.*hm.OmegaM*hm.rhocrit0*(8.*Mpc/hm.h)**3
    print('    sigma_c(R = 8 Mpc/h, z = 0), top-hat, no cutoff: %.4f (CLASS sigma8 of all matter: %.4f)'
          %(nocut.sigma(M8,0.),float(t['sigma8'])))
    g = cs.growth_eff(1.e12*Msun,np.array([1.,3.,10.,30.]))/hm.growthD(np.array([1.,3.,10.,30.]))
    print('    growth_eff(1e12 Msun, z)/growthD(z) at z = 1, 3, 10, 30:',' '.join('%.4f'%v for v in g))
    assert_checks(n0)


# ----------------------------------------------------------------------------------------- main
TESTS = (test_default_unchanged, test_zacc_frame_reduces_to_default, test_class_mode_zmax3,
         test_class_mode_zmax7, test_class_mode_windows, test_errors_and_checks, test_vs_reference)

if __name__=='__main__':
    print('CLASS table: %s'%TABLE)
    t_all   = time.time()
    skipped = []
    for test in TESTS:
        n0 = len(FAILURES)
        try:
            test()
        except unittest.SkipTest as err:
            print('  SKIPPED: %s'%err)
            skipped.append(test.__name__)
        except Exception as err:
            if len(FAILURES)==n0:
                traceback.print_exc()
                FAILURES.append('%s: %r'%(test.__name__,err))
    print('\n(d) timings [s]')
    for key, val in TIMES.items():
        print('    %-62s %7.2f'%(key,val))
    print('    %-62s %7.2f'%('total',time.time()-t_all))
    print('\n%s%s'%('ALL PASSED' if not FAILURES else 'FAILED: %s'%FAILURES,
                    '' if not skipped else ' (skipped: %s)'%', '.join(skipped)))
    sys.exit(1 if FAILURES else 0)
