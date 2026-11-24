"""
sigma(M, z) from a CLASS table of the CDM power spectrum, for SASHIMI-C.

This module is imported lazily by ``sashimi_c.halo_model`` only when
``sigma_model='CLASS'``; the default (``sigma_model='Ludlow'``) never loads it.

Physics choices
---------------
* Field and threshold: the CDM-only density field delta_c (CLASS d_cdm,
  synchronous gauge) with the collapse threshold 1.686. Below the baryon Jeans
  scale the baryons stay smooth and delta_c grows as the CDM-only solution
  (~a^0.9), so the growth is scale dependent; it is taken directly from CLASS
  at the z nodes of the table (no D(z) rescaling of a z = 0 spectrum).
* Spectrum: P_c(k, z) = CLASS P_cdm(k, z) up to k_anchor; above it the
  small-scale CDM shape P ~ k^(n_s-4) ln^2(B k) (Hu & Sugiyama 1996), with ln B
  fitted at the highest z node between k_max/3 and k_max (~2.77 for B in Mpc)
  and the amplitude matched at k_anchor = k_max for z >= 7 and at
  k_anchor = 1e4 Mpc^-1 for z < 7. CLASS's tanh reionisation gives c_b^2 < 0 at
  6.8 < z < 8.2, which boosts delta_cdm at lower z, steeply with k (about 0.4 %
  at 2e4 Mpc^-1 and z = 0); at 1e4 the boost is <~ 0.05 % while the CDM
  transfer function has reached the asymptotic shape (at 3e3 Mpc^-1 it is still
  inside the Jeans transition, and an anchor there overestimates P by 2-2.7 %
  at k >= 1e4). The table must therefore reach k_max >= 1e4 Mpc^-1.
  The free-streaming cutoff exp[-(k/k_fs)^2] multiplies P (as in prompt_cusps).
* Windows (R_TH = (3M/4 pi rho_m)^(1/3)):
    'Sharp-k'  W = Theta(1 - k/k_s), k_s = c/R_TH; default c = 2.5 (Benson+13).
    'TopHat'   W = 3 (sin x - x cos x)/x^3, x = k R_TH (series below x = 1e-3).
    'Smooth-k' W = 1/[1 + (k R_TH/c)^beta]; default beta = 4.8, c = 3.3 (Leo+18).
  Only Sharp-k (or Smooth-k) should be used below the cutoff: with the top-hat
  sigma(M) flattens to a constant and leaves spurious haloes below the cutoff.
* EPS: sashimi's Na_calc takes the walk in the field at z_acc,
  S = sigma_c^2(M, z_acc), with the constant barrier 1.686. growth_eff gives the
  equivalent mass-dependent barrier for S(M) = sigma_c^2(M, 0) (the field at
  z = 0), B(M, z) = 1.686 sigma_c(M, 0)/sigma_c(M, z) = 1.686/growth_eff(M, z).

Interpolation
-------------
ln sigma(M, z_i) - ln D(z_i) (D = sashimi's growthD, a smooth scale-independent
reference that carries most of the z dependence) is tabulated on a fine
log10(M/Msun) grid at every z node and interpolated with a bicubic spline in
(log10 M, ln(1+z)). For Sharp-k, d sigma^2/dM = -Delta^2(k_s, z)/(3M) is
interpolated the same way (ln Delta^2 - 2 ln D). Outside the M grid and above
the highest z node the residual is held at its boundary value (so for
z > z_max of the table only D(z) carries the z dependence).

Units: sashimi multiplicative convention (Mpc = s = Msun = 1). The table is in
CLASS units (k in Mpc^-1, P in Mpc^3) and is converted on reading.

Building the table: make_class_table(out_path) runs CLASS (classy). With the
default kmax = 2e4 Mpc^-1 it takes ~35 min on one thread.
"""
import ast
import os
import time
import warnings
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RectBivariateSpline




# Planck 2018 parameters matched to sashimi's cosmology (h = 0.674, Omega_b = 0.049,
# Omega_m = 0.315 including one massive neutrino of 0.06 eV); identical to the
# orbit-sashimi prototype ps_class.PARAMS.
Z_GRID       = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0,
                200.0, 300.0, 500.0)
CLASS_PARAMS = {'h': 0.674, 'omega_b': 0.02226, 'omega_cdm': 0.1202, 'N_ur': 2.0328, 'N_ncdm': 1,
                'm_ncdm': 0.06, 'n_s': 0.9649, 'A_s': 2.0989e-9, 'tau_reio': 0.0544,
                'output': 'mPk,dTk', 'k_per_decade_for_pk': 20, 'z_max_pk': max(Z_GRID)}
K_PIVOT_MPC  = 0.05     # primordial pivot scale [Mpc^-1]

# extension of the table beyond its k range
K_ANCHOR_LOWZ_MPC = 1.e4    # amplitude anchor [Mpc^-1] for z < Z_ANCHOR_SWITCH (minimum k_max of a table)
Z_ANCHOR_SWITCH   = 7.      # z >= this: anchor at the last k of the table
LNB_FIT_RANGE     = 3.      # ln B is fitted on k_max/LNB_FIT_RANGE <= k <= k_max
K_GRID_TOL        = 0.9     # a table is accepted if its last k >= K_GRID_TOL*K_ANCHOR_LOWZ_MPC (CLASS k grid)

# check of the table's cosmology against sashimi's
COSMO_RTOL        = 1.e-3   # relative tolerance on h, Omega_m and Omega_b
EV_PER_OMEGA_NU   = 93.14   # sum m_nu [eV] = 93.14 omega_nu
DEFAULT_C         = {'Sharp-k': 2.5, 'Smooth-k': 3.3, 'TopHat': 1.}
WINDOWS           = tuple(DEFAULT_C)




def make_class_table(out_path, kmax=2.e4, z_grid=Z_GRID, params=CLASS_PARAMS, omp_threads=None):
    """
    Run CLASS (classy) and write the CDM power-spectrum table read by ClassSigma.

    -----
    Input
    -----
    out_path:    Output .npz file.
    kmax:        P_k_max_1/Mpc of CLASS [Mpc^-1], at least 1e4 (the high-k extension is anchored there
                 at z < 7). kmax = 2e4 (default) takes ~35 min on one thread.
    z_grid:      Redshift nodes (default: the 18 nodes 0, 0.5, ..., 500); must increase from z = 0.
    params:      CLASS parameters; must contain h, omega_b, omega_cdm, A_s and n_s.
    omp_threads: None (default): the OpenMP threads of CLASS follow OMP_NUM_THREADS of the shell.
                 An integer sets OMP_NUM_THREADS during this call (restored afterwards); it only takes
                 effect if OpenMP has not been started in this process yet.

    ------
    Output
    ------
    The npz keys are k_Mpc [Mpc^-1], z, pcdm and pcb [Mpc^3] (rows: z nodes), tcdm (d_cdm transfer
    function), sigma8, runtime_s, kmax and params (repr of the CLASS parameter dict).
    P_cdm(k, z) = 2 pi^2/k^3 A_s (k/k_p)^(n_s-1) d_cdm(k, z)^2 with k_p = 0.05 Mpc^-1; P_cb uses
    (omega_cdm d_cdm + omega_b d_b)/(omega_cdm + omega_b).
    """
    if not kmax>=K_ANCHOR_LOWZ_MPC:
        raise ValueError("kmax must be >= %.0e Mpc^-1 (the Jeans transition must be resolved), not %r"
                         %(K_ANCHOR_LOWZ_MPC,kmax))
    z_grid = tuple(float(z) for z in z_grid)
    if z_grid[0]!=0. or np.any(np.diff(z_grid)<=0.):
        raise ValueError("z_grid must increase from z = 0")
    omp_old = os.environ.get('OMP_NUM_THREADS')
    if omp_threads is not None:
        os.environ['OMP_NUM_THREADS'] = str(int(omp_threads))
    try:
        table = _run_class(kmax,z_grid,params)
    finally:
        if omp_threads is not None:
            if omp_old is None:
                os.environ.pop('OMP_NUM_THREADS',None)
            else:
                os.environ['OMP_NUM_THREADS'] = omp_old
    np.savez(out_path, **table)
    return out_path


def _run_class(kmax, z_grid, params):
    """ CLASS run of make_class_table; returns the npz content as a dict. """
    from classy import Class
    par    = dict(params)
    par['z_max_pk'] = max(z_grid)
    cosmo  = Class()
    cosmo.set(dict(par, **{'P_k_max_1/Mpc': kmax}))
    t0     = time.time()
    cosmo.compute()
    dt     = time.time()-t0
    tr0    = cosmo.get_transfer(0.)
    k      = np.asarray(tr0['k (h/Mpc)'])*par['h']
    prim   = par['A_s']*(k/K_PIVOT_MPC)**(par['n_s']-1.)*2.*np.pi**2/k**3
    wc, wb = par['omega_cdm'], par['omega_b']
    tcdm   = np.empty((len(z_grid),k.size))
    pcb    = np.empty((len(z_grid),k.size))
    for i, z in enumerate(z_grid):
        tr      = cosmo.get_transfer(z)
        tcdm[i] = np.asarray(tr['d_cdm'])
        tcb     = (wc*tcdm[i]+wb*np.asarray(tr['d_b']))/(wc+wb)
        pcb[i]  = prim*tcb**2
    sigma8 = cosmo.sigma8()
    cosmo.struct_cleanup()
    return dict(k_Mpc=k, z=np.array(z_grid), pcdm=prim*tcdm**2, pcb=pcb, tcdm=tcdm, sigma8=sigma8,
                runtime_s=dt, kmax=kmax, params=np.str_(repr(par)))




class ClassSigma:
    """ sigma(M, z) of the CDM field from a CLASS table, with a free-streaming cutoff and a window. """


    def __init__(self, table_path, cosmo, k_fs=None, window='Sharp-k', c=None, beta=4.8,
                 logM_min=-12., logM_max=17., N_M_per_dec=40, N_k_per_dec=200, lnB=None):
        """
        -----
        Input
        -----
        table_path:  npz written by make_class_table (keys k_Mpc, z, pcdm, params, ...).
        cosmo:       sashimi cosmology/halo_model instance (units, OmegaM, rhocrit0, growthD).
        k_fs:        Free-streaming wavenumber in sashimi units (e.g. 1.06e6/Mpc); None: no cutoff.
        window:      'Sharp-k' (default), 'TopHat' or 'Smooth-k'.
        c:           Mass-assignment constant: k_s = c/R_TH (Sharp-k), R_s = R_TH/c (Smooth-k).
                     None: 2.5 for Sharp-k (Benson+13), 3.3 for Smooth-k (Leo+18); unused for TopHat.
        beta:        Smooth-k slope (default 4.8, Leo+18).
        logM_min/logM_max: log10(M/Msun) range of the tabulated sigma (held constant outside).
        N_M_per_dec, N_k_per_dec: Grid densities in log10 M and log10 k.
        lnB:         ln(B/Mpc) of the high-k extension; None: fitted at the highest z node.
        """
        if window not in WINDOWS:
            raise ValueError("window must be one of %s, not %r"%(WINDOWS,window))
        for name, val in (('c',c),('beta',beta),('k_fs',k_fs)):
            if val is not None and not float(val)>0.:
                raise ValueError("%s must be positive, not %r"%(name,val))
        if c is not None and not np.isfinite(c):
            raise ValueError("c must be finite, not %r"%(c,))
        self.cosmo      = cosmo
        self.Mpc        = cosmo.Mpc
        self.Msun       = cosmo.Msun
        self.rhom       = cosmo.OmegaM*cosmo.rhocrit0
        self.window     = window
        self.c          = DEFAULT_C[window] if c is None else float(c)
        self.beta       = float(beta)
        self.k_fs       = k_fs
        self.table_path = table_path

        self._load_table(table_path)
        self.lnB = self._fit_lnB() if lnB is None else float(lnB)
        self._set_anchors()

        self.x_M   = np.linspace(logM_min,logM_max,int(round((logM_max-logM_min)*N_M_per_dec))+1)
        self.y_z   = np.log1p(self.z_nodes)
        self.x_min = self.x_M[0]
        self.x_max = self.x_M[-1]
        self.y_max = self.y_z[-1]
        lnsig2, lnd2s = self._node_sigma2(10.**self.x_M*self.Msun,N_k_per_dec)
        lnD           = np.log(self.cosmo.growthD(self.z_nodes))
        self._spl_sig = RectBivariateSpline(self.x_M,self.y_z,0.5*lnsig2-lnD,kx=3,ky=3,s=0)
        if self.window=='Sharp-k':
            self._spl_d2s = RectBivariateSpline(self.x_M,self.y_z,lnd2s-2.*lnD,kx=3,ky=3,s=0)


    def _load_table(self, table_path):
        if not os.path.isfile(table_path):
            raise FileNotFoundError("CLASS table %r not found: build it with class_sigma.make_class_table(%r) "
                                    "(needs classy; ~35 min on one thread)"%(table_path,table_path))
        with np.load(table_path) as f:
            t = {key: f[key] for key in f.files}
        params        = ast.literal_eval(str(t['params'])) if 'params' in t else {}
        self.n_s      = float(params.get('n_s',CLASS_PARAMS['n_s']))
        self.z_nodes  = np.asarray(t['z'],dtype=float)
        self.k_tab    = np.asarray(t['k_Mpc'],dtype=float)*self.Mpc**-1
        p_tab         = np.asarray(t['pcdm'],dtype=float)*self.Mpc**3
        self.lnk_tab  = np.log(self.k_tab*self.Mpc)
        # ln Delta^2 = ln[k^3 P/(2 pi^2)] on the table (linear in ln k <=> linear in ln P)
        self.lnd2_tab = np.log(self.k_tab**3*p_tab/(2.*np.pi**2))
        if self.z_nodes[0]!=0. or np.any(np.diff(self.z_nodes)<=0.):
            raise ValueError("the z nodes of the table must increase from z = 0")
        if p_tab.shape!=(len(self.z_nodes),len(self.k_tab)) or np.any(np.diff(self.lnk_tab)<=0.):
            raise ValueError("the table must have pcdm of shape (len(z), len(k_Mpc)) on increasing k_Mpc")
        if self.k_tab[-1]*self.Mpc<K_GRID_TOL*K_ANCHOR_LOWZ_MPC:
            # the k^(n_s-4) ln^2(B k) extension has no Jeans suppression (sigma 3-30 % too high)
            raise ValueError("the table reaches k = %.3g Mpc^-1 only; it must reach %.0e Mpc^-1 (above the "
                             "baryon Jeans transition)"%(self.k_tab[-1]*self.Mpc,K_ANCHOR_LOWZ_MPC))
        self._check_cosmology(params)


    def _check_cosmology(self, params):
        """ Warn if h, Omega_m (with massive neutrinos) or Omega_b of the table differ from sashimi's. """
        try:
            h, wb, wc = float(params['h']), float(params['omega_b']), float(params['omega_cdm'])
        except (KeyError, TypeError, ValueError):
            warnings.warn("the CLASS table %r has no h, omega_b and omega_cdm in params: its cosmology "
                          "is not checked against sashimi's"%(self.table_path,))
            return
        m_nu  = params.get('m_ncdm',0.) if int(params.get('N_ncdm',0))>0 else 0.
        m_nu  = sum(float(v) for v in str(m_nu).split(','))
        table = {'h': h, 'Omega_m': (wb+wc+m_nu/EV_PER_OMEGA_NU)/h**2, 'Omega_b': wb/h**2}
        ref   = {'h': self.cosmo.h, 'Omega_m': self.cosmo.OmegaM, 'Omega_b': self.cosmo.OmegaB}
        bad   = ['%s = %.5g (sashimi: %.5g)'%(key,table[key],ref[key]) for key in table
                 if abs(table[key]/ref[key]-1.)>COSMO_RTOL]
        if bad:
            warnings.warn("the cosmology of the CLASS table %r differs from sashimi's: %s"
                          %(self.table_path,', '.join(bad)))


    def _fit_lnB(self):
        """ ln(B/Mpc) of P ~ k^(n_s-4) ln^2(B k), fitted at the highest z node on k_max/3 <= k <= k_max. """
        sel         = self.lnk_tab>=self.lnk_tab[-1]-np.log(LNB_FIT_RANGE)
        lnk         = self.lnk_tab[sel]
        y           = np.sqrt(np.exp(self.lnd2_tab[-1,sel]-(self.n_s-1.)*lnk))
        slope, icpt = np.polyfit(lnk,y,1)
        return icpt/slope


    def _set_anchors(self):
        """ Amplitude of the extension: ln A in Delta^2 = A (k Mpc)^(n_s-1) [ln(k Mpc) + ln B]^2. """
        lnk_low         = np.log(min(K_ANCHOR_LOWZ_MPC,self.k_tab[-1]*self.Mpc))
        ja_low          = int(np.argmin(np.abs(self.lnk_tab-lnk_low)))
        ja_high         = len(self.lnk_tab)-1
        self.ja         = np.where(self.z_nodes>=Z_ANCHOR_SWITCH,ja_high,ja_low)
        self.lnk_anchor = self.lnk_tab[self.ja]
        self.lnA        = self.lnd2_tab[np.arange(len(self.z_nodes)),self.ja] \
                              -(self.n_s-1.)*self.lnk_anchor-2.*np.log(self.lnk_anchor+self.lnB)


    def lnDelta2(self, k, iz):
        """ ln Delta_c^2(k, z_iz), Delta^2 = k^3 P_c/(2 pi^2), extended and cut off; k in sashimi units. """
        lnk = np.log(np.asarray(k,dtype=float)*self.Mpc)
        tab = np.interp(lnk,self.lnk_tab,self.lnd2_tab[iz])
        ext = self.lnA[iz]+(self.n_s-1.)*lnk+2.*np.log(np.abs(lnk+self.lnB))
        out = np.where(lnk<=self.lnk_anchor[iz],tab,ext)
        if self.k_fs is not None:
            out = out-(np.exp(lnk)/(self.k_fs*self.Mpc))**2
        return out


    def power_spectrum(self, k, iz):
        """ P_c(k, z_iz) in sashimi units (extended and cut off); k in sashimi units. """
        k = np.asarray(k,dtype=float)
        return np.exp(self.lnDelta2(k,iz))*2.*np.pi**2/k**3


    def _window2(self, x):
        if self.window=='TopHat':
            xs = np.maximum(x,1.e-3)
            w  = 3.*(np.sin(xs)-xs*np.cos(xs))/xs**3
            return np.where(x<1.e-3,1.-x**2/10.+x**4/280.,w)**2
        return 1./(1.+x**self.beta)**2


    def _node_sigma2(self, M, N_k_per_dec):
        """ ln sigma^2(M, z_i) on the M grid at every z node; for Sharp-k also ln Delta^2(k_s, z_i). """
        R_TH = np.cbrt(3.*M/(4.*np.pi*self.rhom))
        k_lo = self.k_tab[0]
        k_hi = 100.*max(self.c,1.)/R_TH[0]
        if self.k_fs is not None:
            k_hi = min(k_hi,10.*self.k_fs)
        k_hi = max(k_hi,self.c/R_TH[0])
        N_k  = int(np.ceil(np.log10(k_hi/k_lo)*N_k_per_dec))+1
        lnk  = np.linspace(np.log(k_lo),np.log(k_hi),N_k)
        k    = np.exp(lnk)
        N_z  = len(self.z_nodes)
        lnsig2 = np.empty((len(M),N_z))
        lnd2s  = np.empty((len(M),N_z))
        if self.window=='Sharp-k':
            k_s = self.c/R_TH
            for iz in range(N_z):
                cum          = cumulative_trapezoid(np.exp(self.lnDelta2(k,iz)),x=lnk,initial=0.)
                lnsig2[:,iz] = np.log(np.interp(np.log(k_s),lnk,cum))
                lnd2s[:,iz]  = self.lnDelta2(k_s,iz)
        else:
            R   = R_TH if self.window=='TopHat' else R_TH/self.c
            W2  = self._window2(R[:,np.newaxis]*k[np.newaxis,:])
            wts = np.full(N_k,lnk[1]-lnk[0])
            wts[[0,-1]] *= 0.5
            for iz in range(N_z):
                lnsig2[:,iz] = np.log(W2@(np.exp(self.lnDelta2(k,iz))*wts))
        return lnsig2, lnd2s


    def _xy(self, M, z):
        M, z = np.broadcast_arrays(np.asarray(M,dtype=float),np.asarray(z,dtype=float))
        x    = np.clip(np.log10(M/self.Msun),self.x_min,self.x_max)
        y    = np.clip(np.log1p(z),0.,self.y_max)
        return M, z, x, y


    def _ev(self, spl, x, y, dx=0):
        return spl.ev(x.ravel(),y.ravel(),dx=dx).reshape(x.shape)


    def sigma(self, M, z):
        """ sigma_c(M, z), with the true scale-dependent growth. """
        M, z, x, y = self._xy(M,z)
        return (np.exp(self._ev(self._spl_sig,x,y))*self.cosmo.growthD(z))[()]


    def growth_eff(self, M, z):
        """ sigma_c(M, z)/sigma_c(M, 0). """
        M, z, x, y = self._xy(M,z)
        lnr = self._ev(self._spl_sig,x,y)-self._ev(self._spl_sig,x,np.zeros_like(y))
        return (np.exp(lnr)*self.cosmo.growthD(z)/self.cosmo.growthD(0.))[()]


    def dsdm(self, M, z):
        """ d sigma_c^2/dM at z (negative). Sharp-k: -Delta^2(k_s, z)/(3M); otherwise from the spline. """
        M, z, x, y = self._xy(M,z)
        D = self.cosmo.growthD(z)
        if self.window=='Sharp-k':
            return (-np.exp(self._ev(self._spl_d2s,x,y))*D**2/(3.*M))[()]
        lns  = self._ev(self._spl_sig,x,y)
        dlns = self._ev(self._spl_sig,x,y,dx=1)
        return (2.*np.exp(2.*lns)*D**2*dlns/(M*np.log(10.)))[()]
