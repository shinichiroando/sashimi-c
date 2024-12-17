import numpy as np
from scipy import integrate
from scipy import interpolate
from scipy import optimize
from scipy import special
from scipy.integrate import simpson
from scipy.integrate import odeint
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.special import erf
from numpy.polynomial.hermite import hermgauss
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)




def memoize_with_pickle(cache_dir="cache"):
    """
    A decorator to cache the results of an instance method or function in a pickle file.
    It ignores 'self' when creating the cache key, so the cache is shared across all instances.
    """
    import os
    import pickle
    import hashlib

    os.makedirs(cache_dir, exist_ok=True)

    def decorator(func):
        def wrapper(*args, **kwargs):
            # If this is a method, args[0] is 'self'.
            # We'll exclude it from the key to allow caching across instances.
            key_source = (func.__name__, args[1:], tuple(sorted(kwargs.items())))
            key = hashlib.md5(pickle.dumps(key_source)).hexdigest()
            cache_file = os.path.join(cache_dir, f"{key}.pkl")

            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)
                return result
            else:
                result = func(*args, **kwargs)
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
                return result
        return wrapper
    return decorator




class units_and_constants:

    
    def __init__(self):
        self.Mpc      = 1.
        self.kpc      = self.Mpc/1000.
        self.pc       = self.kpc/1000.
        self.cm       = self.pc/3.086e18
        self.km       = 1.e5*self.cm
        self.s        = 1.
        self.yr       = 3.15576e7*self.s
        self.Msun     = 1.
        self.gram     = self.Msun/1.988e33
        self.c        = 2.9979e10*self.cm/self.s
        self.G        = 6.6742e-8*self.cm**3/self.gram/self.s**2



        
class cosmology(units_and_constants):
    
    
    def __init__(self):
        units_and_constants.__init__(self)
        self.OmegaB        = 0.049
        self.OmegaM        = 0.315
        self.OmegaC        = self.OmegaM-self.OmegaB
        self.OmegaL        = 1.-self.OmegaM
        self.h             = 0.674
        self.H0            = self.h*100*self.km/self.s/self.Mpc 
        self.rhocrit0      = 3*self.H0**2/(8.0*np.pi*self.G)


    def g(self, z):
        return self.OmegaM*(1.+z)**3+self.OmegaL


    def Hubble(self, z):
        return self.H0*np.sqrt(self.OmegaM*(1.+z)**3+self.OmegaL)

    
    def rhocrit(self, z):
        return 3.*self.Hubble(z)**2/(np.pi*8.0*self.G)


    def growthD(self, z):
        Omega_Lz = self.OmegaL/(self.OmegaL+self.OmegaM*(1.+z)**3)
        Omega_Mz = 1-Omega_Lz
        phiz     = Omega_Mz**(4./7.)-Omega_Lz+(1.+Omega_Mz/2.0)*(1.+Omega_Lz/70.0)
        phi0     = self.OmegaM**(4./7.)-self.OmegaL+(1.+self.OmegaM/2.0)*(1.+self.OmegaL/70.0)
        return (Omega_Mz/self.OmegaM)*(phi0/phiz)/(1.+z)

        
    def dDdz(self, z):
        def dOdz(z):
            return -self.OmegaL*3*self.OmegaM*(1+z)**2*(self.OmegaL+self.OmegaM*(1+z)**3.)**-2
        Omega_Lz = self.OmegaL*pow(self.OmegaL+self.OmegaM*pow(self.h,-2)*pow(1+z,3),-1)
        Omega_Mz = 1-Omega_Lz
        phiz     = Omega_Mz**(4./7.)-Omega_Lz+(1+Omega_Mz/2.)*(1+Omega_Lz/70.)
        phi0     = self.OmegaM**(4./7.)-self.OmegaL+(1+self.OmegaM/2.)*(1+self.OmegaL/70.)
        dphidz   = dOdz(z)*(-4./7.*Omega_Mz**(-3.0/7.0)+(Omega_Mz-Omega_Lz)/140.+1./70.-3./2.)
        return (phi0/self.OmegaM)*(-dOdz(z)/(phiz*(1+z))-Omega_Mz*(dphidz*(1+z)+phiz)/phiz**2/(1+z)**2)

        


class halo_model(cosmology):

    
    def __init__(self):
        cosmology.__init__(self)

        
    def xi(self, M):
        return (M/((1.e10*self.Msun)/self.h))**-1

    
    def sigmaMz(self, M, z):    
        """ Ludlow et al. (2016) """
        return self.growthD(z)*22.26*self.xi(M)**0.292/(1.+1.53*self.xi(M)**0.275+3.36*self.xi(M)**0.198)


    def deltac_func(self, z):
        return 1.686/self.growthD(z)

    
    def s_func(self, M):
        return self.sigmaMz(M,0)**2


    def fc(self, x):
        return np.log(1+x)-x*pow(1+x,-1)

    
    def Delc(self, x):
        return 18.*np.pi**2+82.*x-39.*x**2

    
    def conc200(self,M200,z): 
        """ Correa et al. (2015) """
        alpha_cMz_1 = 1.7543-0.2766*(1.+z)+0.02039*(1.+z)**2
        beta_cMz_1  = 0.2753+0.00351*(1.+z)-0.3038*(1.+z)**0.0269
        gamma_cMz_1 = -0.01537+0.02102*(1.+z)**-0.1475
        c_Mz_1      = np.power(10.,alpha_cMz_1+beta_cMz_1*np.log10(M200/self.Msun) \
                               *(1+gamma_cMz_1*np.log10(M200/self.Msun)**2))
        alpha_cMz_2 = 1.3081-0.1078*(1.+z)+0.00398*(1.+z)**2
        beta_cMz_2  = 0.0223-0.0944*(1.+z)**-0.3907
        c_Mz_2      = pow(10,alpha_cMz_2+beta_cMz_2*np.log10(M200/self.Msun))
        return np.where(z<=4.,c_Mz_1,c_Mz_2)

    
    def Mvir_from_M200(self, M200, z):
        gz    = self.g(z)
        c200  = self.conc200(M200,z)
        r200  = (3.0*M200/(4*np.pi*200*self.rhocrit0*gz))**(1./3.)
        rs    = r200/c200
        fc200 = self.fc(c200)
        rhos  = M200/(4*np.pi*rs**3*fc200)
        Dc    = self.Delc(self.OmegaM*(1.+z)**3/self.g(z)-1.)
        rvir  = optimize.fsolve(lambda r: 3.*(rs/r)**3*self.fc(r/rs)*rhos-Dc*self.rhocrit0*gz,r200)
        Mvir  = 4*np.pi*rs**3*rhos*self.fc(rvir/rs)
        return Mvir

    
    def Mvir_from_M200_fit(self, M200, z):
        a1 = 0.5116
        a2 = -0.4283
        a3 = -3.13e-3
        a4 = -3.52e-5
        Oz = self.OmegaM*(1.+z)**3/self.g(z)
        def ffunc(x):
            return np.power(x,3.0)*(np.log(1.0+1.0/x)-1.0/(1.0+x))
        def xfunc(f):
            p = a2 + a3*np.log(f) + a4*np.power(np.log(f),2.0)
            return np.power(a1*np.power(f,2.0*p)+(3.0/4.0)**2,-0.5)+2.0*f
        return self.Delc(Oz-1)/200.0*M200 \
            *np.power(self.conc200(M200,z) \
            *xfunc(self.Delc(Oz-1)/200.0*ffunc(1.0/self.conc200(M200,z))),-3.0)

    
    def Mzi(self, M0, z):
        a   = 1.686*np.sqrt(2./np.pi)*self.dDdz(0)+1.
        zf  = -0.0064*np.log10(M0/self.Msun)**2+0.0237*np.log10(M0/self.Msun)+1.8837
        q   = 4.137/zf**0.9476
        fM0 = (self.sigmaMz(M0/q,0)**2-self.sigmaMz(M0,0)**2)**-0.5
        return M0*np.power(1.+z,a*fM0)*np.exp(-fM0*z)

    
    def Mzzi(self, M0, z, zi):
        Mzi0  = self.Mzi(M0,zi)
        zf    = -0.0064*np.log10(M0/self.Msun)**2+0.0237*np.log10(M0/self.Msun)+1.8837
        q     = 4.137/zf**0.9476
        fMzi  = (self.sigmaMz(Mzi0/q,zi)**2-self.sigmaMz(Mzi0,zi)**2)**-0.5
        alpha = fMzi*(1.686*np.sqrt(2.0/np.pi)/self.growthD(zi)**2*self.dDdz(zi)+1.)
        beta  = -fMzi
        return Mzi0*np.power(1.+z-zi,alpha)*np.exp(beta*(z-zi))

    
    def dMdz(self, M0, z, zi):
        Mzi0    = self.Mzi(M0,zi)
        zf      = -0.0064*np.log10(M0/self.Msun)**2+0.0237*np.log10(M0/self.Msun)+1.8837
        q       = 4.137/zf**0.9476
        fMzi    = (self.sigmaMz(Mzi0/q,zi)**2-self.sigmaMz(Mzi0,zi)**2)**-0.5
        alpha   = fMzi*(1.686*np.sqrt(2./np.pi)/self.growthD(zi)**2*self.dDdz(zi)+1)
        beta    = -fMzi
        Mzzidef = Mzi0*(1.+z-zi)**alpha*np.exp(beta*(z-zi))
        Mzzivir = self.Mvir_from_M200_fit(Mzzidef,z)
        return (beta+alpha/(1.+z-zi))*Mzzivir

    
    def dsdm(self,M,z):  
        """ Ludlow et al. (2016) """
        s         = self.sigmaMz(M,z)**2
        dsdsigma  = 2.*self.sigmaMz(M,z)
        dxidm     = -1.e10*self.Msun/self.h/M**2
        dsigmadxi = self.sigmaMz(M,z)*(0.292/self.xi(M)-(0.275*1.53*self.xi(M)**-0.725+0.198*3.36* \
            self.xi(M)**-0.802)/(1.+1.53*self.xi(M)**0.275+3.36*self.xi(M)**0.198))
        return dsdsigma*dsigmadxi*dxidm

    
    def dlogSdlogM(self,M,z):  
        """ Ludlow et al. (2016) """
        s         = self.sigmaMz(M,z)**2
        dsdsigma  = 2.*self.sigmaMz(M,z)
        dxidm     = -1.e10*self.Msun/self.h/M**2
        dsigmadxi = self.sigmaMz(M,z)*(0.292*pow(self.xi(M),-1)-(0.275*1.53*pow(self.xi(M),-0.725)+0.198*3.36* \
            pow(self.xi(M),-0.802))*pow(1.+1.53*pow(self.xi(M),0.275)+3.36*pow(self.xi(M),0.198),-1))
        return (M/s)*dsdsigma*dsigmadxi*dxidm




class TidalStrippingSolver(halo_model):
    """ Solve the tidal stripping equation for a given subhalo. """
    
    def __init__(self, M0, z_min=0.0, z_max=7.0, n_z_interp=64):
        """ Initial function of the class. 
        
        -----
        Input
        -----
        M0: Mass of the host halo defined as M_{200} (200 times critial density) at *z = 0*.
        (Optional) z_min:          Minimum redshift to end the calculation of evolution to. (default: 0.)
        (Optional) z_max:          Maximum redshift to start the calculation of evolution from. (default: 7.)
        (Optional) n_z_interp:     Number of redshifts to calculate epsilon functions. (default: 64)
        """
        halo_model.__init__(self)
        self.z_min       = z_min
        self.z_max       = z_max
        self.n_z_interp  = n_z_interp
        self.M0          = M0


    @property
    def M0(self):
        return self._M0


    @M0.setter
    def M0(self, value):
        self._M0 = value
        self.reset_interpolation(
            z_max=self.z_max, 
            z_min=self.z_min,
            n_z=self.n_z_interp)

    
    def reset_interpolation(self, z_max, z_min, n_z):
        """ Reset interpolation for epsilon functions. 
        
        This function is called when the mass of the host
        halo is changed.

        -----
        Input
        -----
        za_max: float
            Maximum redshift to start the calculation of evolution from.
        z_min: float
            Minimum redshift to end the calculation of evolution to.
        n_z: int
            Number of redshifts to calculate epsilon functions.
        """
        _z, _eps_0 = self._eps_0(z_max, z_min, n_z)
        _, _eps_10, _eps_11 = self._eps_1(z_max, z_min, n_z)
        _, _eps_20, _eps_21, _eps_22 = self._eps_2(z_max, z_min, n_z)
        _, _eps_30, _eps_31, _eps_32, _eps_33 = self._eps_3(z_max, z_min, n_z)
        # get the interpolation functions as indefinite integrals
        self._eps_0_interp = lambda z: np.interp(z, _z[::-1], _eps_0[::-1])
        self._eps_10_interp = lambda z: np.interp(z, _z[::-1], _eps_10[::-1])
        self._eps_11_interp = lambda z: np.interp(z, _z[::-1], _eps_11[::-1])
        self._eps_20_interp = lambda z: np.interp(z, _z[::-1], _eps_20[::-1])
        self._eps_21_interp = lambda z: np.interp(z, _z[::-1], _eps_21[::-1])
        self._eps_22_interp = lambda z: np.interp(z, _z[::-1], _eps_22[::-1])
        self._eps_30_interp = lambda z: np.interp(z, _z[::-1], _eps_30[::-1])
        self._eps_31_interp = lambda z: np.interp(z, _z[::-1], _eps_31[::-1])
        self._eps_32_interp = lambda z: np.interp(z, _z[::-1], _eps_32[::-1])
        self._eps_33_interp = lambda z: np.interp(z, _z[::-1], _eps_33[::-1])
        # define the epsilon functions as definite integrals from za to z
        self.eps_0 = lambda _za, _z: self._eps_0_interp(_z) - self._eps_0_interp(_za)
        self.eps_10 = lambda _za, _z: self._eps_10_interp(_z) - self._eps_10_interp(_za)
        self.eps_11 = lambda _za, _z: self._eps_11_interp(_z) - self._eps_11_interp(_za)
        self.eps_20 = lambda _za, _z: self._eps_20_interp(_z) - self._eps_20_interp(_za)
        self.eps_21 = lambda _za, _z: self._eps_21_interp(_z) - self._eps_21_interp(_za)
        self.eps_22 = lambda _za, _z: self._eps_22_interp(_z) - self._eps_22_interp(_za)
        self.eps_30 = lambda _za, _z: self._eps_30_interp(_z) - self._eps_30_interp(_za)
        self.eps_31 = lambda _za, _z: self._eps_31_interp(_z) - self._eps_31_interp(_za)
        self.eps_32 = lambda _za, _z: self._eps_32_interp(_z) - self._eps_32_interp(_za)
        self.eps_33 = lambda _za, _z: self._eps_33_interp(_z) - self._eps_33_interp(_za)


    def Mzvir(self,z):
        Mz200 = self.Mzzi(self.M0,z,0.)
        Mvir = self.Mvir_from_M200_fit(Mz200,z)
        return Mvir


    def AMz(self,z):
        log10a = (-0.0003*np.log10(self.Mzvir(z)/self.Msun)+0.02)*z \
                        +(0.011*np.log10(self.Mzvir(z)/self.Msun)-0.354)
        return 10.**log10a


    def zetaMz(self,z):
        return (0.00012*np.log10(self.Mzvir(z)/self.Msun)-0.0033)*z \
                    +(-0.0011*np.log10(self.Mzvir(z)/self.Msun)+0.026)


    def tdynz(self,z):
        Oz_z = self.OmegaM*(1.+z)**3/self.g(z)
        return 1.628/self.h*(self.Delc(Oz_z-1.)/178.0)**-0.5/(self.Hubble(z)/self.H0)*1.e9*self.yr


    def msolve(self,m, z):
        return self.AMz(z)*(m/self.tdynz(z))*(m/self.Mzvir(z))**self.zetaMz(z)/(self.Hubble(z)*(1+z))


    def subhalo_mass_stripped_odeint(self, ma, za, z0, **kwargs):
        zcalc = np.linspace(za,z0,100)
        sol = odeint(self.msolve,ma,zcalc,**kwargs)
        return sol[-1]

    
    # Functions to calculate perturbative corrections to the subhalo mass function
    def Phi(self,z):
        """ subhalo stripping factor assuming zetaMz(z) = 0.
        The stripping rate dm/dt is given by
          dm/dt(z) = m(z) * Phi(z) * (m(z)/Mzvir(z))**zetaMz(z)
        """
        return self.AMz(z)/self.tdynz(z)/self.Hubble(z)/(1+z)


    # @memoize_with_pickle()
    def _eps_0(self,za,z,n_z=64):
        """ calculate epsilon0.
        
        Returns
        -------
        _z : array
            redshift array
        eps0 : array
            epsilon0 array.
        """
        _z = np.linspace(za,z,n_z)
        Phi_z = self.Phi(_z)
        return _z, cumulative_trapezoid(Phi_z,x=_z,initial=0)
    

    # @memoize_with_pickle()
    def _eps_1(self,za,z,n_z=64):
        """ calculate the first order correction.

        The first order correction epsilon_1 is given by the following equation:
            epsilon_1 = epsilon_10 + epsilon_11 * ln_ma
        
        Returns
        -------
        _z : array
            redshift array
        eps10 : array
            epsilon10 array.
        eps11 : array
            epsilon11 array.
        """
        _z, eps_0 = self._eps_0(za,z,n_z)
        Phi_z = self.Phi(_z)
        zeta_z = self.zetaMz(_z)
        ln_Mvir_z = np.log(self.Mzvir(_z))
        integrand_10 = Phi_z * (eps_0 - ln_Mvir_z) * zeta_z
        integrand_11 = Phi_z * zeta_z
        integral_10 = cumulative_trapezoid(integrand_10,x=_z,initial=0)
        integral_11 = cumulative_trapezoid(integrand_11,x=_z,initial=0)
        return _z, integral_10, integral_11
    

    # @memoize_with_pickle()
    def _eps_2(self,za,z,n_z=64):
        """ calculate the second order correction.

        The second order correction epsilon_2 is given by the following equation:
            epsilon_2 = epsilon_20 + epsilon_21 * ln_ma + epsilon_22 * ln_ma^2
        
        Returns
        -------
        _z : array
            redshift array
        eps20 : array
            epsilon20 array.
        eps21 : array
            epsilon21 array.
        eps22 : array
            epsilon22 array.
        """
        _z, eps_0 = self._eps_0(za,z,n_z)
        _, eps_10, eps_11 = self._eps_1(za,z,n_z)
        Phi_z = self.Phi(_z)
        zeta_z = self.zetaMz(_z)
        ln_Mvir_z = np.log(self.Mzvir(_z))
        integrand_20 = Phi_z * zeta_z **2 * (eps_0 - ln_Mvir_z)**2 /2 + Phi_z * zeta_z * eps_10
        integrand_21 = Phi_z * zeta_z**2 * (eps_0 - ln_Mvir_z) + Phi_z * zeta_z * eps_11
        integrand_22 = Phi_z * zeta_z**2 / 2
        integral_20 = cumulative_trapezoid(integrand_20,x=_z,initial=0)
        integral_21 = cumulative_trapezoid(integrand_21,x=_z,initial=0)
        integral_22 = cumulative_trapezoid(integrand_22,x=_z,initial=0)
        return _z, integral_20, integral_21, integral_22


    # @memoize_with_pickle()
    def _eps_3(self, za, z, n_z=64):
        """ calculate the third order correction.

        The third order correction epsilon_3 is given by the following equation:
            epsilon_3 = epsilon_30 + epsilon_31 * ln_ma + epsilon_32 * ln_ma^2 + epsilon_33 * ln_ma^3
        
        Returns
        -------
        _z : array
            redshift array
        eps30 : array
            epsilon30 array.
        eps31 : array
            epsilon31 array.
        eps32 : array
            epsilon32 array.
        eps33 : array
            epsilon33 array.
        """
        _z, eps_0 = self._eps_0(za, z, n_z)
        _, eps_10, eps_11 = self._eps_1(za, z, n_z)
        _, eps_20, eps_21, eps_22 = self._eps_2(za, z, n_z)
        Phi_z = self.Phi(_z)
        zeta_z = self.zetaMz(_z)
        ln_Mvir_z = np.log(self.Mzvir(_z))
        integrand_30 = (Phi_z * (eps_0 - ln_Mvir_z)**3 * zeta_z**3 / 6.0
                        + Phi_z * (eps_0 - ln_Mvir_z) * eps_10 * (zeta_z**2)
                        + Phi_z * eps_20 * zeta_z)
        integrand_31 = (Phi_z * (eps_0 - ln_Mvir_z)**2 * (zeta_z**3) / 2.0
                        + Phi_z * eps_10 * (zeta_z**2)
                        + Phi_z * eps_21 * zeta_z
                        + Phi_z * (eps_0 - ln_Mvir_z) * eps_11 * (zeta_z**2))
        integrand_32 = (Phi_z * (eps_0 - ln_Mvir_z) * (zeta_z**3) / 2.0
                        + Phi_z * eps_11 * (zeta_z**2)
                        + Phi_z * eps_22 * zeta_z)
        integrand_33 = Phi_z * (zeta_z**3) / 6.0
        integral_30 = cumulative_trapezoid(integrand_30, x=_z, initial=0)
        integral_31 = cumulative_trapezoid(integrand_31, x=_z, initial=0)
        integral_32 = cumulative_trapezoid(integrand_32, x=_z, initial=0)
        integral_33 = cumulative_trapezoid(integrand_33, x=_z, initial=0)
        return _z, integral_30, integral_31, integral_32, integral_33
    

    def subhalo_mass_stripped_pert0(self, ma, za, z):
        """Calculate subhalo mass stripping using zeroth-order perturbation."""
        eps_0 = self.eps_0(za, z)
        return ma * np.exp(eps_0)
    

    def subhalo_mass_stripped_pert1(self, ma, za, z):
        """Calculate subhalo mass stripping using first-order perturbation."""
        eps_0 = self.eps_0(za, z)
        eps_10 = self.eps_10(za, z)
        eps_11 = self.eps_11(za, z)
        ln_ma = np.log(ma)
        eps = (eps_0 
               + eps_10 + ln_ma * eps_11)
        return ma * np.exp(eps)
    

    def subhalo_mass_stripped_pert2(self, ma, za, z):
        """Calculate subhalo mass stripping using second-order perturbation."""
        eps_0 = self.eps_0(za, z)
        eps_10 = self.eps_10(za, z)
        eps_11 = self.eps_11(za, z)
        eps_20 = self.eps_20(za, z)
        eps_21 = self.eps_21(za, z)
        eps_22 = self.eps_22(za, z)
        ln_ma = np.log(ma)
        eps = (eps_0 
               + eps_10 + ln_ma * eps_11 
               + eps_20 + ln_ma * eps_21 + ln_ma**2 * eps_22)
        return ma * np.exp(eps)
    

    def subhalo_mass_stripped_pert2_shanks(self, ma, za, z):
        """Calculate subhalo mass stripping using second-order perturbation with Shanks transformation."""
        eps_0 = self.eps_0(za, z)
        eps_10 = self.eps_10(za, z)
        eps_11 = self.eps_11(za, z)
        eps_20 = self.eps_20(za, z)
        eps_21 = self.eps_21(za, z)
        eps_22 = self.eps_22(za, z)
        ln_ma = np.log(ma)
        # NOTE: Shanks transformation
        # For A_n = \sum_{i=0}^{n} a_i, the Shanks transformation is given by
        #  S_n = A_{n+1} - (A_{n+1} - A_n)^2 / (A_{n+1} - 2A_n + A_{n-1})
        #      = A_{n+1} - (A_{n+1} - A_n)^2 / ((A_{n+1} - A_n) - (A_n - A_{n-1}))
        # Since A_n = \sum_{i=0}^{n} a_i, we can write
        #  S_n = A_{n+1} - a_{n+1}^2 / (a_{n+1} - a_n)
        # In our case, we calculate the epsilon as eps = \sum_{i=0}^{n} eps_i.
        # Therefore, we can apply Shanks transformation to eps up to the second order as follows:
        # S_2 = (eps_0 + eps_1 + eps_2) - eps_2^2 / (eps_2 - eps_1)
        eps_1 = eps_10 + ln_ma * eps_11
        eps_2 = eps_20 + ln_ma * eps_21 + ln_ma**2 * eps_22
        eps_2m1 = (eps_20 - eps_10) + ln_ma * (eps_21 - eps_11) + ln_ma**2 * eps_22
        eps_shanks = - eps_2**2 / eps_2m1
        eps_shanks = np.where(np.isnan(eps_shanks), 0, eps_shanks)
        # When the correction is too small, the Shanks transformation may not be stable.
        eps_shanks = np.where(np.abs((eps_1+eps_2)/eps_0) < 0.02, 0, eps_shanks)
        eps = eps_0 + eps_1 + eps_2 + eps_shanks
        # import pandas as pd
        # df = pd.DataFrame({'z': _z, 'eps_0': eps_0, 'eps_1': eps_1, 'eps_2': eps_2, 'eps_shanks': eps_shanks, 'eps': eps})
        # display(df[(df['z'] > 4) & (df['z'] < 6)])
        return ma * np.exp(eps)
    
    
    def subhalo_mass_stripped_pert3(self, ma, za, z):
        """Calculate subhalo mass stripping using third-order perturbation."""
        eps_0 = self.eps_0(za, z)
        eps_10 = self.eps_10(za, z)
        eps_11 = self.eps_11(za, z)
        eps_20 = self.eps_20(za, z)
        eps_21 = self.eps_21(za, z)
        eps_22 = self.eps_22(za, z)
        eps_30 = self.eps_30(za, z)
        eps_31 = self.eps_31(za, z)
        eps_32 = self.eps_32(za, z)
        eps_33 = self.eps_33(za, z)
        ln_ma = np.log(ma)
        eps = (eps_0 
               + eps_10 + ln_ma * eps_11 
               + eps_20 + ln_ma * eps_21 + ln_ma**2 * eps_22
               + eps_30 + ln_ma * eps_31 + ln_ma**2 * eps_32 + ln_ma**3 * eps_33)
        return ma * np.exp(eps)
    
    
    def subhalo_mass_stripped(self,ma,za,z,method="pert2_shanks",**kwargs):
        """ A wrapper function to calculate subhalo mass stripping.
        
        Parameters
        ----------
        ma : float
            initial subhalo mass.
        za : float
            initial redshift.
        z : float
            final redshift.
        method : str, optional
            method to calculate the subhalo mass stripping.
            - "odeint" : use odeint to solve the differential equation.
            - "pert0" : use perturbative method with zeroth-order correction.
            - "pert1" : use perturbative method with first-order correction.
            - "pert2" : use perturbative method with second-order correction.
            - "pert2_shanks" : use perturbative method with second-order correction and Shanks transformation.
            - "pert3" : use perturbative method with third-order correction.
        kwargs : dict, optional
            additional arguments for the odeint function.

        Returns
        -------
        zcalc : array
            redshift array.
        mcalc : array
            subhalo mass array.
        """
        match method:
            case "odeint":
                return self.subhalo_mass_stripped_odeint(ma,za,z,**kwargs)
            case "pert0":
                return self.subhalo_mass_stripped_pert0(ma,za,z)
            case "pert1":
                return self.subhalo_mass_stripped_pert1(ma,za,z)
            case "pert2":
                return self.subhalo_mass_stripped_pert2(ma,za,z)
            case "pert2_shanks":
                return self.subhalo_mass_stripped_pert2_shanks(ma,za,z)
            case "pert3":
                return self.subhalo_mass_stripped_pert3(ma,za,z)
            case _:
                raise ValueError(f"Invalid method: {method}")



    
class subhalo_properties(halo_model):

    
    def __init__(self):
        halo_model.__init__(self)

    
    def Ffunc(self, dela, s1, s2):
        """ Returns Eq. (12) of Yang et al. (2011) """
        return 1/np.sqrt(2.*np.pi)*dela/(s2-s1)**1.5

    
    def Gfunc(self, dela, s1, s2):
        G0     = 0.57
        gamma1 = 0.38
        gamma2 = -0.01
        sig1   = np.sqrt(s1)
        sig2   = np.sqrt(s2)
        return G0*pow(sig2/sig1,gamma1)*pow(dela/sig1,gamma2)

    
    def Ffunc_Yang(self, delc1, delc2, s1, s2):
        """ Returns Eq. (14) of Yang et al. (2011) """
        return 1./np.sqrt(2.*np.pi)*(delc2-delc1)/(s2-s1)**1.5 \
            *np.exp(-(delc2-delc1)**2/(2.*(s2-s1)))

    
    def Na_calc(self, ma, zacc, Mhost, z0=0., N_herm=200, Nrand=1000, Na_model=3):
        """ Returns Na, Eq. (3) of Yang et al. (2011) """
        zacc_2d   = zacc.reshape(-1,1)
        M200_0    = self.Mzzi(Mhost,zacc_2d,z0)
        logM200_0 = np.log10(M200_0)

        xxi,wwi = hermgauss(N_herm)
        xxi = xxi.reshape(-1,1,1)
        wwi = wwi.reshape(-1,1,1)
        # Eq. (21) in Yang et al. (2011) 
        sigmalogM200 = 0.12-0.15*np.log10(M200_0/Mhost)
        logM200 = np.sqrt(2.)*sigmalogM200*xxi+logM200_0
        M200 = 10.**logM200
            
        mmax = np.minimum(M200,Mhost/2.)
        Mmax = np.minimum(M200_0+mmax,Mhost)
        
        if Na_model==3:
            zlist    = zacc_2d*np.linspace(1,0,Nrand)
            iMmax    = np.argmin(np.abs(self.Mzzi(Mhost,zlist,z0)-Mmax),axis=-1)
            z_Max    = zlist[np.arange(len(zlist)),iMmax]
            z_Max_3d = z_Max.reshape(N_herm,len(zlist),1)
            delcM    = self.deltac_func(z_Max_3d)
            delca    = self.deltac_func(zacc_2d)
            sM       = self.s_func(Mmax)
            sa       = self.s_func(ma)
            xmax     = (delca-delcM)**2/(2.*(self.s_func(mmax)-sM))
            normB    = special.gamma(0.5)*special.gammainc(0.5,xmax)/np.sqrt(np.pi)
            # those reside in the exponential part of Eq. (14) 
            Phi      = self.Ffunc_Yang(delcM,delca,sM,sa)/normB*np.heaviside(mmax-ma,0)
        elif Na_model==1:
            delca    = self.deltac_func(zacc_2d)
            sM       = self.s_func(M200)
            sa       = self.s_func(ma)
            xmin     = self.s_func(mmax)-self.s_func(M200)
            normB    = 1./np.sqrt(2*np.pi)*delca*2./xmin**0.5*special.hyp2f1(0.5,0.,1.5,-sM/xmin)
            Phi      = self.Ffunc(delca,sM,sa)/normB*np.heaviside(mmax-ma,0)
        elif Na_model==2:
            delca    = self.deltac_func(zacc_2d)
            sM       = self.s_func(M200)
            sa       = self.s_func(ma)
            xmin     = self.s_func(mmax)-self.s_func(M200)
            normB    = 1./np.sqrt(2.*np.pi)*delca*0.57 \
                           *(delca/np.sqrt(sM))**-0.01*(2./(1.-0.38))*sM**(-0.38/2.) \
                           *xmin**(0.5*(0.38-1.)) \
                           *special.hyp2f1(0.5*(1-0.38),-0.38/2.,0.5*(3.-0.38),-sM/xmin)
            Phi      = self.Ffunc(delca,sM,sa)*self.Gfunc(delca,sM,sa)/normB \
                           *np.heaviside(mmax-ma,0)

        if N_herm==1:
            F2t = np.nan_to_num(Phi)
            F2  =F2t.reshape((len(zacc_2d),len(ma)))
        else:
            F2 = np.sum(np.nan_to_num(Phi)*wwi/np.sqrt(np.pi),axis=0)
        Na = F2*self.dsdm(ma,0.)*self.dMdz(Mhost,zacc_2d,z0)*(1.+zacc_2d)
        return Na

    
    def subhalo_properties_calc(self, M0, redshift=0.0, dz=0.01, zmax=7.0, N_ma=500, sigmalogc=0.128,
                                N_herm=5, logmamin=-6, logmamax=None, N_hermNa=200, Na_model=3, 
                                ct_th=0.77, profile_change=True, M0_at_redshift=False, method="pert2_shanks", **kwargs):
        """
        This is the main function of SASHIMI-C, which makes a semi-analytical subhalo catalog.
        
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
                                   be completely desrupted. Suggested values: 0.77 (default) or 0 (no desruption).
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
        
        """

        if M0_at_redshift:
            Mz        = M0
            M0_list   = np.logspace(0.,3.,1000)*Mz
            fint      = interp1d(self.Mzi(M0_list,redshift),M0_list)
            M0        = fint(Mz)
        self.M0       = M0
        self.redshift = redshift
        
        zdist = np.arange(redshift+dz,zmax+dz,dz)
        if logmamax==None:
            logmamax = np.log10(0.1*M0/self.Msun)
        ma200     = np.logspace(logmamin,logmamax,N_ma)*self.Msun
        rs_acc    = np.zeros((len(zdist),N_herm,len(ma200)))
        rhos_acc  = np.zeros((len(zdist),N_herm,len(ma200)))
        rs_z0     = np.zeros((len(zdist),N_herm,len(ma200)))
        rhos_z0   = np.zeros((len(zdist),N_herm,len(ma200)))
        ct_z0     = np.zeros((len(zdist),N_herm,len(ma200)))
        survive   = np.zeros((len(zdist),N_herm,len(ma200)))
        m0_matrix = np.zeros((len(zdist),N_herm,len(ma200)))

        solver = TidalStrippingSolver(
            M0 = M0,
            z_min = redshift,
            z_max = zmax,
            n_z_interp=64
        )

        for iz in range(len(zdist)):
            ma           = self.Mvir_from_M200_fit(ma200,zdist[iz])
            Oz           = self.OmegaM*(1.+zdist[iz])**3/self.g(zdist[iz])
            m0           = solver.subhalo_mass_stripped(ma, zdist[iz], redshift, method=method, **kwargs)
            c200sub      = self.conc200(ma200,zdist[iz])
            rvirsub      = (3.*ma/(4.*np.pi*self.rhocrit0*self.g(zdist[iz]) \
                               *self.Delc(Oz-1)))**(1./3.)
            r200sub      = (3.*ma200/(4.*np.pi*self.rhocrit0*self.g(zdist[iz])*200.))**(1./3.)
            c_mz         = c200sub*rvirsub/r200sub
            x1,w1        = hermgauss(N_herm)
            x1           = x1.reshape(len(x1),1)
            w1           = w1.reshape(len(w1),1)
            log10c_sub   = np.sqrt(2.)*sigmalogc*x1+np.log10(c_mz)
            c_sub        = 10.0**log10c_sub
            rs_acc[iz]   = rvirsub/c_sub
            rhos_acc[iz] = ma/(4.*np.pi*rs_acc[iz]**3*self.fc(c_sub))
            if(profile_change==True):
                rmax_acc    = rs_acc[iz]*2.163
                Vmax_acc    = np.sqrt(rhos_acc[iz]*4*np.pi*self.G/4.625)*rs_acc[iz]
                Vmax_z0     = Vmax_acc*(2.**0.4*(m0/ma)**0.3*(1+m0/ma)**-0.4)
                rmax_z0     = rmax_acc*(2.**-0.3*(m0/ma)**0.4*(1+m0/ma)**0.3)
                rs_z0[iz]   = rmax_z0/2.163
                rhos_z0[iz] = (4.625/(4.*np.pi*self.G))*(Vmax_z0/rs_z0[iz])**2
            else:
                rs_z0[iz]   = rs_acc[iz]
                rhos_z0[iz] = rhos_acc[iz]
            ctemp         = np.linspace(0,100,1000)
            ftemp         = interp1d(self.fc(ctemp),ctemp,fill_value='extrapolate')
            ct_z0[iz]     = ftemp(m0/(4.*np.pi*rhos_z0[iz]*rs_z0[iz]**3))
            survive[iz]   = np.where(ct_z0[iz]>ct_th,1,0)
            m0_matrix[iz] = m0*np.ones((N_herm,1))

        Na           = self.Na_calc(ma,zdist,M0,z0=0.,N_herm=N_hermNa,Nrand=1000,
                                    Na_model=Na_model)
        Na_total     = integrate.simpson(integrate.simpson(Na,x=np.log(ma)),x=np.log(1+zdist))
        weight       = Na/(1.+zdist.reshape(-1,1))
        weight       = weight/np.sum(weight)*Na_total
        weight       = (weight.reshape((len(zdist),1,len(ma))))*w1/np.sqrt(np.pi)
        z_acc        = (zdist.reshape(-1,1,1))*np.ones((1,N_herm,N_ma))
        z_acc        = z_acc.reshape(-1)
        ma200        = ma200*np.ones((len(zdist),N_herm,1))
        ma200        = ma200.reshape(-1)
        m_z0         = m0_matrix.reshape(-1)
        rs_acc       = rs_acc.reshape(-1)
        rhos_acc     = rhos_acc.reshape(-1)
        rs_z0        = rs_z0.reshape(-1)
        rhos_z0      = rhos_z0.reshape(-1)
        ct_z0        = ct_z0.reshape(-1)
        weight       = weight.reshape(-1)
        survive      = (survive==1).reshape(-1)

        return ma200, z_acc, rs_acc, rhos_acc, m_z0, rs_z0, rhos_z0, ct_z0, weight, survive




class subhalo_observables(subhalo_properties):
    
    
    def __init__(self, M0_per_Msun, redshift=0., dz=0.01, zmax=7.0, N_ma=500, sigmalogc=0.128,
                 N_herm=5, logmamin=-6, logmamax=None, N_hermNa=200, Na_model=3,
                 ct_th=0.77, profile_change=True, M0_at_redshift=False,method="pert2_shanks", **kwargs):
        """
        This class computes various subhalo observables in a host halo. 
        
        -----
        Input
        -----
        M0_per_Msun:               Mass of the host halo defined as M_{200} (200 times critial density)
                                   at z = 0, in units of solar mass, Msun. Note that this is *not* the 
                                   host mass at the given redshift! It can be obtained via Mzi(M0,redshift).
                                   If you want to give this parameter as the mass at the given redshift,
                                   then turn 'M0_at_redshift' parameter on (see below).
                                   
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
                                   be completely disrupted. Suggested values: 0.77 (default) or 0 (no disruption).
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


        
        When called with these input parameters, this class initially creates a list of subhalos, characterized
        by the following output parameters.
        
        ------
        Output
        ------

        ma200:    Mass m_{200} at accretion.
        z_a:      Redshift at accretion.
        rs_a:     Scale radius r_s at accretion.
        rhos_a:   Characteristic density \rho_s at accretion.
        m0:       Mass up to tidal truncation radius at a given redshift.
        rs0:      Scale radius r_s at a given redshift.
        rhos0:    Characteristic density \rho_s at a given redshift.
        ct0:      Tidal truncation radius in units of r_s at a given redshift.
        weight:   Effective number of subhalos that are characterized by the same set of the parameters above.
        survive:  If that subhalo survive against tidal disruption or not.
        
        Vmax:     Maximum circular velocity at a given redshift.
        rmax:     Radius at which the orbital speed reaches Vmax.
        Vpeak:    Maximum circular velocity at accretion.
        rpeak:    Radius at which the orbital speed reaches Vpeak.

        """

        subhalo_properties.__init__(self)
        ma200, z_a, rs_a, rhos_a, m0, rs0, rhos0, ct0, weight, survive \
            = self.subhalo_properties_calc(M0_per_Msun*self.Msun,redshift,dz,zmax,N_ma,sigmalogc,N_herm,
                                           logmamin,logmamax,N_hermNa,Na_model,ct_th,profile_change,
                                           M0_at_redshift,method,**kwargs)
        self.ma200  = ma200[survive]
        self.z_a    = z_a[survive]
        self.rs_a   = rs_a[survive]
        self.rhos_a = rhos_a[survive]
        self.m0     = m0[survive]
        self.rs0    = rs0[survive]
        self.rhos0  = rhos0[survive]
        self.ct0    = ct0[survive]
        self.weight = weight[survive]
        self.rmax   = 2.163*self.rs0
        self.Vmax   = np.sqrt(4.*np.pi*self.G*self.rhos0/4.625)*self.rs0
        self.rpeak  = 2.163*self.rs_a
        self.Vpeak  = np.sqrt(4.*np.pi*self.G*self.rhos_a/4.625)*self.rs_a


    def mass_function(self, evolved=True):
        """
        Subhalo mass function
        
        -----
        Input
        -----
        (Optional) evolved: If True (False), this function calculates evolved (unevolved) mass function.
                            Here 'evolved' means that subhalos experiences tidal mass loss, whereas
                            'unevolved' means that mass loss is ignored.
                            
        ------
        Output
        ------
        m:       Mass of subhalo in units of [Msun].
        dNdlnm:  Subhalo mass function dN/dln(m).
        
        """

        if evolved:
            N,lnm_edges = np.histogram(np.log(self.m0),weights=self.weight,bins=100)
        else:
            N,lnm_edges = np.histogram(np.log(self.ma200),weights=self.weight,bins=100)

        lnm = (lnm_edges[1:]+lnm_edges[:-1])/2.
        dlnm = lnm_edges[1:]-lnm_edges[:-1]

        m = np.exp(lnm)
        dNdlnm = N/dlnm

        return m/self.Msun, dNdlnm


    def Nsat_Mpeak(self, Mpeak_th):
        """
        Calculate expected number of satellites for a given host halo. Satellites are assumed
        to be formed in a subhalo whose peak mass (equivalent to the mass at accretion) is above
        a given threshold value, Mpeak_th.
        
        -----
        Input
        -----
        Mpeak_th:   Threshold value for m_{peak} (= mass at accretion) above which a satellite
                    galaxy is (assumed to be) formed. E.g., Mpeak_th = 1.e8*Msun.
        
        ------
        Output
        ------
        m:          Subhalo masses within tidal radius in units of [Msun].
        Nccum_m:    Complementary cumulative number of subhalos N(>m) 
                    with the condition m_{peak}>Mpeak_th.
        Vmax:       Vmax of a subhalo in units of [km/s].
        Nccum_Vmax: Complementary cumulative number of subhalos N(>Vmax)
                    with the condition m_{peak}>Mpeak_th.
        
        """

        N,lnm_edges = np.histogram(np.log(self.m0[self.ma200>Mpeak_th]),
                                   weights=self.weight[self.ma200>Mpeak_th],bins=100)
        lnm         = (lnm_edges[1:]+lnm_edges[:-1])/2.
        m           = np.exp(lnm)
        Ncum        = np.cumsum(N)
        Nccum_m     = Ncum[-1]-Ncum

        N,lnVmax_edges = np.histogram(np.log(self.Vmax[self.ma200>Mpeak_th]),
                                      weights=self.weight[self.ma200>Mpeak_th],bins=100)
        lnVmax         = (lnVmax_edges[1:]+lnVmax_edges[:-1])/2.
        Vmax           = np.exp(lnVmax)
        Ncum           = np.cumsum(N)
        Nccum_Vmax     = Ncum[-1]-Ncum

        return m/self.Msun, Nccum_m, Vmax/(self.km/self.s), Nccum_Vmax


    def Nsat_Vpeak(self, Vpeak_th):
        """
        Calculate expected number of satellites for a given host halo. Satellites are assumed
        to be formed in a subhalo whose Vpeak (equivalent to the Vmax at accretion) is above
        a given threshold value, Vpeak_th.
        
        -----
        Input
        -----
        Vpeak_th:   Threshold value for V_{peak} (= V_{max} at accretion) above which a satellite
                    galaxy is (assumed to be) formed. E.g., Vpeak_th = 18*km/s.
        
        ------
        Output
        ------
        m:          Subhalo masses within tidal radius in units of [Msun].
        Nccum_m:    Complementary cumulative number of subhalos N(>m) 
                    with the condition V_{peak}>Vpeak_th.
        Vmax:       Vmax of a subhalo in units of [km/s].
        Nccum_Vmax: Complementary cumulative number of subhalos N(>Vmax)
                    with the condition V_{peak}>Vpeak_th.
        
        """

        N,lnm_edges = np.histogram(np.log(self.m0[self.Vpeak>Vpeak_th]),
                                   weights=self.weight[self.Vpeak>Vpeak_th],bins=100)
        lnm         = (lnm_edges[1:]+lnm_edges[:-1])/2.
        m           = np.exp(lnm)
        Ncum        = np.cumsum(N)
        Nccum_m     = Ncum[-1]-Ncum

        N,lnVmax_edges = np.histogram(np.log(self.Vmax[self.Vpeak>Vpeak_th]),
                                      weights=self.weight[self.Vpeak>Vpeak_th],bins=100)
        lnVmax         = (lnVmax_edges[1:]+lnVmax_edges[:-1])/2.
        Vmax           = np.exp(lnVmax)
        Ncum           = np.cumsum(N)
        Nccum_Vmax     = Ncum[-1]-Ncum

        return m/self.Msun, Nccum_m, Vmax/(self.km/self.s), Nccum_Vmax


    def mass_fraction(self, evolved=True):
        """
        Subhalo mass fraction: \sum_i m_i / M_host
        
        -----
        Input
        -----
        (Optional) evolved: If True (False), this function calculates evolved (unevolved) mass function.
                            Here 'evolved' means that subhalos experiences tidal mass loss, whereas
                            'unevolved' means that mass loss is ignored.
                            
        ------
        Output
        ------
        fsh: Fractional mass of the host in the form of subhalos.
        
        """

        Mhost     = self.Mzi(self.M0,self.redshift)
        if evolved:
            fsh = np.sum(self.m0*self.weight)/Mhost
        else:
            fsh = np.sum(self.ma200*self.weight)/Mhost

        return fsh


    def annihilation_boost_factor(self, n=0, evolved=True):
        """
        Annihilation boost factor B_{sh}. Note that the effect of sub-subhalos and higher order
        structure is not included.
        
        -----
        Input
        -----
        (Optional) n:       The effects up to sub^{n}-subhalos will be included. 
                            If n=0 (default), no sub-subhalos and beyond is considered.
                            For other values of n, the function requires pre-computed boost factors
                            B_sh from the previous (n-1)th iteration and subhalo mass fraction f_sh.
                            These are stored under 'data/boost/' directory. If the directory does not
                            exist, excecuting 'boost_iteraction.py' will generate the  necessary files
                            and store them in the directory, up to n = 3.
        (Optional) evolved: If True (False), this function calculates evolved (unevolved) mass function.
                            Here 'evolved' means that subhalos experiences tidal mass loss, whereas
                            'unevolved' means that mass loss is ignored.
                            
        ------
        Output
        ------
        Bsh:              Annihilation boost factor B_{sh}. See Eq. (37) of Ando et al.
                          arXiv:1903.11427 for definition.
        luminosity_ratio: Ratio of the total luminosity (subhalos+host) and the luminosity due
                          to host only in the absence of subhalos:
                          L_{total}/L_{host,0} = 1-f_{sh}^2+B_{sh}
        
        """
        
        fsh = self.mass_fraction(evolved)
        if n==0:
            fssh = 0.
            Bssh = 0.
        else:                        
            list_Bssh = np.loadtxt('data/boost/Bsh_%s.txt'%(n-1))
            list_fssh = np.loadtxt('data/boost/fsh.txt')
            list_za  = np.loadtxt('data/boost/za.txt')
            list_ma  = np.loadtxt('data/boost/ma.txt')

            list_log_ma_flat   = np.log10(list_ma.flatten())
            list_za_flat       = (list_za.reshape(-1,1)*np.ones_like(list_ma[0])).flatten()
            list_log_fssh_flat = np.log10(list_fssh.flatten())
            list_log_Bssh_flat = np.log10((list_Bssh+1.e-30).flatten())

            log_Bssh = griddata((list_log_ma_flat,list_za_flat),list_log_Bssh_flat,(np.log10(self.ma200),self.z_a),method='linear')
            log_fssh = griddata((list_log_ma_flat,list_za_flat),list_log_fssh_flat,(np.log10(self.ma200),self.z_a),method='linear')
            log_Bssh[~np.isfinite(log_Bssh)] = -np.inf
            log_fssh[~np.isfinite(log_fssh)] = -np.inf
            Bssh = 10.**log_Bssh
            fssh = 10.**log_fssh
                        
            mavir = self.Mvir_from_M200_fit(self.ma200,self.z_a)
            Oz    = self.OmegaM*(1.+self.z_a)**3/self.g(self.z_a)
            ravir = np.cbrt(3.*mavir/(4.*np.pi*self.rhocrit(self.z_a)*self.Delc(Oz-1.)))
            cavir = ravir/self.rs_a

            Bssh = Bssh*self.rs0**3*(np.arcsinh(self.ct0)-self.ct0/np.sqrt(1.+self.ct0**2))
            Bssh = Bssh/(self.rs_a**3*(np.arcsinh(cavir)-cavir/np.sqrt(1.+cavir**2)))
            Bssh = Bssh/(self.rhos0**2*self.rs0**3*(1.-1./(1.+self.ct0)**3))
            Bssh = Bssh*(self.rhos_a**2*self.rs_a**3*(1.-1./(1.+cavir)**3))
            fssh = fssh*self.rs0**3*(np.arcsinh(self.ct0)-self.ct0/np.sqrt(1.+self.ct0**2))
            fssh = fssh/(self.rs_a**3*(np.arcsinh(cavir)-cavir/np.sqrt(1.+cavir**2)))
            fssh = fssh/(self.rhos0*self.rs0**3*self.fc(self.ct0))
            fssh = fssh*(self.rhos_a*self.rs_a**3*self.fc(cavir))

        if evolved:
            Lsh  = np.sum((1.-fssh**2+Bssh)*self.rhos0**2*self.rs0**3*(1.-1./(1.+self.ct0)**3)*self.weight)
        else:
            r200 = (3.*self.ma200/(4.*np.pi*self.rhocrit(self.redshift)*200.))**(1./3.)
            c200 = r200/self.rs_a
            Lsh  = np.sum(self.rhos_a**2*self.rs_a**3*(1.-1./(1.+c200)**3)*self.weight)
            
        Mhost     = self.Mzi(self.M0,self.redshift)
        r200_host = (3.*Mhost/(4.*np.pi*self.rhocrit(self.redshift)*200.))**(1./3.)
        c200_host = self.conc200(Mhost,self.redshift)
        rs_host   = r200_host/c200_host
        rhos_host = Mhost/(4.*np.pi*rs_host**3*self.fc(c200_host))
        Lhost0    = rhos_host**2*rs_host**3*(1.-1./(1.+c200_host)**3)

        Bsh = Lsh/Lhost0
        luminosity_ratio = 1.-fsh**2+Bsh
        
        return Bsh, luminosity_ratio

    
    def subhalo_catalog_MC(self, mth):
        """
        This function returns a subhalo catalog generated with the Monte Carlo simulations.
        
        -----
        Input
        -----
        mth:  Threshold of subhalo mass above which the catalog is generated.
        
        ------
        Output
        ------
        ma200_MC:    Mass m_{200} at accretion. [Msun]
        z_a_MC:      Redshift at accretion.
        rs_a_MC:     Scale radius r_s at accretion. [kpc]
        rhos_a_MC:   Characteristic density \rho_s at accretion. [Msun/pc^3]
        m0_MC:       Mass up to tidal truncation radius at a given redshift. [Msun]
        rs0_MC:      Scale radius r_s at a given redshift. [kpc]
        rhos0_MC:    Characteristic density \rho_s at a given redshift. [Msun/pc^3]
        ct0_MC:      Tidal truncation radius in units of r_s at a given redshift.
        
        """
        
        condition = self.m0>mth
        ma200     = self.ma200[condition]
        z_a       = self.z_a[condition]
        rs_a      = self.rs_a[condition]
        rhos_a    = self.rhos_a[condition]
        m0        = self.m0[condition]
        rs0       = self.rs0[condition]
        rhos0     = self.rhos0[condition]
        ct0       = self.ct0[condition]
        weight    = self.weight[condition]
        
        mu_sh      = np.sum(weight)
        prob       = weight/mu_sh
        subhalo_id = np.arange(len(prob))
        N_sh       = np.random.poisson(mu_sh)
        id_MC      = np.random.choice(subhalo_id,size=N_sh,p=prob)
        ma200_MC   = ma200[id_MC]
        z_a_MC     = z_a[id_MC]
        rs_a_MC    = rs_a[id_MC]
        rhos_a_MC  = rhos_a[id_MC]
        m0_MC      = m0[id_MC]
        rs0_MC     = rs0[id_MC]
        rhos0_MC   = rhos0[id_MC]
        ct0_MC     = ct0[id_MC]

        return ma200_MC/self.Msun, z_a_MC, rs_a_MC/self.kpc, rhos_a_MC/(self.Msun/self.pc**3), \
                   m0_MC/self.Msun, rs0_MC/self.kpc, rhos0_MC/(self.Msun/self.pc**3), ct0_MC