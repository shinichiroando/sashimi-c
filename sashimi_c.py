import numpy as np
from scipy import integrate
from scipy import interpolate
from scipy import optimize
from scipy import special
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from numpy.polynomial.hermite import hermgauss
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, append=1)




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

    
    def subhalo_properties_calc(self, M0, redshift=0.0, dz=0.1, zmax=7.0, N_ma=500, sigmalogc=0.128,
                                N_herm=5, logmamin=-6, logmamax=None, N_hermNa=200, Na_model=3, 
                                ct_th=0.77, profile_change=True, M0_at_redshift=False):
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

        def Mzvir(z):
            Mz200 = self.Mzzi(M0,z,0.)
            Mvir = self.Mvir_from_M200(Mz200,z)
            return Mvir

        def AMz(z):
            log10a = (-0.0003*np.log10(Mzvir(z)/self.Msun)+0.02)*z \
                         +(0.011*np.log10(Mzvir(z)/self.Msun)-0.354)
            return 10.**log10a

        def zetaMz(z):
            return (0.00012*np.log10(Mzvir(z)/self.Msun)-0.0033)*z \
                       +(-0.0011*np.log10(Mzvir(z)/self.Msun)+0.026)

        def tdynz(z):
            Oz_z = self.OmegaM*(1.+z)**3/self.g(z)
            return 1.628/self.h*(self.Delc(Oz_z-1.)/178.0)**-0.5/(self.Hubble(z)/self.H0)*1.e9*self.yr

        def msolve(m, z):
            return AMz(z)*(m/tdynz(z))*(m/Mzvir(z))**zetaMz(z)/(self.Hubble(z)*(1+z))

        for iz in range(len(zdist)):
            ma           = self.Mvir_from_M200(ma200,zdist[iz])
            Oz           = self.OmegaM*(1.+zdist[iz])**3/self.g(zdist[iz])
            zcalc        = np.linspace(zdist[iz],redshift,100)
            sol          = odeint(msolve,ma,zcalc)
            m0           = sol[-1]
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
    
    
    def __init__(self, M0_per_Msun, redshift=0., dz=0.1, zmax=7.0, N_ma=500, sigmalogc=0.128,
                 N_herm=5, logmamin=-6, logmamax=None, N_hermNa=200, Na_model=3,
                 ct_th=0.77, profile_change=True, M0_at_redshift=False):
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
                                   be completely disrupted. Suggested values: 0.77 (default) or 0 (no disruption).
        (Optional) profile_change: Whether we implement the evolution of subhalo density profile through tidal
                                   mass loss. (default: True)
        (Optional) M0_at_redshift: If True, M0 is regarded as the mass at a given redshift, instead of z=0.

        
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
                                           M0_at_redshift)
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


    def annihilation_boost_factor(self, evolved=True):
        """
        Annihilation boost factor B_{sh}. Note that the effect of sub-subhalos and higher order
        structure is not included.
        
        -----
        Input
        -----
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
        if evolved:
            Lsh = np.sum(self.rhos0**2*self.rs0**3*(1.-1./(1.+self.ct0)**3)*self.weight)
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