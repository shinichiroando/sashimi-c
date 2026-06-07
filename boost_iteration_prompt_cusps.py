from sashimi_c import *
import os
import tqdm

f_surv          = 1.
f_surv_stripped = 1.

n_values = [0,1,2,3,4,5]

if not os.path.exists('data/prompt_cusps/boost'):
    os.makedirs('data/prompt_cusps/boost',exist_ok=True)

sh = subhalo_properties()

list_za  = np.arange(0.,7.,0.1)
ma_max   = sh.Mzi(1.e16,list_za)
list_ma  = np.logspace(-5,np.log10(ma_max/sh.Msun),22).T*sh.Msun

np.savetxt('data/prompt_cusps/boost/za.txt',list_za)
np.savetxt('data/prompt_cusps/boost/ma.txt',list_ma)

list_fsh            = np.empty_like(list_ma)
list_Bsh            = np.empty_like(list_ma)
list_Bcusp_dressed  = np.empty_like(list_ma)
list_Bcusp_naked    = np.empty_like(list_ma)
list_Ncusp_dressed  = np.empty_like(list_ma)
list_Ncusp_naked    = np.empty_like(list_ma)

for n in n_values:
    print(f"Running for n = {n}")
    for i,za in tqdm.tqdm(enumerate(list_za),total=len(list_za)):
        #print(za)
        #print(np.log10(list_ma[i]/sh.Msun))
        for j,ma in enumerate(list_ma[i]):
            sh    = subhalo_observables(ma/sh.Msun,za,M0_at_redshift=True,prompt_cusps=True,logmamin=-9,ct_th=0.0,N_ma=600)
            fsh   = sh.mass_fraction()
            Bsh, Bcusp_dressed, Bcusp_naked, luminosity_ratio, Ncusp_dressed, Ncusp_naked \
                = sh.annihilation_boost_factor_prompt_cusps(n=n,f_surv=f_surv,f_surv_stripped=f_surv_stripped)

            list_fsh[i,j]      = fsh
            list_Bsh[i,j]      = Bsh
            list_Bcusp_dressed[i,j] = Bcusp_dressed
            list_Bcusp_naked[i,j]   = Bcusp_naked
            list_Ncusp_dressed[i,j] = Ncusp_dressed
            list_Ncusp_naked[i,j]   = Ncusp_naked
            #print(za,np.log10(ma/sh.Msun),fsh,Bsh,Bcusp_dressed,Bcusp_naked,
            #      np.log10(Ncusp_dressed),np.log10(Ncusp_naked))

    if not os.path.exists('data/prompt_cusps/boost/fsh.txt'):
        np.savetxt('data/prompt_cusps/boost/fsh.txt',list_fsh)

    np.savetxt('data/prompt_cusps/boost/Bsh_%s_%.1f_%.1f.txt'%(n,f_surv,f_surv_stripped),list_Bsh)
    np.savetxt('data/prompt_cusps/boost/Bcusp_dressed_%s_%.1f_%.1f.txt'%(n,f_surv,f_surv_stripped),list_Bcusp_dressed)
    np.savetxt('data/prompt_cusps/boost/Bcusp_naked_%s_%.1f_%.1f.txt'%(n,f_surv,f_surv_stripped),list_Bcusp_naked)
    np.savetxt('data/prompt_cusps/boost/Ncusp_dressed_%s_%.1f_%.1f.txt'%(n,f_surv,f_surv_stripped),list_Ncusp_dressed)
    np.savetxt('data/prompt_cusps/boost/Ncusp_naked_%s_%.1f_%.1f.txt'%(n,f_surv,f_surv_stripped),list_Ncusp_naked)