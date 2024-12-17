from sashimi_c import *
import os
import tqdm

n_values = [1,2,3]

if not os.path.exists('data/boost'):
    os.makedirs('data/boost',exist_ok=True)

sh = subhalo_properties()

list_za  = np.arange(0.,7.,0.1)
ma_max   = sh.Mzi(1.e16,list_za)
list_ma  = np.logspace(-5,np.log10(ma_max/sh.Msun),22).T*sh.Msun

#if not os.path.exists('data/boost/za.txt'):
np.savetxt('data/boost/za.txt',list_za)
#if not os.path.exists('data/boost/ma.txt'):
np.savetxt('data/boost/ma.txt',list_ma)

list_fsh = np.empty_like(list_ma)
list_Bsh = np.empty_like(list_ma)

for n in n_values:
    print(f"Running for n = {n}")
    for i,za in tqdm.tqdm(enumerate(list_za),total=len(list_za)):
        #print(za)
        #print(np.log10(list_ma[i]/sh.Msun))
        for j,ma in enumerate(list_ma[i]):
            sh    = subhalo_observables(ma/sh.Msun,za,M0_at_redshift=True)
            fsh   = sh.mass_fraction()
            Bsh,_ = sh.annihilation_boost_factor(n=n)
        
            list_fsh[i,j] = fsh
            list_Bsh[i,j] = Bsh
            #print(za,np.log10(ma/sh.Msun),fsh,Bsh)

    if not os.path.exists('data/boost/fsh.txt'):
        np.savetxt('data/boost/fsh.txt',list_fsh)

    np.savetxt('data/boost/Bsh_%s.txt'%n,list_Bsh)
