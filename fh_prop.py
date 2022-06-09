import numpy as np
import h5py as h5
import baryon_contractions_gb as contractions
import gamma
import sys
import warnings
warnings.filterwarnings("ignore")

'''
This routine computes a Feynman-Hellmann propagator for bi-linear currents 
1. Construct the derivative correlation function, see eqn. 11 in https://arxiv.org/pdf/1612.06963.pdf
    - Replace 1 of quark propagators eg. U^{i,i'}_{a,a'} in the 2-pt corr func with the FH propagator
Parameters:
------------
Input: 
- Propagator 
- List of currents
    - spin
    - space
    - color
    - momentum

- Params for linear solver ??
'''

#ef bilinear_gamma(current_0, fh_prop_src, quark_propagator,u):

def flavor_conserving_fh_baryon_contraction(q1,q2,q3,corr,spin):
    '''
    3-pt contraction routine. Current is inserted on each line in the 2-point function 
    Use relation between 3 point correlator and 
    eg. for the proton: 
        proton[lam] = U[lam] U[lam] D[lam]
        d/dlam proton[lam] |lam=0 = dU[0] U[0] D[0] + U[0] dU[0] D[0] + U[0] U[0] dD[0]
        dU = FH_U 
        dD = FH_D
    '''
    nt,nz,ny,nx = q1.shape[0:4]
    two_point_obj = np.zeros([nt,nz,ny,nx],dtype=np.complex128)
    if corr in ['lambda_z', 'omega_m']:
        two_point_obj = isospin_zero_spin_contract(q1, q2, q3, corr, spin)
    elif corr in ['proton', 'neutron', 'xi_z', 'xi_m', 'xi_star_z','xi_star_m']:
        two_point_obj = contractions.isospin_half_spin_contract(q1, q2, q3, corr, spin)
    elif corr in  ['sigma_z','sigma_p','sigma_m', 'sigma_star_z', 'sigma_star_p','sigma_star_m']:
        two_point_obj = contractions.isospin_one_spin_contract(q1, q2, q3, corr, spin)
    elif corr in ['delta_m', 'delta_p','delta_z','delta_pp']:
        two_point_obj = contractions.isospin_three_half_spin_contract(q1, q2, q3, corr, spin)
    elif corr in ['sigma_to_lambda']:
        two_point_obj = contractions.sigma_to_lambda(q1, q2, q3, corr, spin)    
    elif corr in ['lambda_to_sigma']:
        two_point_obj = contractions.lambda_to_sigma(q1, q2, q3, corr, spin)

    U = gamma.U
    U_adj = gamma.Uadj

    

    return     

    
        
        

