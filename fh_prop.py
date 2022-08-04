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

'''
 
'''
def spin_projection(interp_op):
    '''
    Define the quasi-local nucleon operators as in: 
      "Clebsch-Gordan construction of lattice interpolating fields for excited baryons" hep-lat/0508018
        Input:
        zero-momentum   Baryon interpolating field operator 

        output:
        gauge invariant quark trilinear with appropiate flavor structure.
     '''
    operator = dict()
    # there are 23 G_1g operators, 7 G_2g operators
    operator['']




#ef bilinear_gamma(current_0, fh_prop_src, quark_propagator,u):

def flavor_conserving_fh_baryon_contraction(q1,q2,q3,corr,spin):
    '''
    3-pt contraction routine. Current is inserted on each line in the 2-point function 
    Use relation between 3 point correlator and 
    eg. for the proton: 
        proton[lam] = U[lam] U[lam] D[lam]
        d/dlam proton[lam] |lam=0 = dU[0] U[0] D[0] + U[0] dU[0] D[0] + U[0] U[0] dD[0]
        where:  dU = FH_U , dD = FH_D
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

    U = gamma.U_DR_to_DP
    U_adj = gamma.U_DR_to_DP_adj

    known_path = sys.argv[1]
    """ propagators for two point contractions """
    f = h5.File(known_path+'/test_propagator.h5')
    ps_prop_up = f['sh_sig2p0_n5/PS_up'][()]
    ps_prop_down = f['sh_sig2p0_n5/PS_dn'][()]
    ps_prop_strange = f['sh_sig2p0_n5/PS_strange'][()]

    """ fh propagators"""
    f_fh = h5.File(known_path + '/test_fh_propagator.h5')
    ps_fh_dn_A3     = f_fh['PS/fh_dn_A3'][()]
    ps_fh_dn_V4     = f_fh['PS/fh_dn_V4'][()]
    ps_fh_up_A3     = f_fh['PS/fh_up_A3'][()]
    ps_fh_up_V4     = f_fh['PS/fh_up_V4'][()]
    ps_fh_up_dn_A3  = f_fh['PS/fh_up_dn_A3'][()]
    ps_fh_up_dn_V4  = f_fh['PS/fh_up_dn_V4'][()]
    
    f.close()

    g_a3 = np.einsum('ik,kj->ij',gamma.g_3,gamma.g_5) # axial charge
    g_v4 = np.einsum('ik,kj->ij',gamma.g_4,gamma.g_5) # vector charge

    '''
    Rotate from Euclidean Degrand-Rossi to Euclidean Dirac-Pauli basis:
    SpinMatrix U = DiracToDRMat();
    quark_to_be_rotated = Uadj*quark_to_be_rotated*U
    np.einsum: evaluates einstein summation convention on operands
    '''
    ps_DP_up = np.einsum('ik,tzyxklab,lj->tzyxijab',Uadj,ps_prop_up,U)
    ps_DP_down = np.einsum('ik,tzyxklab,lj->tzyxijab',Uadj,ps_prop_down,U)

    print(known_path+'/test_propagator.h5/sh_sig2p0_n5/PS_prop shape')
    print(ps_DP_up.shape)
    Nt = ps_DP_up.shape[0]

    """ fh contract U """

    d




    return     

    
        
        

