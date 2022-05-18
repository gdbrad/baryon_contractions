'''
Author: Amy Nicholson
Futher modifications by Grant Bradley
'''

import numpy as np
import sys

def two_eps_color_contract(q1,q2,q3):
    ''' 
    to handle gauge invariance. take 3 quark props of definite spin and perform color contractions
        e.g. q1[:,:,:,:,sf,si,:,:]
        eps_a,b,c eps_d,e,f q1[a,d] q2[b,e] q3[c,f]
        Antisymmetry makes 1/2 of terms negative
        -----
        Returns:
        quark propagator
        TODO: USE PERMUTATION LIB TO DO THIS
    '''
    result  = q1[:,:,:,:,0,0] * q2[:,:,:,:,1,1] * q3[:,:,:,:,2,2]
    result -= q1[:,:,:,:,0,0] * q2[:,:,:,:,1,2] * q3[:,:,:,:,2,1]
    result -= q1[:,:,:,:,0,1] * q2[:,:,:,:,1,0] * q3[:,:,:,:,2,2]
    result += q1[:,:,:,:,0,1] * q2[:,:,:,:,1,2] * q3[:,:,:,:,2,0]
    result += q1[:,:,:,:,0,2] * q2[:,:,:,:,1,0] * q3[:,:,:,:,2,1]
    result -= q1[:,:,:,:,0,2] * q2[:,:,:,:,1,1] * q3[:,:,:,:,2,0]
    result -= q1[:,:,:,:,0,0] * q2[:,:,:,:,2,1] * q3[:,:,:,:,1,2]
    result += q1[:,:,:,:,0,0] * q2[:,:,:,:,2,2] * q3[:,:,:,:,1,1]
    result += q1[:,:,:,:,0,1] * q2[:,:,:,:,2,0] * q3[:,:,:,:,1,2]
    result -= q1[:,:,:,:,0,1] * q2[:,:,:,:,2,2] * q3[:,:,:,:,1,0]
    result -= q1[:,:,:,:,0,2] * q2[:,:,:,:,2,0] * q3[:,:,:,:,1,1]
    result += q1[:,:,:,:,0,2] * q2[:,:,:,:,2,1] * q3[:,:,:,:,1,0]
    result -= q1[:,:,:,:,1,0] * q2[:,:,:,:,0,1] * q3[:,:,:,:,2,2]
    result += q1[:,:,:,:,1,0] * q2[:,:,:,:,0,2] * q3[:,:,:,:,2,1]
    result += q1[:,:,:,:,1,1] * q2[:,:,:,:,0,0] * q3[:,:,:,:,2,2]
    result -= q1[:,:,:,:,1,1] * q2[:,:,:,:,0,2] * q3[:,:,:,:,2,0]
    result -= q1[:,:,:,:,1,2] * q2[:,:,:,:,0,0] * q3[:,:,:,:,2,1]
    result += q1[:,:,:,:,1,2] * q2[:,:,:,:,0,1] * q3[:,:,:,:,2,0]
    result += q1[:,:,:,:,1,0] * q2[:,:,:,:,2,1] * q3[:,:,:,:,0,2]
    result -= q1[:,:,:,:,1,0] * q2[:,:,:,:,2,2] * q3[:,:,:,:,0,1]
    result -= q1[:,:,:,:,1,1] * q2[:,:,:,:,2,0] * q3[:,:,:,:,0,2]
    result += q1[:,:,:,:,1,1] * q2[:,:,:,:,2,2] * q3[:,:,:,:,0,0]
    result += q1[:,:,:,:,1,2] * q2[:,:,:,:,2,0] * q3[:,:,:,:,0,1]
    result -= q1[:,:,:,:,1,2] * q2[:,:,:,:,2,1] * q3[:,:,:,:,0,0]
    result += q1[:,:,:,:,2,0] * q2[:,:,:,:,0,1] * q3[:,:,:,:,1,2]
    result -= q1[:,:,:,:,2,0] * q2[:,:,:,:,0,2] * q3[:,:,:,:,1,1]
    result -= q1[:,:,:,:,2,1] * q2[:,:,:,:,0,0] * q3[:,:,:,:,1,2]
    result += q1[:,:,:,:,2,1] * q2[:,:,:,:,0,2] * q3[:,:,:,:,1,0]
    result += q1[:,:,:,:,2,2] * q2[:,:,:,:,0,0] * q3[:,:,:,:,1,1]
    result -= q1[:,:,:,:,2,2] * q2[:,:,:,:,0,1] * q3[:,:,:,:,1,0]
    result -= q1[:,:,:,:,2,0] * q2[:,:,:,:,1,1] * q3[:,:,:,:,0,2]
    result += q1[:,:,:,:,2,0] * q2[:,:,:,:,1,2] * q3[:,:,:,:,0,1]
    result += q1[:,:,:,:,2,1] * q2[:,:,:,:,1,0] * q3[:,:,:,:,0,2]
    result -= q1[:,:,:,:,2,1] * q2[:,:,:,:,1,2] * q3[:,:,:,:,0,0]
    result -= q1[:,:,:,:,2,2] * q2[:,:,:,:,1,0] * q3[:,:,:,:,0,1]
    result += q1[:,:,:,:,2,2] * q2[:,:,:,:,1,1] * q3[:,:,:,:,0,0]

    return result

def isospin_zero_spin_contract(q1,q2,q3,corr,spin):
    ''' 
    Color/spin contract a pair of lattice color matrix objects 
    with isospin 0. np denotes negative parity. 
    Baryons: 
    - lambda_z (dsu)
    - lambda_z_np (dsu)
    - omega_m (sss)
    - omega_m_np (sss)

    Parameters:
    -----------
    Input:
    - source and sink weights
    - source and sink matrices in Dirac spin space

    Output: 
    - spin-color matrix ?
    '''
    src_weights = np.zeros([4],dtype=np.complex128)
    src_weights[0] = 1.
    src_weights[1] = -1.
    src_weights[2] =  -1.
    src_weights[3] = 1.
    snk_weights = np.zeros([4],dtype=np.complex128)
    snk_weights[0] =  1.
    snk_weights[1] = -1.
    snk_weights[2] =  -1.
    snk_weights[3] = 1.

    src_spins = np.zeros([4,3],dtype=np.int)
    snk_spins = np.zeros([4,3],dtype=np.int)
    if corr in ['lambda_z', 'omega_m']:
        if spin == 'upup':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 0
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 0

            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 0;
            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 0;
            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 0;
            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 0;




        if spin == 'up':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 0; src_spins[1,1] = 1; src_spins[1,2] = 0;
            src_spins[2,0] = 0; src_spins[2,1] = 0; src_spins[2,2] = 1;
            src_spins[3,0] = 1; src_spins[3,1] = 0; src_spins[3,2] = 0;
            
            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 0; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 0; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 0; snk_spins[3,2] = 0;
        elif spin == 'dn':
            src_spins[0,0] = 1; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 1; src_spins[1,1] = 1; src_spins[1,2] = 0;
            src_spins[2,0] = 0; src_spins[2,1] = 1; src_spins[2,2] = 1;
            src_spins[3,0] = 1; src_spins[3,1] = 1; src_spins[3,2] = 0;

            snk_spins[0,0] = 1; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 1; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 1; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 1; snk_spins[3,2] = 0;
        else:
            print('unrecognized spin - aborting',spin)
            sys.exit(-1)
    elif corr in ['lambda_z_np', 'omega_m_np']:
        if spin == 'up':
            src_spins[0,0] = 2; src_spins[0,1] = 2; src_spins[0,2] = 3;
            src_spins[1,0] = 2; src_spins[1,1] = 3; src_spins[1,2] = 2;
            src_spins[2,0] = 2; src_spins[2,1] = 2; src_spins[2,2] = 3;
            src_spins[3,0] = 3; src_spins[3,1] = 2; src_spins[3,2] = 2;

            snk_spins[0,0] = 2; snk_spins[0,1] = 2; snk_spins[0,2] = 3;
            snk_spins[1,0] = 2; snk_spins[1,1] = 3; snk_spins[1,2] = 2;
            snk_spins[2,0] = 2; snk_spins[2,1] = 2; snk_spins[2,2] = 3;
            snk_spins[3,0] = 3; snk_spins[3,1] = 2; snk_spins[3,2] = 2;
        elif spin == 'dn':
            src_spins[0,0] = 3; src_spins[0,1] = 2; src_spins[0,2] = 3;
            src_spins[1,0] = 3; src_spins[1,1] = 3; src_spins[1,2] = 2;
            src_spins[2,0] = 2; src_spins[2,1] = 3; src_spins[2,2] = 3;
            src_spins[3,0] = 3; src_spins[3,1] = 3; src_spins[3,2] = 2;
            
            snk_spins[0,0] = 3; snk_spins[0,1] = 2; snk_spins[0,2] = 3;
            snk_spins[1,0] = 3; snk_spins[1,1] = 3; snk_spins[1,2] = 2;
            snk_spins[2,0] = 2; snk_spins[2,1] = 3; snk_spins[2,2] = 3;
            snk_spins[3,0] = 3; snk_spins[3,1] = 3; snk_spins[3,2] = 2;
        else:
            print('unrecognized spin - aborting',spin)
            sys.exit(-1)
    else:
        print('unrecognized corr',corr)
        sys.exit(-1)

    nt,nz,ny,nx = q1.shape[0:4]
    result = np.zeros([nt,nz,ny,nx],dtype=np.complex128)
    for sf,wf in enumerate(snk_weights):
        for si,wi in enumerate(src_weights):
            tmp1 = q1[:,:,:,:,snk_spins[sf,0],src_spins[si,0]]
            tmp2 = q2[:,:,:,:,snk_spins[sf,1],src_spins[si,1]]
            tmp3 = q3[:,:,:,:,snk_spins[sf,2],src_spins[si,2]]
            result += two_eps_color_contract(tmp1,tmp2,tmp3) * 1/2 * wi * wf
    return result

def isospin_half_spin_contract(q1,q2,q3,corr,spin):
    ''' Color/spin contract a pair of lattice color matrix objects 
    with isospin 1/2. 
    Baryons: 
    - Proton (duu)
    - Neutron (ddu)
    - xi_z (uss)
    - xi_m (dss)
    - xi_star_z (ssu)
    - xi_star_m (dss)
    
    Parameters:
    -----------
    Input:
    - source and sink weights
    - source and sink matrices in Dirac spin space

    Output: 
    - spin-color matrix ?
    '''
    src_weights = np.zeros([2],dtype=np.complex128)
    src_weights[0] = 1./np.sqrt(2)
    src_weights[1] = -1./np.sqrt(2)
    snk_weights = np.zeros([4],dtype=np.complex128)
    snk_weights[0] =  1./np.sqrt(2)
    snk_weights[1] = -1./np.sqrt(2)
    snk_weights[2] =  1./np.sqrt(2)
    snk_weights[3] = -1./np.sqrt(2)
    if corr in ['xi_z', 'xi_m', 'xi_z_np' , 'xi_m_np']:
        coeff = 4/3 # where does this coeff come from?? #
    else:
        coeff = 1
    src_spins = np.zeros([2,3],dtype=np.int)
    snk_spins = np.zeros([4,3],dtype=np.int)
    #positive parity =0
    if corr in ['proton', 'neutron', 'xi_z', 'xi_m', 'xi_star_z','xi_star_m']:
        #spin-up to spin-up
        if spin == 'up':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 0; src_spins[1,1] = 1; src_spins[1,2] = 0;

            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 0; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 0; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 0; snk_spins[3,2] = 0;
            #spin-down to spin-down
        elif spin == 'dn':
            src_spins[0,0] = 1; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 1; src_spins[1,1] = 1; src_spins[1,2] = 0;

            snk_spins[0,0] = 1; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 1; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 1; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 1; snk_spins[3,2] = 0;
        else:
            print('unrecognized spin - aborting',spin)
            sys.exit(-1)
    #negative parity = 1
    elif corr in ['proton_np', 'neutron_np', 'xi_z_np', 'xi_m_np','xi_star_z_np','xi_star_m_np']:
        if spin == 'up':
            src_spins[0,0] = 2; src_spins[0,1] = 2; src_spins[0,2] = 3;
            src_spins[1,0] = 2; src_spins[1,1] = 3; src_spins[1,2] = 2;

            snk_spins[0,0] = 2; snk_spins[0,1] = 2; snk_spins[0,2] = 3;
            snk_spins[1,0] = 2; snk_spins[1,1] = 3; snk_spins[1,2] = 2;
            snk_spins[2,0] = 2; snk_spins[2,1] = 2; snk_spins[2,2] = 3;
            snk_spins[3,0] = 3; snk_spins[3,1] = 2; snk_spins[3,2] = 2;
        elif spin == 'dn':
            src_spins[0,0] = 3; src_spins[0,1] = 2; src_spins[0,2] = 3;
            src_spins[1,0] = 3; src_spins[1,1] = 3; src_spins[1,2] = 2;

            snk_spins[0,0] = 3; snk_spins[0,1] = 2; snk_spins[0,2] = 3;
            snk_spins[1,0] = 3; snk_spins[1,1] = 3; snk_spins[1,2] = 2;
            snk_spins[2,0] = 2; snk_spins[2,1] = 3; snk_spins[2,2] = 3;
            snk_spins[3,0] = 3; snk_spins[3,1] = 3; snk_spins[3,2] = 2;
        else:
            print('unrecognized spin - aborting',spin)
            sys.exit(-1)
    else:
        print('unrecognized corr',corr)
        sys.exit(-1)

    nt,nz,ny,nx = q1.shape[0:4]
    result = np.zeros([nt,nz,ny,nx],dtype=np.complex128)
    for sf,wf in enumerate(snk_weights):
        for si,wi in enumerate(src_weights):
            tmp1 = q1[:,:,:,:,snk_spins[sf,0],src_spins[si,0]]
            tmp2 = q2[:,:,:,:,snk_spins[sf,1],src_spins[si,1]]
            tmp3 = q3[:,:,:,:,snk_spins[sf,2],src_spins[si,2]]
            result += two_eps_color_contract(tmp1,tmp2,tmp3) * wf * wi * coeff

    return result

def isospin_one_spin_contract(q1,q2,q3,corr,spin):
    ''' Color/spin contract a pair of lattice color matrix objects 
    with isospin 1. 
    Baryons(plus negative parity): 
    - sigma_p('s', 'u', 'u'),
    - sigma_z('d', 's', 'u'),
    - sigma_m('d', 'd', 's'),
    - sigma_star_p, ('s', 'u', 'u')
    - sigma_star_z, ('d', 's', 'u')
    - sigma_star_m, ('d', 'd', 's')

    Parameters:
    -----------
    Input:
    - source and sink weights
    - source and sink matrices in Dirac spin space

    Output: 
    - spin-color matrix ?
    '''
    src_weights = np.zeros([4],dtype=np.complex128)
    src_weights[0] = 1.
    src_weights[1] = -1.
    src_weights[2] = 1.
    src_weights[3] = -1.
    snk_weights = np.zeros([4],dtype=np.complex128)
    snk_weights[0] =  1.
    snk_weights[1] = -1.
    snk_weights[2] =  1.
    snk_weights[3] = -1.

    src_spins = np.zeros([4,3],dtype=np.int)
    snk_spins = np.zeros([4,3],dtype=np.int)
    if corr in ['sigma_z','sigma_p','sigma_m', 'sigma_star_z', 'sigma_star_p','sigma_star_m']:
        if spin == 'up':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 0; src_spins[1,1] = 1; src_spins[1,2] = 0;
            src_spins[2,0] = 0; src_spins[2,1] = 0; src_spins[2,2] = 1;
            src_spins[3,0] = 1; src_spins[3,1] = 0; src_spins[3,2] = 0;
            
            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 0; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 0; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 0; snk_spins[3,2] = 0;
        elif spin == 'dn':
            src_spins[0,0] = 1; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 1; src_spins[1,1] = 1; src_spins[1,2] = 0;
            src_spins[2,0] = 0; src_spins[2,1] = 1; src_spins[2,2] = 1;
            src_spins[3,0] = 1; src_spins[3,1] = 1; src_spins[3,2] = 0;

            snk_spins[0,0] = 1; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 1; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 1; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 1; snk_spins[3,2] = 0;
        else:
            print('unrecognized spin - aborting',spin)
            sys.exit(-1)
    elif corr in ['sigma_z_np','sigma_p_np', 'sigma_m_np', 'sigma_star_z_np', 'sigma_star_p_np','sigma_star_m_np']:
        if spin == 'up':
            src_spins[0,0] = 2; src_spins[0,1] = 2; src_spins[0,2] = 3;
            src_spins[1,0] = 2; src_spins[1,1] = 3; src_spins[1,2] = 2;
            src_spins[2,0] = 2; src_spins[2,1] = 2; src_spins[2,2] = 3;
            src_spins[3,0] = 3; src_spins[3,1] = 2; src_spins[3,2] = 2;
            
            snk_spins[0,0] = 2; snk_spins[0,1] = 2; snk_spins[0,2] = 3;
            snk_spins[1,0] = 2; snk_spins[1,1] = 3; snk_spins[1,2] = 2;
            snk_spins[2,0] = 2; snk_spins[2,1] = 2; snk_spins[2,2] = 3;
            snk_spins[3,0] = 3; snk_spins[3,1] = 2; snk_spins[3,2] = 2;
        elif spin == 'dn':
            src_spins[0,0] = 3; src_spins[0,1] = 2; src_spins[0,2] = 3;
            src_spins[1,0] = 3; src_spins[1,1] = 3; src_spins[1,2] = 2;
            src_spins[2,0] = 2; src_spins[2,1] = 3; src_spins[2,2] = 3;
            src_spins[3,0] = 3; src_spins[3,1] = 3; src_spins[3,2] = 2;
            
            snk_spins[0,0] = 3; snk_spins[0,1] = 2; snk_spins[0,2] = 3;
            snk_spins[1,0] = 3; snk_spins[1,1] = 3; snk_spins[1,2] = 2;
            snk_spins[2,0] = 2; snk_spins[2,1] = 3; snk_spins[2,2] = 3;
            snk_spins[3,0] = 3; snk_spins[3,1] = 3; snk_spins[3,2] = 2;
        else:
            print('unrecognized spin - aborting',spin)
            sys.exit(-1)
    else:
        print('unrecognized corr',corr)
        sys.exit(-1)

    nt,nz,ny,nx = q1.shape[0:4]
    result = np.zeros([nt,nz,ny,nx],dtype=np.complex128)
    for sf,wf in enumerate(snk_weights):
        for si,wi in enumerate(src_weights):
            tmp1 = q1[:,:,:,:,snk_spins[sf,0],src_spins[si,0]]
            tmp2 = q2[:,:,:,:,snk_spins[sf,1],src_spins[si,1]]
            tmp3 = q3[:,:,:,:,snk_spins[sf,2],src_spins[si,2]]
            result += two_eps_color_contract(tmp1,tmp2,tmp3) * wi * wf * np.sqrt(1/3) * np.sqrt(1/3)
    return result

def isospin_three_half_spin_contract(q1,q2,q3,corr,spin):
    ''' 
    Color/spin contract a pair of lattice color matrix objects 
    with isospin 1. 
    Baryons(plus negative parity):
    - delta_m ('d', 'd', 'd')  
    - delta_p('d', 'u', 'u')
    - delta_z('d', 'd', 'u')
    - delta_pp ('u', 'u', 'u')

    Parameters:
    -----------
    Input:
    - source and sink weights
    - source and sink matrices in Dirac spin space

    Output: 
    - spin-color matrix ?
    '''
    src_weights = np.zeros([2],dtype=np.complex128)
    src_weights[0] = 1./np.sqrt(2)
    src_weights[1] = -1./np.sqrt(2)
    snk_weights = np.zeros([4],dtype=np.complex128)
    snk_weights[0] =  1./np.sqrt(2)
    snk_weights[1] = -1./np.sqrt(2)
    snk_weights[2] =  1./np.sqrt(2)
    snk_weights[3] = -1./np.sqrt(2)
    if corr in ['delta_z' ,'delta_p']:
        coeff = 1/np.sqrt(3)
    else:
        coeff = 1 #????

    '''
    these are the general matrices in dirac spin basis, the spin projection
    matrices given by 
    T_i := (\sigma_i 0
            0       0)
        with standard gamma matrix basis used 
    '''
    src_spins = np.zeros([2,3],dtype=np.int)
    snk_spins = np.zeros([4,3],dtype=np.int)

    if corr in ['delta_pp']:
        if spin == 'upup':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 0;
            src_spins[1,0] = 0; src_spins[1,1] = 0; src_spins[1,2] = 0;

            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 0;
            snk_spins[1,0] = 0; snk_spins[1,1] = 0; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 0; snk_spins[2,2] = 0;
            snk_spins[3,0] = 0; snk_spins[3,1] = 0; snk_spins[3,2] = 0;

        elif spin == 'up':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 0;
            src_spins[1,0] = 0; src_spins[1,1] = 0; src_spins[1,2] = 0;

            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 0;
            snk_spins[1,0] = 0; snk_spins[1,1] = 0; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 0; snk_spins[2,2] = 0;
            snk_spins[3,0] = 0; snk_spins[3,1] = 0; snk_spins[3,2] = 0;

        elif spin == 'dn':

        elif spin == 'dndn':

        

    if corr in ['delta_p']:
        if spin == 'up':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 0; src_spins[1,1] = 1; src_spins[1,2] = 0;

            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 0; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 0; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 0; snk_spins[3,2] = 0;
        elif spin == 'dn':
            src_spins[0,0] = 1; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 1; src_spins[1,1] = 1; src_spins[1,2] = 0;

            snk_spins[0,0] = 1; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 1; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 1; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 1; snk_spins[3,2] = 0;

        elif spin =='upup':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 6;
            src_spins[1,0] = 0; src_spins[1,1] = 1; src_spins[1,2] = 0;

            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 6;
            snk_spins[1,0] = 0; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 0; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 0; snk_spins[3,2] = 0;
        elif spin =='dndn':
            src_spins[0,0] = 1; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 1; src_spins[1,1] = 1; src_spins[1,2] = 0;

            snk_spins[0,0] = 1; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 1; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 1; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 1; snk_spins[3,2] = 0;

        else:
            print('unrecognized spin - aborting',spin)
            sys.exit(-1)
    elif corr in ['delta_z_np' ,'delta_m_np' ,'delta_p_np' ,'delta_pp_np']:
        if spin == 'up':
            src_spins[0,0] = 2; src_spins[0,1] = 2; src_spins[0,2] = 3;
            src_spins[1,0] = 2; src_spins[1,1] = 3; src_spins[1,2] = 2;

            snk_spins[0,0] = 2; snk_spins[0,1] = 2; snk_spins[0,2] = 3;
            snk_spins[1,0] = 2; snk_spins[1,1] = 3; snk_spins[1,2] = 2;
            snk_spins[2,0] = 2; snk_spins[2,1] = 2; snk_spins[2,2] = 3;
            snk_spins[3,0] = 3; snk_spins[3,1] = 2; snk_spins[3,2] = 2;
        elif spin == 'dn':
            src_spins[0,0] = 3; src_spins[0,1] = 2; src_spins[0,2] = 3;
            src_spins[1,0] = 3; src_spins[1,1] = 3; src_spins[1,2] = 2;

            snk_spins[0,0] = 3; snk_spins[0,1] = 2; snk_spins[0,2] = 3;
            snk_spins[1,0] = 3; snk_spins[1,1] = 3; snk_spins[1,2] = 2;
            snk_spins[2,0] = 2; snk_spins[2,1] = 3; snk_spins[2,2] = 3;
            snk_spins[3,0] = 3; snk_spins[3,1] = 3; snk_spins[3,2] = 2;

        elif spin =='upup':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 0; src_spins[1,1] = 1; src_spins[1,2] = 0;

            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 0; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 0; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 0; snk_spins[3,2] = 0;

        elif spin =='dndn':
            src_spins[0,0] = 1; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 1; src_spins[1,1] = 1; src_spins[1,2] = 0;

            snk_spins[0,0] = 1; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 1; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 1; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 1; snk_spins[3,2] = 0;
        else:
            print('unrecognized spin - aborting',spin)
            sys.exit(-1)
    else:
        print('unrecognized corr',corr)
        sys.exit(-1)

    nt,nz,ny,nx = q1.shape[0:4]
    result = np.zeros([nt,nz,ny,nx],dtype=np.complex128)
    for sf,wf in enumerate(snk_weights):
        for si,wi in enumerate(src_weights):
            tmp1 = q1[:,:,:,:,snk_spins[sf,0],src_spins[si,0]]
            tmp2 = q2[:,:,:,:,snk_spins[sf,1],src_spins[si,1]]
            tmp3 = q3[:,:,:,:,snk_spins[sf,2],src_spins[si,2]]
            result += two_eps_color_contract(tmp1,tmp2,tmp3) * wf * wi * coeff

    return result

def sigma_lambda_spin_contract(q1,q2,q3,corr,spin):
    src_weights = np.zeros([4],dtype=np.complex128)
    src_weights[0] = 1.
    src_weights[1] = -1.
    src_weights[2] = 1.
    src_weights[3] = -1.
    snk_weights = np.zeros([4],dtype=np.complex128)
    snk_weights[0] =  1.
    snk_weights[1] = -1.
    snk_weights[2] =  -1.
    snk_weights[3] = 1.

    src_spins = np.zeros([4,3],dtype=np.int)
    snk_spins = np.zeros([4,3],dtype=np.int)
    if corr == 'sigma_to_lambda':
        if spin == 'up':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 0; src_spins[1,1] = 1; src_spins[1,2] = 0;
            src_spins[2,0] = 0; src_spins[2,1] = 0; src_spins[2,2] = 1;
            src_spins[3,0] = 1; src_spins[3,1] = 0; src_spins[3,2] = 0;
            
            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 0; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 0; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 0; snk_spins[3,2] = 0;
        elif spin == 'dn':
            src_spins[0,0] = 1; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 1; src_spins[1,1] = 1; src_spins[1,2] = 0;
            src_spins[2,0] = 0; src_spins[2,1] = 1; src_spins[2,2] = 1;
            src_spins[3,0] = 1; src_spins[3,1] = 1; src_spins[3,2] = 0;


            snk_spins[0,0] = 1; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 1; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 1; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 1; snk_spins[3,2] = 0;
        else:
            print('unrecognized spin - aborting',spin)
            sys.exit(-1)
    elif corr == 'sigma_to_lambda_np':
        if spin == 'up':
            src_spins[0,0] = 2; src_spins[0,1] = 2; src_spins[0,2] = 3;
            src_spins[1,0] = 2; src_spins[1,1] = 3; src_spins[1,2] = 2;
            src_spins[2,0] = 2; src_spins[2,1] = 2; src_spins[2,2] = 3;
            src_spins[3,0] = 3; src_spins[3,1] = 2; src_spins[3,2] = 2;
            
            snk_spins[0,0] = 2; snk_spins[0,1] = 2; snk_spins[0,2] = 3;
            snk_spins[1,0] = 2; snk_spins[1,1] = 3; snk_spins[1,2] = 2;
            snk_spins[2,0] = 2; snk_spins[2,1] = 2; snk_spins[2,2] = 3;
            snk_spins[3,0] = 3; snk_spins[3,1] = 2; snk_spins[3,2] = 2;
        elif spin == 'dn':
            src_spins[0,0] = 3; src_spins[0,1] = 2; src_spins[0,2] = 3;
            src_spins[1,0] = 3; src_spins[1,1] = 3; src_spins[1,2] = 2;
            src_spins[2,0] = 2; src_spins[2,1] = 3; src_spins[2,2] = 3;
            src_spins[3,0] = 3; src_spins[3,1] = 3; src_spins[3,2] = 2;


            snk_spins[0,0] = 3; snk_spins[0,1] = 2; snk_spins[0,2] = 3;
            snk_spins[1,0] = 3; snk_spins[1,1] = 3; snk_spins[1,2] = 2;
            snk_spins[2,0] = 2; snk_spins[2,1] = 3; snk_spins[2,2] = 3;
            snk_spins[3,0] = 3; snk_spins[3,1] = 3; snk_spins[3,2] = 2;
        else:
            print('unrecognized spin - aborting',spin)
            sys.exit(-1)
    else:
        print('unrecognized corr',corr)
        sys.exit(-1)

    nt,nz,ny,nx = q1.shape[0:4]
    result = np.zeros([nt,nz,ny,nx],dtype=np.complex128)
    for sf,wf in enumerate(snk_weights):
        for si,wi in enumerate(src_weights):
            tmp1 = q1[:,:,:,:,snk_spins[sf,0],src_spins[si,0]]
            tmp2 = q2[:,:,:,:,snk_spins[sf,1],src_spins[si,1]]
            tmp3 = q3[:,:,:,:,snk_spins[sf,2],src_spins[si,2]]
            result += two_eps_color_contract(tmp1,tmp2,tmp3) * wi * wf * (-1) * np.sqrt(2/3) * 1/2
    return result

def lambda_sigma_spin_contract(q1,q2,q3,corr,spin):
    src_weights = np.zeros([4],dtype=np.complex128)
    src_weights[0] = 1.
    src_weights[1] = -1.
    src_weights[2] = -1.
    src_weights[3] = 1.
    snk_weights = np.zeros([4],dtype=np.complex128)
    snk_weights[0] =  1.
    snk_weights[1] = -1.
    snk_weights[2] =  1.
    snk_weights[3] = -1.

    src_spins = np.zeros([4,3],dtype=np.int)
    snk_spins = np.zeros([4,3],dtype=np.int)
    if corr == 'lambda_to_sigma':
        if spin == 'up':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 0; src_spins[1,1] = 1; src_spins[1,2] = 0;
            src_spins[2,0] = 0; src_spins[2,1] = 0; src_spins[2,2] = 1;
            src_spins[3,0] = 1; src_spins[3,1] = 0; src_spins[3,2] = 0;
            
            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 0; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 0; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 0; snk_spins[3,2] = 0;
        elif spin == 'dn':
            src_spins[0,0] = 1; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 1; src_spins[1,1] = 1; src_spins[1,2] = 0;
            src_spins[2,0] = 0; src_spins[2,1] = 1; src_spins[2,2] = 1;
            src_spins[3,0] = 1; src_spins[3,1] = 1; src_spins[3,2] = 0;


            snk_spins[0,0] = 1; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 1; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 1; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 1; snk_spins[3,2] = 0;
        else:
            print('unrecognized spin - aborting',spin)
            sys.exit(-1)
    elif corr == 'lambda_to_sigma_np':
        if spin == 'up':
            src_spins[0,0] = 2; src_spins[0,1] = 2; src_spins[0,2] = 3;
            src_spins[1,0] = 2; src_spins[1,1] = 3; src_spins[1,2] = 2;
            src_spins[2,0] = 2; src_spins[2,1] = 2; src_spins[2,2] = 3;
            src_spins[3,0] = 3; src_spins[3,1] = 2; src_spins[3,2] = 2;
            
            snk_spins[0,0] = 2; snk_spins[0,1] = 2; snk_spins[0,2] = 3;
            snk_spins[1,0] = 2; snk_spins[1,1] = 3; snk_spins[1,2] = 2;
            snk_spins[2,0] = 2; snk_spins[2,1] = 2; snk_spins[2,2] = 3;
            snk_spins[3,0] = 3; snk_spins[3,1] = 2; snk_spins[3,2] = 2;
        elif spin == 'dn':
            src_spins[0,0] = 3; src_spins[0,1] = 2; src_spins[0,2] = 3;
            src_spins[1,0] = 3; src_spins[1,1] = 3; src_spins[1,2] = 2;
            src_spins[2,0] = 2; src_spins[2,1] = 3; src_spins[2,2] = 3;
            src_spins[3,0] = 3; src_spins[3,1] = 3; src_spins[3,2] = 2;


            snk_spins[0,0] = 3; snk_spins[0,1] = 2; snk_spins[0,2] = 3;
            snk_spins[1,0] = 3; snk_spins[1,1] = 3; snk_spins[1,2] = 2;
            snk_spins[2,0] = 2; snk_spins[2,1] = 3; snk_spins[2,2] = 3;
            snk_spins[3,0] = 3; snk_spins[3,1] = 3; snk_spins[3,2] = 2;
        else:
            print('unrecognized spin - aborting',spin)
            sys.exit(-1)
    else:
        print('unrecognized corr',corr)
        sys.exit(-1)

    nt,nz,ny,nx = q1.shape[0:4]
    result = np.zeros([nt,nz,ny,nx],dtype=np.complex128)
    for sf,wf in enumerate(snk_weights):
        for si,wi in enumerate(src_weights):
            tmp1 = q1[:,:,:,:,snk_spins[sf,0],src_spins[si,0]]
            tmp2 = q2[:,:,:,:,snk_spins[sf,1],src_spins[si,1]]
            tmp3 = q3[:,:,:,:,snk_spins[sf,2],src_spins[si,2]]
            result += two_eps_color_contract(tmp1,tmp2,tmp3) * wi * wf * (-1) * np.sqrt(2/3) * 1/2
    return result