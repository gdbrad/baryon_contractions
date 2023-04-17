'''
Author: Amy Nicholson
Futher modifications by Grant Bradley
'''

import numpy as np
import sys

import itertools
from itertools import permutations

# Gamma matrices
G_u = (1 / np.sqrt(2)) * np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

G_v = (1 / np.sqrt(2)) * np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, -1, 0]
])

# Kronecker delta function
def kronecker_delta(i, j):
    return 1 if i == j else 0

# Spin projectors
P_u_plus = np.zeros(shape=(4,4),dtype=np.complex128)
P_u_plus[0,0] = 1

P_u_min = np.zeros(shape=(4,4),dtype=np.complex128)
P_u_min[1,1] = 1
P_v_plus = np.zeros(shape=(4,4),dtype=np.complex128)
P_v_plus[2,2] = 1
P_v_min = np.zeros(shape=(4,4),dtype=np.complex128)
P_v_min[3,3] = 1

P={
    'pp':{'up':P_u_plus,'dn':P_u_min,'upup':P_u_plus,'dndn':P_u_min},
    'np':{'up':P_v_plus,'dn':P_v_min,'upup':P_v_min,'dndn':P_v_min}
}
Gamma={
    'pp':G_u,
    'np':G_v
    }
# Performing the contraction
# Contracting color indices with the epsilon tensor

def two_eps_color_contract_compact(q1,q2,q3):
    eps = np.zeros(shape=(3, 3, 3))
    eps[0, 1, 2] , eps[1, 2, 0] , eps[2, 0, 1] = 1,1,1
    eps[2, 1, 0] ,eps[0, 2, 1] , eps[1, 0, 2] = -1,-1,-1
    ''' take 3 quark props of definite spin and perform color contractions
        e.g. q1[:,:,:,:,sf,si,:,:]
        eps_a,b,c eps_d,e,f q1[m,n,a,d] q2[o,p,b,e] q3[q,r,c,f]
    '''
    return np.einsum('abc,def,xyztmnad,xyztopbe,xyztqrcf->xyztmnopqr',eps,eps,q1,q2,q3)

def isospin_half_spin_contract_(q1,q2,q3,corr,spin):
    parity='pp'
    if 'omega' in corr:
        coeff = 1/4
    else:
        coeff =1 
    if 'np' in corr:
        parity='np'
    colContractedQquarks=two_eps_color_contract_compact(q1,q2,q3)
    sinkProjs=np.einsum('ab,cd->abcd',P[parity][spin],Gamma[parity]) #For the sink
    result=np.einsum('ab,cd,efgh,xyztfbgchd->xyzt',P[parity][spin],Gamma[parity],sinkProjs,colContractedQquarks)

    srcProjs=np.einsum('ab,cd->acbd',P[parity][spin],Gamma[parity]) #for the src 
    result+=np.einsum('ab,cd,efgh,xyztfbgchd->xyzt',P[parity][spin],Gamma[parity],srcProjs,colContractedQquarks) * coeff

    return result

def two_eps_color_contract(q1,q2,q3):
    ''' take 3 quark props of definite spin and perform color contractions
        e.g. q1[:,:,:,:,sf,si,:,:]
        eps_a,b,c eps_d,e,f q1[a,d] q2[b,e] q3[c,f]
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


def isospin_zero_spin_contract_omega(q1,q2,q3,corr,spin):
    ''' 
    Color/spin contract a pair of lattice color matrix objects 
    with isospin 0. np denotes negative parity. 
    Baryons: 
    - omega_m (sss)
    - omega_m_np (sss)

    Parameters:
    -----------
    Input: 
    Spin wavefunctions:
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

    src_spins = np.zeros([4,3],dtype=int)
    snk_spins = np.zeros([4,3],dtype=int)

    if corr in ['omega_m'] :
        '''
        isospin factor
        '''
        coeff = 2
    else:
        coeff = 6
    if corr in ['omega_m']:
        if spin == 'upup':
            '''
            Each line in this "loop" is applying a lowering operator on a single quark spin.
             2 embeddings of the H irreducible rep. & 1 of G_1. 
            '''
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 0; src_spins[1,1] = 1; src_spins[1,2] = 0;
            src_spins[2,0] = 1; src_spins[2,1] = 0; src_spins[2,2] = 0;
            src_spins[3,0] = 0; src_spins[3,1] = 0; src_spins[3,2] = 1;

            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 0; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 1; snk_spins[2,1] = 0; snk_spins[2,2] = 0;
            snk_spins[3,0] = 0; snk_spins[3,1] = 1; snk_spins[3,2] = 0;
        elif spin == 'dndn':
            src_spins[0,0] = 1; src_spins[0,1] = 1; src_spins[0,2] = 0;
            src_spins[1,0] = 1; src_spins[1,1] = 1; src_spins[1,2] = 0;

            snk_spins[0,0] = 1; snk_spins[0,1] = 1; snk_spins[0,2] = 0;
            snk_spins[1,0] = 1; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 1; snk_spins[2,1] = 1; snk_spins[2,2] = 0;
            snk_spins[3,0] = 1; snk_spins[3,1] = 1; snk_spins[3,2] = 0;
        
        elif spin == 'up':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 0; src_spins[1,1] = 1; src_spins[1,2] = 0;
            # src_spins[2,0] = 0; src_spins[2,1] = 0; src_spins[2,2] = 1;
            # src_spins[3,0] = 1; src_spins[3,1] = 0; src_spins[3,2] = 0;

            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 0; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 0; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 0; snk_spins[3,2] = 0;
        elif spin == 'dn':
            src_spins[0,0] = 0; src_spins[0,1] = 1; src_spins[0,2] = 1;
            src_spins[1,0] = 1; src_spins[1,1] = 0; src_spins[1,2] = 1;
            # src_spins[2,0] = 1; src_spins[2,1] = 1; src_spins[2,2] = 0;
            # src_spins[3,0] = 1; src_spins[3,1] = 1; src_spins[3,2] = 0;

            snk_spins[0,0] = 0; snk_spins[0,1] = 1; snk_spins[0,2] = 1;
            snk_spins[1,0] = 1; snk_spins[1,1] = 0; snk_spins[1,2] = 1;
            snk_spins[2,0] = 1; snk_spins[2,1] = 1; snk_spins[2,2] = 0;
            snk_spins[3,0] = 1; snk_spins[3,1] = 1; snk_spins[3,2] = 0;

    else:
        print('unrecognized corr',corr)
        sys.exit(-1)

    nt,nz,ny,nx = q1.shape[0:4]
    result = np.zeros([nt,nz,ny,nx],dtype=np.complex128)
    for sf,wf in enumerate(snk_weights):
        print(sf,wf)
        for si,wi in enumerate(src_weights):
            print(si,wi)
            tmp1 = q1[:,:,:,:,snk_spins[sf,0],src_spins[si,0]]
            tmp2 = q2[:,:,:,:,snk_spins[sf,1],src_spins[si,1]]
            tmp3 = q3[:,:,:,:,snk_spins[sf,2],src_spins[si,2]]
            result += two_eps_color_contract(tmp1,tmp2,tmp3) * coeff *wi * wf 
    return result



def isospin_zero_spin_contract(q1,q2,q3,corr,spin):
    ''' 
    Color/spin contract a pair of lattice color matrix objects 
    with isospin 0. np denotes negative parity. 
    Baryons: 
    - lambda_z (dsu)
    - lambda_z_np (dsu)

    Parameters:
    -----------
    Input: 
    Spin wavefunctions:
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

    src_spins = np.zeros([4,3],dtype=int)
    snk_spins = np.zeros([4,3],dtype=int)
    if corr in ['lambda_z'] :
        coeff = 1
    else:
        coeff = 6
    if corr in ['lambda_z']:
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
            src_spins[0,0] = 0; src_spins[0,1] = 1; src_spins[0,2] = 1;
            src_spins[1,0] = 1; src_spins[1,1] = 0; src_spins[1,2] = 1;
            src_spins[2,0] = 1; src_spins[2,1] = 1; src_spins[2,2] = 0;
            src_spins[3,0] = 1; src_spins[3,1] = 1; src_spins[3,2] = 0;

            snk_spins[0,0] = 0; snk_spins[0,1] = 1; snk_spins[0,2] = 1;
            snk_spins[1,0] = 1; snk_spins[1,1] = 0; snk_spins[1,2] = 1;
            snk_spins[2,0] = 1; snk_spins[2,1] = 1; snk_spins[2,2] = 0;
            snk_spins[3,0] = 1; snk_spins[3,1] = 1; snk_spins[3,2] = 0;
        else:
            print('unrecognized spin - aborting',spin)
            sys.exit(-1)
    elif corr in ['lambda_z_np']:
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

    elif corr in ['omega_m']:
        if spin == 'upup':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 0; src_spins[1,1] = 0; src_spins[1,2] = 1;
            src_spins[2,0] = 0; src_spins[2,1] = 0; src_spins[2,2] = 1;
            src_spins[3,0] = 0; src_spins[3,1] = 0; src_spins[3,2] = 1;

            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 0; snk_spins[1,1] = 0; snk_spins[1,2] = 1;
            snk_spins[2,0] = 0; snk_spins[2,1] = 0; snk_spins[2,2] = 1;
            snk_spins[3,0] = 0; snk_spins[3,1] = 0; snk_spins[3,2] = 1;
        elif spin == 'dndn':
            src_spins[0,0] = 1; src_spins[0,1] = 1; src_spins[0,2] = 1;
            src_spins[1,0] = 1; src_spins[1,1] = 1; src_spins[1,2] = 1;

            snk_spins[0,0] = 1; snk_spins[0,1] = 1; snk_spins[0,2] = 1;
            snk_spins[1,0] = 1; snk_spins[1,1] = 1; snk_spins[1,2] = 1;
            snk_spins[2,0] = 1; snk_spins[2,1] = 1; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 1; snk_spins[3,2] = 1;
        
        elif spin == 'up':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 0; src_spins[1,1] = 1; src_spins[1,2] = 0;
            src_spins[2,0] = 0; src_spins[2,1] = 0; src_spins[2,2] = 1;
            src_spins[3,0] = 1; src_spins[3,1] = 0; src_spins[3,2] = 0;

            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 0; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 0; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 0; snk_spins[3,2] = 0;
        elif spin == 'dn':
            src_spins[0,0] = 0; src_spins[0,1] = 1; src_spins[0,2] = 1;
            src_spins[1,0] = 1; src_spins[1,1] = 0; src_spins[1,2] = 1;
            src_spins[2,0] = 1; src_spins[2,1] = 1; src_spins[2,2] = 0;
            src_spins[3,0] = 1; src_spins[3,1] = 1; src_spins[3,2] = 0;

            snk_spins[0,0] = 0; snk_spins[0,1] = 1; snk_spins[0,2] = 1;
            snk_spins[1,0] = 1; snk_spins[1,1] = 0; snk_spins[1,2] = 1;
            snk_spins[2,0] = 1; snk_spins[2,1] = 1; snk_spins[2,2] = 0;
            snk_spins[3,0] = 1; snk_spins[3,1] = 1; snk_spins[3,2] = 0;

    else:
        print('unrecognized corr',corr)
        sys.exit(-1)
    # Performing a color contraction then contraction on Dirac indices?
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
    - xi_z (ssu)
    - xi_m (dss)
    - xi_star_z (ssu)
    - xi_star_m (dss)
    
    Parameters:
    -----------
    Input:
    - source and sink weights ; Come from flavor structure?
    - source and sink spin matrices in Dirac spin space

    Output: 
    - spin-color matrix ?
    '''
    src_weights = np.zeros([2],dtype=np.complex128)
    src_weights[0] = 1/np.sqrt(2)
    src_weights[1] = -1/np.sqrt(2)
    snk_weights = np.zeros([4],dtype=np.complex128)
    snk_weights[0] =  1/np.sqrt(2)
    snk_weights[1] = -1/np.sqrt(2)
    snk_weights[2] =  1/np.sqrt(2)
    snk_weights[3] = -1/np.sqrt(2)
    if corr in ['xi_z', 'xi_m', 'xi_z_np' , 'xi_m_np','xi_star_z','xi_star_z_np']:
        coeff = 4/3 # where does this coeff come from?? #
    else:
        coeff = 1
    src_spins = np.zeros([2,3],dtype=int)
    snk_spins = np.zeros([4,3],dtype=int)
    #positive parity =0
    if corr in ['proton', 'neutron', 'xi_z', 'xi_m','xi_star_z']:
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
    elif corr in ['proton_np', 'neutron_np', 'xi_z_np', 'xi_m_np','xi_star_z_np']:
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

    elif corr in ['xi_star_z','xi_star_m']:
        if spin == 'up':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 0; src_spins[1,1] = 1; src_spins[1,2] = 0;

            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 0; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 0; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 0; snk_spins[3,2] = 0;
            #spin-down to spin-down
        elif spin == 'dn':
            src_spins[0,0] = 0; src_spins[0,1] = 1; src_spins[0,2] = 1;
            src_spins[1,0] = 1; src_spins[1,1] = 0; src_spins[1,2] = 1;

            snk_spins[0,0] = 0; snk_spins[0,1] = 1; snk_spins[0,2] = 1;
            snk_spins[1,0] = 1; snk_spins[1,1] = 0; snk_spins[1,2] = 1;
            snk_spins[2,0] = 0; snk_spins[2,1] = 1; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 1; snk_spins[3,2] = 0;
        
        elif spin == 'upup':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 0;
            src_spins[1,0] = 0; src_spins[1,1] = 0; src_spins[1,2] = 0;

            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 0;
            snk_spins[1,0] = 0; snk_spins[1,1] = 0; snk_spins[1,2] = 0;
            snk_spins[2,0] = 0; snk_spins[2,1] = 0; snk_spins[2,2] = 0;
            snk_spins[3,0] = 0; snk_spins[3,1] = 0; snk_spins[3,2] = 0;
        elif spin == 'dndn':
            src_spins[0,0] = 1; src_spins[0,1] = 1; src_spins[0,2] = 1;
            src_spins[1,0] = 1; src_spins[1,1] = 1; src_spins[1,2] = 1;

            snk_spins[0,0] = 1; snk_spins[0,1] = 1; snk_spins[0,2] = 1;
            snk_spins[1,0] = 1; snk_spins[1,1] = 1; snk_spins[1,2] = 1;
            snk_spins[2,0] = 1; snk_spins[2,1] = 1; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 1; snk_spins[3,2] = 1;


    elif corr in ['xi_star_z_np','xi_star_m_np']:
        if spin == 'up':
            src_spins[0,0] = 2; src_spins[0,1] = 2; src_spins[0,2] = 3;
            src_spins[1,0] = 2; src_spins[1,1] = 3; src_spins[1,2] = 2;

            snk_spins[0,0] = 2; snk_spins[0,1] = 2; snk_spins[0,2] = 3;
            snk_spins[1,0] = 2; snk_spins[1,1] = 3; snk_spins[1,2] = 2;
            snk_spins[2,0] = 2; snk_spins[2,1] = 2; snk_spins[2,2] = 3;
            snk_spins[3,0] = 3; snk_spins[3,1] = 2; snk_spins[3,2] = 2;
        elif spin == 'dn':
            src_spins[0,0] = 2; src_spins[0,1] = 3; src_spins[0,2] = 3;
            src_spins[1,0] = 3; src_spins[1,1] = 2; src_spins[1,2] = 3;

            snk_spins[0,0] = 3; snk_spins[0,1] = 2; snk_spins[0,2] = 3;
            snk_spins[1,0] = 3; snk_spins[1,1] = 3; snk_spins[1,2] = 2;
            snk_spins[2,0] = 2; snk_spins[2,1] = 3; snk_spins[2,2] = 3;
            snk_spins[3,0] = 3; snk_spins[3,1] = 3; snk_spins[3,2] = 2;

        elif spin =='upup':
            src_spins[0,0] = 2; src_spins[0,1] = 2; src_spins[0,2] = 2;
            src_spins[1,0] = 2; src_spins[1,1] = 2; src_spins[1,2] = 2;

            snk_spins[0,0] = 2; snk_spins[0,1] = 2; snk_spins[0,2] = 2;
            snk_spins[1,0] = 2; snk_spins[1,1] = 2; snk_spins[1,2] = 2;
            snk_spins[2,0] = 2; snk_spins[2,1] = 2; snk_spins[2,2] = 2;
            snk_spins[3,0] = 2; snk_spins[3,1] = 2; snk_spins[3,2] = 2;

        elif spin =='dndn':
            src_spins[0,0] = 3; src_spins[0,1] = 3; src_spins[0,2] = 3;
            src_spins[1,0] = 3; src_spins[1,1] = 3; src_spins[1,2] = 3;

            snk_spins[0,0] = 3; snk_spins[0,1] = 3; snk_spins[0,2] = 3;
            snk_spins[1,0] = 3; snk_spins[1,1] = 3; snk_spins[1,2] = 3;
            snk_spins[2,0] = 3; snk_spins[2,1] = 3; snk_spins[2,2] = 3;
            snk_spins[3,0] = 3; snk_spins[3,1] = 3; snk_spins[3,2] = 3;



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

    src_spins = np.zeros([4,3],dtype=int)
    snk_spins = np.zeros([4,3],dtype=int)
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
    with isospin 3/2, isospin projections I_z = -3/2,-1/2,1/2,3/2
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
    src_weights[0] = 1/2
    src_weights[1] = 1/2
    snk_weights = np.zeros([4],dtype=np.complex128)
    snk_weights[0] =  1/2
    snk_weights[1] = 1/2
    snk_weights[2] =  1/2
    snk_weights[3] = 1/2
    if corr in ['delta_pp','delta_pp_np','delta_m','delta_m_np']: 
        coeff = 1 #????
    else:
        coeff = 1/np.sqrt(3) #????

    '''
    these are the general matrices in dirac spin basis, the spin projection
    matrices given by 
    T_i := (\sigma_i 0
            0       0)
        with standard gamma matrix basis used 
    '''
    src_spins = np.zeros([4,3],dtype=int)
    snk_spins = np.zeros([4,3],dtype=int)
    if corr == 'delta_pp': # chgange all to 0 
        if spin =='upup':
            src_spins[0,0] = 0; src_spins[0,1] = 1; src_spins[0,2] = 0;
            src_spins[1,0] = 0; src_spins[1,1] = 0; src_spins[1,2] = 1;
            # src_spins[2,0] = 0; src_spins[2,1] = 0; src_spins[2,2] = 1;
            # src_spins[3,0] = 1; src_spins[3,1] = 0; src_spins[3,2] = 0;

            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 0; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            # snk_spins[2,0] = 0; snk_spins[2,1] = 0; snk_spins[2,2] = 1;
            # snk_spins[3,0] = 1; snk_spins[3,1] = 0; snk_spins[3,2] = 0;

        elif spin == 'up':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 0; src_spins[1,1] = 1; src_spins[1,2] = 0;
            # src_spins[2,0] = 1; src_spins[2,1] = 0; src_spins[2,2] = 1;
            # src_spins[3,0] = 1; src_spins[3,1] = 1; src_spins[3,2] = 0;

            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 0; snk_spins[1,1] = 1; snk_spins[1,2] = 0;
            snk_spins[2,0] = 1; snk_spins[2,1] = 0; snk_spins[2,2] = 0;
            snk_spins[3,0] = 0; snk_spins[3,1] = 0; snk_spins[3,2] = 1;

        elif spin == 'dn':
            src_spins[0,0] = 0; src_spins[0,1] = 0; src_spins[0,2] = 1;
            src_spins[1,0] = 1; src_spins[1,1] = 0; src_spins[1,2] = 0;
            # src_spins[2,0] = 1; src_spins[2,1] = 0; src_spins[2,2] = 1;
            # src_spins[3,0] = 1; src_spins[3,1] = 1; src_spins[3,2] = 0;

            snk_spins[0,0] = 0; snk_spins[0,1] = 0; snk_spins[0,2] = 1;
            snk_spins[1,0] = 1; snk_spins[1,1] = 0; snk_spins[1,2] = 0;
            snk_spins[2,0] = 1; snk_spins[2,1] = 0; snk_spins[2,2] = 0;
            snk_spins[3,0] = 0; snk_spins[3,1] = 1; snk_spins[3,2] = 0;

        elif spin =='dndn':
            src_spins[0,0] = 1; src_spins[0,1] = 1; src_spins[0,2] = 1;
            src_spins[1,0] = 1; src_spins[1,1] = 1; src_spins[1,2] = 1;
            # src_spins[2,0] = 1; src_spins[2,1] = 0; src_spins[2,2] = 1;
            # src_spins[3,0] = 1; src_spins[3,1] = 1; src_spins[3,2] = 0;

            snk_spins[0,0] = 1; snk_spins[0,1] = 1; snk_spins[0,2] = 1;
            snk_spins[1,0] = 1; snk_spins[1,1] = 1; snk_spins[1,2] = 1;
            snk_spins[2,0] = 1; snk_spins[2,1] = 1; snk_spins[2,2] = 1;
            snk_spins[3,0] = 1; snk_spins[3,1] = 1; snk_spins[3,2] = 1;

    elif corr == 'delta_pp_np':
        if spin == 'up':
            src_spins[0,0] = 2; src_spins[0,1] = 2; src_spins[0,2] = 3;
            src_spins[1,0] = 2; src_spins[1,1] = 3; src_spins[1,2] = 2;
            # src_spins[2,0] = 2; src_spins[2,1] = 2; src_spins[2,2] = 2;
            # src_spins[3,0] = 2; src_spins[3,1] = 2; src_spins[3,2] = 2;

            snk_spins[0,0] = 2; snk_spins[0,1] = 2; snk_spins[0,2] = 3;
            snk_spins[1,0] = 2; snk_spins[1,1] = 3; snk_spins[1,2] = 2;
            snk_spins[2,0] = 3; snk_spins[2,1] = 2; snk_spins[2,2] = 2;
            snk_spins[3,0] = 2; snk_spins[3,1] = 2; snk_spins[3,2] = 3;
        elif spin == 'dn':
            src_spins[0,0] = 2; src_spins[0,1] = 2; src_spins[0,2] = 3;
            src_spins[1,0] = 2; src_spins[1,1] = 3; src_spins[1,2] = 2;
            # src_spins[2,0] = 2; src_spins[2,1] = 2; src_spins[2,2] = 3;
            # src_spins[3,0] = 2; src_spins[3,1] = 3; src_spins[3,2] = 2;

            snk_spins[0,0] = 3; snk_spins[0,1] = 2; snk_spins[0,2] = 3;
            snk_spins[1,0] = 2; snk_spins[1,1] = 3; snk_spins[1,2] = 3;
            snk_spins[2,0] = 2; snk_spins[2,1] = 2; snk_spins[2,2] = 3;
            snk_spins[3,0] = 2; snk_spins[3,1] = 3; snk_spins[3,2] = 3;

        elif spin =='upup':
            src_spins[0,0] = 2; src_spins[0,1] = 2; src_spins[0,2] = 2;
            src_spins[1,0] = 2; src_spins[1,1] = 2; src_spins[1,2] = 2;
            # src_spins[2,0] = 1; src_spins[2,1] = 0; src_spins[2,2] = 0;
            # src_spins[3,0] = 1; src_spins[3,1] = 0; src_spins[3,2] = 0;

            snk_spins[0,0] = 2; snk_spins[0,1] = 2; snk_spins[0,2] = 2;
            snk_spins[1,0] = 2; snk_spins[1,1] = 2; snk_spins[1,2] = 2;
            snk_spins[2,0] = 2; snk_spins[2,1] = 2; snk_spins[2,2] = 2;
            snk_spins[3,0] = 2; snk_spins[3,1] = 2; snk_spins[3,2] = 2;

        elif spin =='dndn':
            src_spins[0,0] = 3; src_spins[0,1] = 3; src_spins[0,2] = 2;
            src_spins[1,0] = 3; src_spins[1,1] = 2; src_spins[1,2] = 3;
            # src_spins[2,0] = 1; src_spins[2,1] = 0; src_spins[2,2] = 0;
            # src_spins[3,0] = 1; src_spins[3,1] = 0; src_spins[3,2] = 0;

            snk_spins[0,0] = 3; snk_spins[0,1] = 3; snk_spins[0,2] = 2;
            snk_spins[1,0] = 3; snk_spins[1,1] = 2; snk_spins[1,2] = 3;
            snk_spins[2,0] = 2; snk_spins[2,1] = 3; snk_spins[2,2] = 3;
            snk_spins[3,0] = 3; snk_spins[3,1] = 3; snk_spins[3,2] = 2;

    # elif corr in 'delta_m','delta_p','delta_z'
    # ,'delta_m_np','delta_p_np','delta_z_np'
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
            result += two_eps_color_contract(tmp1,tmp2,tmp3) * wf * wi  * 12

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

    src_spins = np.zeros([4,3],dtype=int)
    snk_spins = np.zeros([4,3],dtype=int)
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

    src_spins = np.zeros([4,3],dtype=int)
    snk_spins = np.zeros([4,3],dtype=int)
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




    