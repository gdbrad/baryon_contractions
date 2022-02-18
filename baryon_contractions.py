import numpy as np
import sys

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

def isospin_half_spin_contract(q1,q2,q3,corr,spin):
    src_weights = np.zeros([2],dtype=np.complex128)
    src_weights[0] = 1./np.sqrt(2)
    src_weights[1] = -1./np.sqrt(2)
    snk_weights = np.zeros([4],dtype=np.complex128)
    snk_weights[0] =  1./np.sqrt(2)
    snk_weights[1] = -1./np.sqrt(2)
    snk_weights[2] =  1./np.sqrt(2)
    snk_weights[3] = -1./np.sqrt(2)
    if corr == 'xi_z' or corr == 'xi_m' or corr == 'xi_z_np' or corr == 'xi_m_np':
        coeff = 4/3
    else:
        coeff = 1

    src_spins = np.zeros([2,3],dtype=np.int)
    snk_spins = np.zeros([4,3],dtype=np.int)
    if corr == 'proton' or corr == 'neutron' or corr == 'xi_z' or corr == 'xi_m' or corr == 'lambda_z':
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
        else:
            print('unrecognized spin - aborting',spin)
            sys.exit(-1)
    elif corr == 'proton_np' or corr == 'neutron_np' or corr == 'xi_z_np' or corr == 'xi_m_np':
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
    if corr == 'sigma_z' or corr == 'sigma_p' or corr == 'sigma_m':
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
    elif corr == 'sigma_z_np' or corr == 'sigma_p_np' or corr == 'sigma_m_np':
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

def isospin_zero_spin_contract(q1,q2,q3,corr,spin):
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
    if corr == 'lambda_z':
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
    elif corr == 'lambda_z_np':
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

