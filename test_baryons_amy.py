import numpy as np
import h5py as h5
import baryon_contractions as contractions
import gamma
import sys

U    = gamma.U_DR_to_DP
Uadj = gamma.U_DR_to_DP_adj

known_path = sys.argv[1]

f = h5.File(known_path+'/test_propagator.h5')

ps_prop_up = f['sh_sig2p0_n5/PS_up'][()]
ps_prop_down = f['sh_sig2p0_n5/PS_dn'][()]
ps_prop_strange = f['sh_sig2p0_n5/PS_strange'][()]
f.close()

# Rotate from Degrand-Rossi to Dirac-Pauli basis
ps_DP_up = np.einsum('ik,tzyxklab,lj->tzyxijab',Uadj,ps_prop_up,U)
ps_DP_down = np.einsum('ik,tzyxklab,lj->tzyxijab',Uadj,ps_prop_down,U)
ps_DP_strange = np.einsum('ik,tzyxklab,lj->tzyxijab',Uadj,ps_prop_strange,U)
print(known_path+'/test_propagator.h5/sh_sig2p0_n5/PS_prop shape')
print(ps_DP_up.shape)
Nt = ps_DP_up.shape[0]



print('\nPS')
print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
for corr in ['neutron','neutron_np']:
    for spin in ['up','dn']:
        print(corr,spin)
        baryon = contractions.isospin_half_spin_contract(ps_DP_down,ps_DP_down,ps_DP_up,corr,spin)
        baryon_time = np.einsum('tzyx->t',baryon)
        '''
        for t in range(Nt):
            print(t,baryon_up_time[t])
        '''
        f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
        known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        f.close()
        if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
            print('    FAIL')
        else:
            print('    PASS')

print('\nPS')
print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
for corr in ['xi_m','xi_m_np']:
    for spin in ['up','dn']:
        print(corr,spin)
        baryon = contractions.isospin_half_spin_contract(ps_DP_strange,ps_DP_strange,ps_DP_down,corr,spin)
        baryon_time = np.einsum('tzyx->t',baryon)
        '''
        for t in range(Nt):
            print(t,baryon_up_time[t])
        '''
        f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
        known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        f.close()
        if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
            print('    FAIL')
        else:
            print('    PASS')


print('\nPS')
print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
for corr in ['lambda_z','lambda_z_np']:
    for spin in ['up','dn']:
        print(corr,spin)
        baryon = contractions.isospin_zero_spin_contract(ps_DP_up,ps_DP_down,ps_DP_strange,corr,spin)
        baryon_time = np.einsum('tzyx->t',baryon)
        '''
        for t in range(Nt):
            print(t,baryon_up_time[t])
        '''
        f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
        known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        f.close()
        if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
            print('    FAIL')
        else:
            print('    PASS')

print('\nPS')
print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
for corr in ['sigma_p','sigma_p_np']:
    for spin in ['up','dn']:
        print(corr,spin)
        baryon = contractions.isospin_one_spin_contract(ps_DP_up,ps_DP_up,ps_DP_strange,corr,spin)
        baryon_time = np.einsum('tzyx->t',baryon)
        '''
        for t in range(Nt):
            print(t,baryon_up_time[t])
        '''
        f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
        known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        f.close()
        if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
            print('    FAIL')
        else:
            print('    PASS')

print('\nPS')
print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
for corr in ['lambda_to_sigma','lambda_to_sigma_np']:
    for spin in ['up','dn']:
        print(corr,spin)
        baryon = contractions.lambda_sigma_spin_contract(ps_DP_up,ps_DP_down,ps_DP_strange,corr,spin)
        baryon_time = np.einsum('tzyx->t',baryon)
        '''
        for t in range(Nt):
            print(t,baryon_up_time[t])
        '''
        f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
        known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        f.close()
        if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
            print('    FAIL')
        else:
            print('    PASS')

print('\nPS')
print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
for corr in ['sigma_to_lambda','sigma_to_lambda_np']:
    for spin in ['up','dn']:
        print(corr,spin)
        baryon = contractions.sigma_lambda_spin_contract(ps_DP_up,ps_DP_down,ps_DP_strange,corr,spin)
        baryon_time = np.einsum('tzyx->t',baryon)
        '''
        for t in range(Nt):
            print(t,baryon_up_time[t])
        '''
        f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
        known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        f.close()
        if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
            print('    FAIL')
        else:
            print('    PASS')

            
