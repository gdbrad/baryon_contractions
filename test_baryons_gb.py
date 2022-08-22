import numpy as np
import h5py as h5
import baryon_contractions_gb as contractions
import gamma
import sys

U    = gamma.U_DR_to_DP
Uadj = gamma.U_DR_to_DP_adj

known_path = sys.argv[1]

f = h5.File(known_path+'/test_propagator.h5')

ps_prop_up = f['sh_sig2p0_n5/PS_up'][()]
ps_prop_down = f['sh_sig2p0_n5/PS_dn'][()]
ps_prop_strange = f['sh_sig2p0_n5/PS_strange'][()]
# fh props
f_fh = h5.File(known_path + '/test_fh_propagator.h5')
ps_fh_dn_A3     = f_fh['PS/fh_dn_A3'][()]
ps_fh_dn_V4     = f_fh['PS/fh_dn_V4'][()]
ps_fh_up_A3     = f_fh['PS/fh_up_A3'][()]
ps_fh_up_V4     = f_fh['PS/fh_up_V4'][()]
ps_fh_up_dn_A3  = f_fh['PS/fh_up_dn_A3'][()]
ps_fh_up_dn_V4  = f_fh['PS/fh_up_dn_V4'][()]
f.close()
'''
Rotate from Euclidean Degrand-Rossi to Euclidean Dirac-Pauli basis:
SpinMatrix U = DiracToDRMat();
quark_to_be_rotated = Uadj*quark_to_be_rotated*U
'''
ps_DP_up = np.einsum('ik,tzyxklab,lj->tzyxijab',Uadj,ps_prop_up,U)
ps_DP_down = np.einsum('ik,tzyxklab,lj->tzyxijab',Uadj,ps_prop_down,U)
ps_DP_strange = np.einsum('ik,tzyxklab,lj->tzyxijab',Uadj,ps_prop_strange,U)

ps_fh_DP_up = np.einsum('ik,tzyxklab,lj->tzyxijab',Uadj,ps_fh_up_A3,U)
ps_fh_DP_down = np.einsum('ik,tzyxklab,lj->tzyxijab',Uadj,ps_fh_dn_A3,U)
# ps_fh_DP_strange = np.einsum('ik,tzyxklab,lj->tzyxijab',Uadj,ps_prop_strange,U)

print(known_path+'/test_propagator.h5/sh_sig2p0_n5/PS_prop shape')
print(ps_DP_up.shape)
Nt = ps_DP_up.shape[0]


print(gamma)




# print('\nPS')
# print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
# for corr in ['delta_m','delta_m_np']:
#     for spin in ['up','dn']:
#         print(corr,spin)
#         baryon = contractions.isospin_three_half_spin_contract(ps_DP_strange,ps_DP_strange,ps_DP_down,corr,spin)
#         baryon_time = np.einsum('tzyx->t',baryon)
#         '''
#         for t in range(Nt):
#             print(t,baryon_up_time[t])
#         '''
#         f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
#         known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
#         f.close()
#         if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
#             print('    FAIL')
#         else:
#             print('    PASS')

print('\nPS')
print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
for corr in ['delta_pp','delta_pp_np']:
    for spin in ['up','dn','upup','dndn']:
        print(corr,spin)
        baryon = contractions.isospin_three_half_spin_contract(ps_DP_up,ps_DP_up,ps_DP_up,corr,spin)
        baryon_time = np.einsum('tzyx->t',baryon)

        # fh_UU = np.complex128(0)
        # fh_UU+= contractions.isospin_three_half_spin_contract(ps_fh_DP_up,ps_DP_up,ps_DP_up, corr, spin) 
        # fh_UU+= contractions.isospin_three_half_spin_contract(ps_DP_up,ps_fh_DP_up,ps_DP_up, corr, spin)
        # fh_UU+= contractions.isospin_three_half_spin_contract(ps_DP_up,ps_DP_up,ps_fh_DP_up, corr, spin)
        # print(fh_UU)
        # f_fh = h5.File(known_path+'/lalibe_fh_proton.h5')
        # known_fh_UU = {}
        # known_fh_UU['A3'] = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        # known_fh_UU['V4'] = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        
        
        for t in range(Nt):
            print(t,baryon_time[t])
        
        f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
        known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        print('\n',known_baryon)
        
        f.close()
        if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-6):
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

        fh_UU = np.complex128(0)
        fh_DD = np.complex128(0)

        fh_UU+= contractions.isospin_zero_spin_contract(ps_fh_DP_up,ps_DP_down,ps_DP_strange, corr, spin) 
        fh_DD+= contractions.isospin_zero_spin_contract(ps_DP_up,ps_fh_DP_down,ps_DP_strange, corr, spin)
        
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
for corr in ['omega_m','omega_m_np']:
    for spin in ['up','dn']:
        print(corr,spin)
        baryon = contractions.isospin_zero_spin_contract(ps_DP_strange,ps_DP_strange,ps_DP_strange,corr,spin)
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
for corr in ['proton','proton_np']:
    for spin in ['up','dn']:
        print(corr,spin)
        baryon = contractions.isospin_half_spin_contract(ps_DP_up,ps_DP_up,ps_DP_down,corr,spin)
        baryon_time = np.einsum('tzyx->t',baryon)
        fh_UU = np.complex128(0)
        fh_UU+= contractions.isospin_half_spin_contract(ps_fh_DP_up,ps_DP_up,ps_DP_down, corr, spin) 
        fh_UU+= contractions.isospin_half_spin_contract(ps_DP_up,ps_fh_DP_up,ps_DP_down, corr, spin)
        fh_DD= contractions.isospin_half_spin_contract(ps_DP_up,ps_DP_up,ps_fh_DP_down, corr, spin)
        # print(fh_UU)

        '''
        for t in range(Nt):
            print(t,baryon_up_time[t])
        '''
        f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
        known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

        f_fh = h5.File(known_path+'/lalibe_fh_proton.h5') 
        known_fh_UU = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        known_fh_DD = f_fh['PS/fh_'+corr+'_A3_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

        known_fh_UU = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        known_fh_DD = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

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
for corr in ['sigma_star_p','sigma_star_p_np']:
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
for corr in ['xi_z','xi_z_np']:
    for spin in ['up','dn']:
        print(corr,spin)
        baryon = contractions.isospin_half_spin_contract(ps_DP_strange,ps_DP_strange,ps_DP_up,corr,spin)
        baryon_time = np.einsum('tzyx->t',baryon)

        fh_UU = np.complex128(0)
        fh_UU+= contractions.isospin_half_spin_contract(ps_fh_DP_up,ps_DP_strange,ps_DP_strange, corr, spin) 
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
for corr in ['xi_star_z','xi_star_z_np']:
    for spin in ['up','dn','upup','dndn']:
        print(corr,spin)
        baryon = contractions.isospin_half_spin_contract(ps_DP_strange,ps_DP_strange,ps_DP_up,corr,spin)

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


            
