import numpy as np
import h5py as h5
import baryon_contractions_gb as contractions
import gamma
import sys
import argparse

'''
Only testing maximal isospin for 2pt corrs. Entire octet/decuplet for FH coors.
'''

# def main():
#     parser = argparse.ArgumentParser(description = 
        
#         "testing baryon 2pt and FH contractions")
        
#     parser.add_argument('--states',      nargs='+',
#                     help=            'specify states to test?')
#     args = parser.parse_args()



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

print(ps_DP_up,"UP")
ps_fh_DP_up = dict()
ps_fh_DP_down = dict()
ps_fh_DP_up['A3'] = np.einsum('ik,tzyxklab,lj->tzyxijab',Uadj,ps_fh_up_A3,U)
ps_fh_DP_down['A3'] = np.einsum('ik,tzyxklab,lj->tzyxijab',Uadj,ps_fh_dn_A3,U)

ps_fh_DP_up['V4'] = np.einsum('ik,tzyxklab,lj->tzyxijab',Uadj,ps_fh_up_V4,U)
ps_fh_DP_down['V4'] = np.einsum('ik,tzyxklab,lj->tzyxijab',Uadj,ps_fh_dn_V4,U)

print(known_path+'/test_propagator.h5/sh_sig2p0_n5/PS_prop shape')
print(ps_DP_up.shape)
Nt = ps_DP_up.shape[0]

print('\nPS')
print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
for corr in ['proton','proton_np']:
    for spin in ['up','dn']:
        print(corr,spin)
        baryon_orig = contractions.isospin_half_spin_contract(ps_DP_up,ps_DP_up,ps_DP_down,corr,spin)
        baryon = contractions.isospin_half_spin_contract_(ps_DP_up,ps_DP_up,ps_DP_down,corr,spin)
        print(baryon.shape,"colorcont")
        print(baryon_orig.shape,"orig")
        baryon_time = np.einsum('tzyx->t',baryon)
        '''
        for t in range(Nt):
            print(t,baryon_up_time[t])
        '''
        f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
        known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

        if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
            print(corr,spin,'    FAIL')
        else:
            print(corr,spin,'    PASS')


print('\nPS')
print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
for corr in ['lambda_z','lambda_z_np']: #dsu
    for spin in ['up','dn']:
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
            print(corr,spin,'    FAIL')
        else:
            print(corr,spin,'    PASS')
        '''
        fh contraction routine
        '''
        # fh_UU_A3 = np.complex128(0)
        # fh_DD_A3 = np.complex128(0)
        # fh_UU_V4 = np.complex128(0)
        # fh_DD_V4 = np.complex128(0)

        # fh_UU_A3+= contractions.isospin_zero_spin_contract(ps_fh_DP_up['A3'],ps_DP_down,ps_DP_strange, corr, spin) 
        # fh_DD_A3+= contractions.isospin_zero_spin_contract(ps_DP_up,ps_fh_DP_down['A3'],ps_DP_strange, corr, spin)

        # fh_UU_V4 += contractions.isospin_zero_spin_contract(ps_fh_DP_up['V4'],ps_DP_down,ps_DP_strange, corr, spin) 
        # fh_DD_V4 +=  contractions.isospin_zero_spin_contract(ps_DP_up,ps_fh_DP_down['V4'],ps_DP_strange, corr, spin)

        # f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
        # known_fh = {}
        # known_fh['A3_UU'] = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        # known_fh['V4_UU'] = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        # known_fh['A3_DD'] = f_fh['PS/fh_'+corr+'_A3_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        # known_fh['V4_DD'] = f_fh['PS/fh_'+corr+'_V4_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

        # fh_baryon_time = {} 
        # fh_baryon_time['A3_UU'] = np.einsum('tzyx->t',fh_UU_A3)
        # fh_baryon_time['V4_UU'] = np.einsum('tzyx->t',fh_UU_V4)

        # fh_baryon_time['A3_DD'] = np.einsum('tzyx->t',fh_DD_A3)
        # fh_baryon_time['V4_DD'] = np.einsum('tzyx->t',fh_DD_V4)


        # # print(fh_baryon_time,"fhtime")
        # for curr_contract in ['A3_UU','V4_UU','A3_DD','V4_DD']:
        #         if np.any(abs(np.real(fh_baryon_time[curr_contract] - known_fh[curr_contract])/np.real(fh_baryon_time[curr_contract] + known_fh[curr_contract])) > 1.e-7):
        #             print(curr_contract,'    FAIL')
        #         else:
        #             print(curr_contract,'    PASS')
        # # print(known_fh,"fh")
for corr in ['omega_m']:#,'omega_m_np']: #dsu
    for spin in ['up','dn','upup','dndn']:
        baryon = contractions.isospin_half_spin_contract_(ps_DP_strange,ps_DP_strange,ps_DP_strange,corr,spin)
        baryon_time = np.einsum('tzyx->t',baryon)

        '''
        for t in range(Nt):
            print(t,baryon_up_time[t])
        '''
        f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
        known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        f.close()
        for t in range(Nt):
            # print(known_baryon[t],"known")
            print(t,baryon_time[t],"known" ,known_baryon[t]) 
            print(baryon_time[t]/ known_baryon[t])
        if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
            print(corr,spin,'    FAIL')
        else:
            print(corr,spin,'    PASS')

for corr in ['proton','proton_np']:
    for spin in ['up','dn']:
        baryon = contractions.isospin_half_spin_contract(ps_DP_up,ps_DP_up,ps_DP_down,corr,spin)
        baryon_time = np.einsum('tzyx->t',baryon)

        '''
        for t in range(Nt):
            print(t,baryon_up_time[t])
        '''
        f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
        known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        f.close()
        for t in range(Nt):
            # print(known_baryon[t],"known")
            print(t,baryon_time[t],"known" ,known_baryon[t]) 
            print(baryon_time[t]/ known_baryon[t])
        if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
            print(corr,spin,'    FAIL')
        else:
            print(corr,spin,'    PASS')
        
if False:

    for corr in ['delta_pp','delta_pp_np']:
        for spin in ['up','dn','upup','dndn']:
            print(corr,spin)
            baryon = contractions.isospin_three_half_spin_contract(ps_DP_up,ps_DP_up,ps_DP_up,corr,spin)
            baryon_time = np.einsum('tzyx->t',baryon)
            f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
            known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
            # print('\n',known_baryon)

            for t in range(Nt):
                # print(known_baryon[t],"known")
                print(t,baryon_time[t],"known" ,known_baryon[t]) 
                print(baryon_time[t]/ known_baryon[t])
            
            f.close()
            if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
                print('    FAIL')
            else:
                print('    PASS')
    print('\nPS FH')
    print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
    for corr in ['neutron','neutron_np']: #ddu
        for spin in ['up','dn']:
            print(corr,spin)
            # baryon = contractions.isospin_half_spin_contract(ps_DP_up,ps_DP_up,ps_DP_down,corr,spin)
            # baryon_time = np.einsum('tzyx->t',baryon)
            fh_DD_A3 = np.complex128(0)
            fh_DD_V4 = np.complex128(0)
            fh_UU_A3 = np.complex128(0)
            fh_UU_V4 = np.complex128(0)

            fh_DD_A3+= contractions.isospin_half_spin_contract(ps_fh_DP_down['A3'],ps_DP_down,ps_DP_up, corr, spin)
            fh_DD_A3+= contractions.isospin_half_spin_contract(ps_DP_down,ps_fh_DP_down['A3'],ps_DP_up, corr, spin)

            fh_DD_V4+= contractions.isospin_half_spin_contract(ps_fh_DP_down['V4'],ps_DP_down,ps_DP_up, corr, spin)
            fh_DD_V4+= contractions.isospin_half_spin_contract(ps_DP_down,ps_fh_DP_down['V4'],ps_DP_up, corr, spin)

            fh_UU_A3+= contractions.isospin_half_spin_contract(ps_DP_down,ps_DP_down, ps_fh_DP_up['A3'], corr, spin)
            fh_UU_V4+= contractions.isospin_half_spin_contract(ps_DP_down,ps_DP_down, ps_fh_DP_up['V4'], corr, spin)

            
            f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
            known_fh = {}
            known_fh['A3_DD'] = f_fh['PS/fh_'+corr+'_A3_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
            known_fh['V4_DD'] = f_fh['PS/fh_'+corr+'_V4_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
            known_fh['V4_UU'] = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
            known_fh['A3_UU'] = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]


            fh_baryon_time = {} 
            fh_baryon_time['A3_DD'] = np.einsum('tzyx->t',fh_DD_A3)
            fh_baryon_time['V4_DD'] = np.einsum('tzyx->t',fh_DD_V4)
            fh_baryon_time['V4_UU'] = np.einsum('tzyx->t',fh_UU_V4)
            fh_baryon_time['A3_UU'] = np.einsum('tzyx->t',fh_UU_A3)


            print(fh_baryon_time,"fhtime")
            for curr_contract in ['V4_UU','A3_UU','A3_DD','V4_DD']:
                    if np.any(abs(np.real(fh_baryon_time[curr_contract] - known_fh[curr_contract])/np.real(fh_baryon_time[curr_contract] + known_fh[curr_contract])) > 1.e-7):
                        print(curr_contract,'    FAIL')
                    else:
                        print(curr_contract,'    PASS')
            # print(fh_UU)

            # '''
            # for t in range(Nt):
            #     print(t,baryon_up_time[t])
            # '''
            # f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
            # known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

            # f.close()
            # if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
            #     print('    FAIL')
            # else:
            #     print('    PASS')

    
            # if False:

            #     # fh_UU_dict = dict()
            #     fh_UU_A3 = np.complex128(0)
            #     fh_UU_A3 += contractions.isospin_three_half_spin_contract(ps_fh_DP_up['A3'],ps_DP_up,ps_DP_up, corr, spin) 
            #     fh_UU_A3 += contractions.isospin_three_half_spin_contract(ps_DP_up,ps_fh_DP_up['A3'],ps_DP_up, corr, spin)
            #     fh_UU_A3 += contractions.isospin_three_half_spin_contract(ps_DP_up,ps_DP_up,ps_fh_DP_up['A3'], corr, spin)

            #     fh_UU_V4 = np.complex128(0)
            #     fh_UU_V4 += contractions.isospin_three_half_spin_contract(ps_fh_DP_up['V4'],ps_DP_up,ps_DP_up, corr, spin) 
            #     fh_UU_V4 += contractions.isospin_three_half_spin_contract(ps_DP_up,ps_fh_DP_up['V4'],ps_DP_up, corr, spin)
            #     fh_UU_V4 += contractions.isospin_three_half_spin_contract(ps_DP_up,ps_DP_up,ps_fh_DP_up['V4'], corr, spin)
            #     # print(fh_UU)
            #     f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
            #     known_fh_UU = {}
            #     known_fh_UU['A3'] = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
            #     known_fh_UU['V4'] = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

            #     fh_baryon_time = {} 
            #     fh_baryon_time['A3'] = np.einsum('tzyx->t',fh_UU_A3)
            #     fh_baryon_time['V4'] = np.einsum('tzyx->t',fh_UU_V4)

            #     print(fh_baryon_time,"fhtime")
            #     for curr in ['A3','V4']:
            #         if np.any(abs(np.real(fh_baryon_time[curr] - known_fh_UU[curr])/np.real(fh_baryon_time[curr] + known_fh_UU[curr])) > 1.e-7):
            #             print('    FAIL')
            #         else:
            #             print('    PASS')
            #     print(known_fh_UU,"fh")
            #     for t in range(Nt):
            #         # print(t,baryon_time[t])
    # print(gamma)
    if False:
        print('\nPS FH')
        print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
        for corr in ['delta_m','delta_m_np']:
            for spin in ['up','dn','upup','dndn']:
                print(corr,spin)
                baryon = contractions.isospin_three_half_spin_contract(ps_DP_down,ps_DP_down,ps_DP_down,corr,spin)
                baryon_time = np.einsum('tzyx->t',baryon)

                fh_DD_A3 = np.complex128(0)
                fh_DD_V4 = np.complex128(0)

                fh_DD_A3+= contractions.isospin_three_half_spin_contract(ps_DP_down,ps_DP_down,ps_fh_DP_down['A3'], corr, spin)
                fh_DD_A3+= contractions.isospin_three_half_spin_contract(ps_DP_down,ps_fh_DP_down['A3'],ps_DP_down, corr, spin)
                fh_DD_A3+= contractions.isospin_three_half_spin_contract(ps_fh_DP_down['A3'],ps_DP_down,ps_DP_down, corr, spin)
                fh_DD_V4+= contractions.isospin_three_half_spin_contract(ps_DP_down,ps_DP_down,ps_fh_DP_down['V4'], corr, spin)
                fh_DD_V4+= contractions.isospin_three_half_spin_contract(ps_DP_down,ps_fh_DP_down['V4'],ps_DP_down, corr, spin)
                fh_DD_V4+= contractions.isospin_three_half_spin_contract(ps_fh_DP_down['V4'],ps_DP_down,ps_DP_down, corr, spin)


                f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
                known_fh = {}
                known_fh['A3_DD'] = f_fh['PS/fh_'+corr+'_A3_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
                known_fh['V4_DD'] = f_fh['PS/fh_'+corr+'_V4_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

                fh_baryon_time = {} 
                fh_baryon_time['A3_DD'] = np.einsum('tzyx->t',fh_DD_A3)
                fh_baryon_time['V4_DD'] = np.einsum('tzyx->t',fh_DD_V4)


                print(fh_baryon_time,"fhtime")
                for curr_contract in ['A3_DD','V4_DD']:
                        if np.any(abs(np.real(fh_baryon_time[curr_contract] - known_fh[curr_contract])/np.real(fh_baryon_time[curr_contract] + known_fh[curr_contract])) > 1.e-7):
                            print(curr_contract,'    FAIL')
                        else:
                            print(curr_contract,'    PASS')
                '''
                for t in range(Nt):
                    print(t,baryon_up_time[t])
                '''
                # f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
                # known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
                # if known_baryon is None:
                #     raise ValueError("this baryon does not have a dataset in the 2pt spectrum file")
                # f.close()
                # if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
                #     print('    FAIL')
                # else:
                #     print('    PASS')

        print('\nPS FH')
        print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
        for corr in ['delta_p','delta_p_np']:  #duu, duu
            for spin in ['up','dn']:
                print(corr,spin)
                # baryon = contractions.isospin_three_half_spin_contract(ps_DP_up,ps_DP_up,ps_DP_down,corr,spin)
                # baryon_time = np.einsum('tzyx->t',baryon)
                fh_DD_A3 = np.complex128(0)
                fh_DD_V4 = np.complex128(0)
                fh_UU_A3 = np.complex128(0)
                fh_UU_V4 = np.complex128(0)

                fh_DD_A3+= contractions.isospin_three_half_spin_contract(ps_fh_DP_down['A3'],ps_DP_up,ps_DP_up, corr, spin)
                fh_DD_V4+= contractions.isospin_three_half_spin_contract(ps_fh_DP_down['V4'],ps_DP_up,ps_DP_up, corr, spin)

                fh_UU_A3+= contractions.isospin_three_half_spin_contract(ps_DP_down,ps_fh_DP_up['A3'],ps_DP_up, corr, spin)
                fh_UU_A3+= contractions.isospin_three_half_spin_contract(ps_DP_down,ps_DP_up,ps_fh_DP_up['A3'], corr, spin)

                fh_UU_V4+= contractions.isospin_three_half_spin_contract(ps_DP_down,ps_fh_DP_up['V4'],ps_DP_up, corr, spin)
                fh_UU_V4+= contractions.isospin_three_half_spin_contract(ps_DP_down,ps_DP_up,ps_fh_DP_up['V4'], corr, spin)
                f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
                known_fh = {}
                known_fh['A3_DD'] = f_fh['PS/fh_'+corr+'_A3_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
                known_fh['V4_DD'] = f_fh['PS/fh_'+corr+'_V4_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
                known_fh['V4_UU'] = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
                known_fh['A3_UU'] = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]


                fh_baryon_time = {} 
                fh_baryon_time['A3_DD'] = np.einsum('tzyx->t',fh_DD_A3)
                fh_baryon_time['V4_DD'] = np.einsum('tzyx->t',fh_DD_V4)
                fh_baryon_time['V4_UU'] = np.einsum('tzyx->t',fh_UU_V4)
                fh_baryon_time['A3_UU'] = np.einsum('tzyx->t',fh_UU_A3)


                print(fh_baryon_time,"fhtime")
                for curr_contract in ['V4_UU','A3_UU','A3_DD','V4_DD']:
                        if np.any(abs(np.real(fh_baryon_time[curr_contract] - known_fh[curr_contract])/np.real(fh_baryon_time[curr_contract] + known_fh[curr_contract])) > 1.e-7):
                            print(curr_contract,'    FAIL')
                        else:
                            print(curr_contract,'    PASS')

                '''
                for t in range(Nt):
                    print(t,baryon_up_time[t])
                '''
                # f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
                # known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
                # f.close()
                # if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
                #     print('    FAIL')
                # else:
                #     print('    PASS')


            
            
    if False:
        print('\nPS')
        print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
        for corr in ['delta_z','delta_z_np']: #ddu
            for spin in ['up','dn','upup','dndn']:
                print(corr,spin)
                print('computing fh coors for %s'%corr)
                # baryon = contractions.isospin_three_half_spin_contract(ps_DP_up,ps_DP_up,ps_DP_up,corr,spin)
                # baryon_time = np.einsum('tzyx->t',baryon)

                fh_DD_A3 = np.complex128(0)
                fh_DD_V4 = np.complex128(0)
                fh_UU_A3 = np.complex128(0)
                fh_UU_V4 = np.complex128(0)

                fh_DD_A3+= contractions.isospin_three_half_spin_contract(ps_fh_DP_down['A3'],ps_DP_down,ps_DP_up, corr, spin)
                fh_DD_A3+= contractions.isospin_three_half_spin_contract(ps_DP_down,ps_fh_DP_down['A3'],ps_DP_up, corr, spin)

                fh_DD_V4+= contractions.isospin_three_half_spin_contract(ps_fh_DP_down['V4'],ps_DP_down,ps_DP_up, corr, spin)
                fh_DD_V4+= contractions.isospin_three_half_spin_contract(ps_DP_down,ps_fh_DP_down['V4'],ps_DP_up, corr, spin)

                fh_UU_A3+= contractions.isospin_three_half_spin_contract(ps_DP_down,ps_DP_down, ps_fh_DP_up['A3'], corr, spin)
                fh_UU_V4+= contractions.isospin_three_half_spin_contract(ps_DP_down,ps_DP_down, ps_fh_DP_up['V4'], corr, spin)

                
                f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
                known_fh = {}
                known_fh['A3_DD'] = f_fh['PS/fh_'+corr+'_A3_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
                known_fh['V4_DD'] = f_fh['PS/fh_'+corr+'_V4_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
                known_fh['V4_UU'] = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
                known_fh['A3_UU'] = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]


                fh_baryon_time = {} 
                fh_baryon_time['A3_DD'] = np.einsum('tzyx->t',fh_DD_A3)
                fh_baryon_time['V4_DD'] = np.einsum('tzyx->t',fh_DD_V4)
                fh_baryon_time['V4_UU'] = np.einsum('tzyx->t',fh_UU_V4)
                fh_baryon_time['A3_UU'] = np.einsum('tzyx->t',fh_UU_A3)


                print(fh_baryon_time,"fhtime")
                for curr_contract in ['V4_UU','A3_UU','A3_DD','V4_DD']:
                        if np.any(abs(np.real(fh_baryon_time[curr_contract] - known_fh[curr_contract])/np.real(fh_baryon_time[curr_contract] + known_fh[curr_contract])) > 1.e-7):
                            print(curr_contract,'    FAIL')
                        else:
                            print(curr_contract,'    PASS')

                # IF 2PT CORR IS MADE:
        #         # for t in range(Nt):
        #         #     # print(t,baryon_time[t])
                
        #         f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
        #         known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        #         # print('\n',known_baryon)

        #         for t in range(Nt):
        #             # print(known_baryon[t],"known")
        #             print(t,baryon_time[t]/known_baryon[t])
                
        #         f.close()
        #         if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
        #             print('    FAIL')
        #         else:
        #             print('    PASS')


    # print('\nPS')
    # print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
    # for corr in ['lambda_z','lambda_z_np']: #dsu
    #     for spin in ['up','dn']:
    #         print(corr,spin)
    #         baryon = contractions.isospin_zero_spin_contract(ps_DP_up,ps_DP_down,ps_DP_strange,corr,spin)
    #         baryon_time = np.einsum('tzyx->t',baryon)

    #         fh_UU_A3 = np.complex128(0)
    #         fh_DD_A3 = np.complex128(0)
    #         fh_UU_V4 = np.complex128(0)
    #         fh_DD_V4 = np.complex128(0)

    #         fh_UU_A3+= contractions.isospin_zero_spin_contract(ps_fh_DP_up['A3'],ps_DP_down,ps_DP_strange, corr, spin) 
    #         fh_DD_A3+= contractions.isospin_zero_spin_contract(ps_DP_up,ps_fh_DP_down['A3'],ps_DP_strange, corr, spin)

    #         fh_UU_V4 += contractions.isospin_zero_spin_contract(ps_fh_DP_up['V4'],ps_DP_down,ps_DP_strange, corr, spin) 
    #         fh_DD_V4 +=  contractions.isospin_zero_spin_contract(ps_DP_up,ps_fh_DP_down['V4'],ps_DP_strange, corr, spin)

    #         f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
    #         known_fh = {}
    #         known_fh['A3_UU'] = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['V4_UU'] = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['A3_DD'] = f_fh['PS/fh_'+corr+'_A3_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['V4_DD'] = f_fh['PS/fh_'+corr+'_V4_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         fh_baryon_time = {} 
    #         fh_baryon_time['A3_UU'] = np.einsum('tzyx->t',fh_UU_A3)
    #         fh_baryon_time['V4_UU'] = np.einsum('tzyx->t',fh_UU_V4)

    #         fh_baryon_time['A3_DD'] = np.einsum('tzyx->t',fh_DD_A3)
    #         fh_baryon_time['V4_DD'] = np.einsum('tzyx->t',fh_DD_V4)


    #         # print(fh_baryon_time,"fhtime")
    #         for curr_contract in ['A3_UU','V4_UU','A3_DD','V4_DD']:
    #                 if np.any(abs(np.real(fh_baryon_time[curr_contract] - known_fh[curr_contract])/np.real(fh_baryon_time[curr_contract] + known_fh[curr_contract])) > 1.e-7):
    #                     print(curr_contract,'    FAIL')
    #                 else:
    #                     print(curr_contract,'    PASS')
    #         # print(known_fh,"fh")
            
            
    #         '''
    #         for t in range(Nt):
    #             print(t,baryon_up_time[t])
    #         '''
    #         f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
    #         known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         f.close()
    #         if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
    #             print(corr,spin,'    FAIL')
    #         else:
    #             print(corr,spin,'    PASS')

    # print('\nPS FH')
    # print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
    # for corr in ['neutron','neutron_np']: #ddu
    #     for spin in ['up','dn']:
    #         print(corr,spin)
    #         # baryon = contractions.isospin_half_spin_contract(ps_DP_up,ps_DP_up,ps_DP_down,corr,spin)
    #         # baryon_time = np.einsum('tzyx->t',baryon)
    #         fh_DD_A3 = np.complex128(0)
    #         fh_DD_V4 = np.complex128(0)
    #         fh_UU_A3 = np.complex128(0)
    #         fh_UU_V4 = np.complex128(0)

    #         fh_DD_A3+= contractions.isospin_half_spin_contract(ps_fh_DP_down['A3'],ps_DP_down,ps_DP_up, corr, spin)
    #         fh_DD_A3+= contractions.isospin_half_spin_contract(ps_DP_down,ps_fh_DP_down['A3'],ps_DP_up, corr, spin)

    #         fh_DD_V4+= contractions.isospin_half_spin_contract(ps_fh_DP_down['V4'],ps_DP_down,ps_DP_up, corr, spin)
    #         fh_DD_V4+= contractions.isospin_half_spin_contract(ps_DP_down,ps_fh_DP_down['V4'],ps_DP_up, corr, spin)

    #         fh_UU_A3+= contractions.isospin_half_spin_contract(ps_DP_down,ps_DP_down, ps_fh_DP_up['A3'], corr, spin)
    #         fh_UU_V4+= contractions.isospin_half_spin_contract(ps_DP_down,ps_DP_down, ps_fh_DP_up['V4'], corr, spin)

            
    #         f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
    #         known_fh = {}
    #         known_fh['A3_DD'] = f_fh['PS/fh_'+corr+'_A3_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['V4_DD'] = f_fh['PS/fh_'+corr+'_V4_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['V4_UU'] = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['A3_UU'] = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]


    #         fh_baryon_time = {} 
    #         fh_baryon_time['A3_DD'] = np.einsum('tzyx->t',fh_DD_A3)
    #         fh_baryon_time['V4_DD'] = np.einsum('tzyx->t',fh_DD_V4)
    #         fh_baryon_time['V4_UU'] = np.einsum('tzyx->t',fh_UU_V4)
    #         fh_baryon_time['A3_UU'] = np.einsum('tzyx->t',fh_UU_A3)


    #         print(fh_baryon_time,"fhtime")
    #         for curr_contract in ['V4_UU','A3_UU','A3_DD','V4_DD']:
    #                 if np.any(abs(np.real(fh_baryon_time[curr_contract] - known_fh[curr_contract])/np.real(fh_baryon_time[curr_contract] + known_fh[curr_contract])) > 1.e-7):
    #                     print(curr_contract,'    FAIL')
    #                 else:
    #                     print(curr_contract,'    PASS')
    #         # print(fh_UU)

    #         # '''
    #         # for t in range(Nt):
    #         #     print(t,baryon_up_time[t])
    #         # '''
    #         # f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
    #         # known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         # f.close()
    #         # if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
    #         #     print('    FAIL')
    #         # else:
    #         #     print('    PASS')

    # print('\nPS')
    # print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
    # for corr in ['proton','proton_np']:
    #     for spin in ['up','dn']:
    #         print(corr,spin)
    #         baryon = contractions.isospin_half_spin_contract(ps_DP_up,ps_DP_up,ps_DP_down,corr,spin)
    #         baryon_time = np.einsum('tzyx->t',baryon)

    #         '''
    #         for t in range(Nt):
    #             print(t,baryon_up_time[t])
    #         '''
    #         f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
    #         known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         f_fh = h5.File(known_path+'/lalibe_fh_proton.h5') 
    #         # known_fh_UU = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         # known_fh_DD = f_fh['PS/fh_'+corr+'_A3_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         # known_fh_UU = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         # known_fh_DD = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         fh_UU_A3 = np.complex128(0)
    #         fh_DD_A3 = np.complex128(0)
    #         fh_UU_V4 = np.complex128(0)
    #         fh_DD_V4 = np.complex128(0)

    #         fh_UU_A3+= contractions.isospin_half_spin_contract(ps_fh_DP_up['A3'],ps_DP_up,ps_DP_down, corr, spin) 
    #         fh_UU_A3+= contractions.isospin_half_spin_contract(ps_DP_up,ps_fh_DP_up['A3'],ps_DP_down, corr, spin)
    #         fh_DD_A3+= contractions.isospin_half_spin_contract(ps_DP_up,ps_DP_up,ps_fh_DP_down['A3'], corr, spin)

    #         fh_UU_V4+= contractions.isospin_half_spin_contract(ps_fh_DP_up['V4'],ps_DP_up,ps_DP_down, corr, spin) 
    #         fh_UU_V4+= contractions.isospin_half_spin_contract(ps_DP_up,ps_fh_DP_up['V4'],ps_DP_down, corr, spin)
    #         fh_DD_V4+= contractions.isospin_half_spin_contract(ps_DP_up,ps_DP_up,ps_fh_DP_down['V4'], corr, spin)

    #         f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
    #         known_fh = {}
    #         known_fh['A3_UU'] = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['V4_UU'] = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['A3_DD'] = f_fh['PS/fh_'+corr+'_A3_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['V4_DD'] = f_fh['PS/fh_'+corr+'_V4_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         fh_baryon_time = {} 
    #         fh_baryon_time['A3_UU'] = np.einsum('tzyx->t',fh_UU_A3)
    #         fh_baryon_time['V4_UU'] = np.einsum('tzyx->t',fh_UU_V4)

    #         fh_baryon_time['A3_DD'] = np.einsum('tzyx->t',fh_DD_A3)
    #         fh_baryon_time['V4_DD'] = np.einsum('tzyx->t',fh_DD_V4)


    #         # print(fh_baryon_time,"fhtime")
    #         for curr_contract in ['A3_UU','V4_UU','A3_DD','V4_DD']:
    #                 if np.any(abs(np.real(fh_baryon_time[curr_contract] - known_fh[curr_contract])/np.real(fh_baryon_time[curr_contract] + known_fh[curr_contract])) > 1.e-7):
    #                     print('FH',curr_contract,'    FAIL')
    #                 else:
    #                     print('FH',curr_contract,'    PASS')
    #         # print(known_fh,"fh")

    #         f.close()
    #         if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
    #             print('2PT','          FAIL')
    #         else:
    #             print('2PT', '         PASS')

    # print('\nPS')
    # print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
    # for corr in ['sigma_m','sigma_m_np']: #dds
    #     for spin in ['up','dn']:
    #         print(corr,spin)
    #         # baryon = contractions.isospin_one_spin_contract(ps_DP_up,ps_DP_up,ps_DP_strange,corr,spin)
    #         # baryon_time = np.einsum('tzyx->t',baryon)

    #         fh_DD_A3 = np.complex128(0)
    #         fh_DD_V4 = np.complex128(0)

    #         fh_DD_A3+= contractions.isospin_one_spin_contract(ps_fh_DP_down['A3'],ps_DP_down,ps_DP_strange, corr, spin)
    #         fh_DD_A3+= contractions.isospin_one_spin_contract(ps_DP_down,ps_fh_DP_down['A3'],ps_DP_strange, corr, spin)

    #         fh_DD_V4+= contractions.isospin_one_spin_contract(ps_fh_DP_down['V4'],ps_DP_down,ps_DP_strange, corr, spin)
    #         fh_DD_V4+= contractions.isospin_one_spin_contract(ps_DP_down,ps_fh_DP_down['V4'],ps_DP_strange, corr, spin)
            
    #         f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
    #         known_fh = {}
    #         known_fh['A3_DD'] = f_fh['PS/fh_'+corr+'_A3_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['V4_DD'] = f_fh['PS/fh_'+corr+'_V4_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
        
    #         fh_baryon_time = {} 
    #         fh_baryon_time['A3_DD'] = np.einsum('tzyx->t',fh_DD_A3)
    #         fh_baryon_time['V4_DD'] = np.einsum('tzyx->t',fh_DD_V4)
        
    #         print(fh_baryon_time,"fhtime")
    #         for curr_contract in ['A3_DD','V4_DD']:
    #                 if np.any(abs(np.real(fh_baryon_time[curr_contract] - known_fh[curr_contract])/np.real(fh_baryon_time[curr_contract] + known_fh[curr_contract])) > 1.e-7):
    #                     print(curr_contract,'      FAIL')
    #                 else:
    #                     print('FH',curr_contract,'    PASS')
    #         '''
    #         IF 2PT CORR ADDED:
    #         for t in range(Nt):
    #             print(t,baryon_up_time[t])
    #         '''
    #         # f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
    #         # known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    # #         f.close()
    # #         if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
    # #             print('    FAIL')
    # #         else:
    # #             print('    PASS')

    # print('\nPS')
    # print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
    # for corr in ['sigma_p','sigma_p_np']: #suu
    #     for spin in ['up','dn']:
    #         print(corr,spin)
    #         baryon = contractions.isospin_one_spin_contract(ps_DP_up,ps_DP_up,ps_DP_strange,corr,spin)
    #         baryon_time = np.einsum('tzyx->t',baryon)

    #         fh_UU_A3 = np.complex128(0)
    #         fh_UU_V4 = np.complex128(0)

    #         fh_UU_A3+= contractions.isospin_one_spin_contract(ps_fh_DP_up['A3'],ps_DP_up,ps_DP_strange, corr, spin) 
    #         fh_UU_A3+= contractions.isospin_one_spin_contract(ps_DP_up,ps_fh_DP_up['A3'],ps_DP_strange, corr, spin)
    #         fh_UU_V4+= contractions.isospin_one_spin_contract(ps_fh_DP_up['V4'],ps_DP_up,ps_DP_strange, corr, spin) 
    #         fh_UU_V4+= contractions.isospin_one_spin_contract(ps_DP_up,ps_fh_DP_up['V4'],ps_DP_strange, corr, spin)
            
    #         '''
    #         for t in range(Nt):
    #             print(t,baryon_up_time[t])
    #         '''
    #         f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
    #         known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
    #         known_fh = {}
    #         known_fh['A3_UU'] = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['V4_UU'] = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         fh_baryon_time = {} 
    #         fh_baryon_time['A3_UU'] = np.einsum('tzyx->t',fh_UU_A3)
    #         fh_baryon_time['V4_UU'] = np.einsum('tzyx->t',fh_UU_V4)

    #         # print(fh_baryon_time,"fhtime")
    #         for curr_contract in ['A3_UU','V4_UU']:
    #                 if np.any(abs(np.real(fh_baryon_time[curr_contract] - known_fh[curr_contract])/np.real(fh_baryon_time[curr_contract] + known_fh[curr_contract])) > 1.e-7):
    #                     print('FH',curr_contract,'    FAIL')
    #                 else:
    #                     print('FH',curr_contract,'    PASS')
    #         # print(known_fh,"fh")
        

    #         f.close()
    #         if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
    #             print('2PT','          FAIL')
    #         else:
    #             print('2PT','          PASS')

    # print('\nPS')
    # print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
    # for corr in ['sigma_star_p','sigma_star_p_np']: #suu
    #     for spin in ['up','dn']:
    #         print(corr,spin)
    #         baryon = contractions.isospin_one_spin_contract(ps_DP_up,ps_DP_up,ps_DP_strange,corr,spin)
    #         baryon_time = np.einsum('tzyx->t',baryon)

    #         fh_UU_A3 = np.complex128(0)
    #         fh_UU_V4 = np.complex128(0)

    #         fh_UU_A3+= contractions.isospin_one_spin_contract(ps_fh_DP_up['A3'],ps_DP_up,ps_DP_strange, corr, spin) 
    #         fh_UU_A3+= contractions.isospin_one_spin_contract(ps_DP_up,ps_fh_DP_up['A3'],ps_DP_strange, corr, spin)
    #         fh_UU_V4+= contractions.isospin_one_spin_contract(ps_fh_DP_up['V4'],ps_DP_up,ps_DP_strange, corr, spin) 
    #         fh_UU_V4+= contractions.isospin_one_spin_contract(ps_DP_up,ps_fh_DP_up['V4'],ps_DP_strange, corr, spin)
    #         '''
    #         for t in range(Nt):
    #             print(t,baryon_up_time[t])
    #         '''
    #         f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
    #         known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
    #         known_fh = {}
    #         known_fh['A3_UU'] = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['V4_UU'] = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         fh_baryon_time = {} 
    #         fh_baryon_time['A3_UU'] = np.einsum('tzyx->t',fh_UU_A3)
    #         fh_baryon_time['V4_UU'] = np.einsum('tzyx->t',fh_UU_V4)

    #         # print(fh_baryon_time,"fhtime")
    #         for curr_contract in ['A3_UU','V4_UU']:
    #                 if np.any(abs(np.real(fh_baryon_time[curr_contract] - known_fh[curr_contract])/np.real(fh_baryon_time[curr_contract] + known_fh[curr_contract])) > 1.e-7):
    #                     print('FH',curr_contract,'    FAIL')
    #                 else:
    #                     print('FH',curr_contract,'    PASS')
    #         # print(known_fh,"fh")
    #         f.close()
    #         if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
    #             print('2PT','          FAIL')
    #         else:
    #             print('2PT','          PASS')


    # print('\nPS FH')
    # print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
    # for corr in ['sigma_star_z','sigma_star_z_np']: #dsu 
    #     for spin in ['up','dn']:
    #         print(corr,spin)
    #         # baryon = contractions.isospin_one_spin_contract(ps_DP_up,ps_DP_up,ps_DP_strange,corr,spin)
    #         # baryon_time = np.einsum('tzyx->t',baryon)

    #         fh_UU_A3 = np.complex128(0)
    #         fh_UU_V4 = np.complex128(0)
    #         fh_DD_A3 = np.complex128(0)
    #         fh_DD_V4 = np.complex128(0)

    #         fh_DD_A3+= contractions.isospin_one_spin_contract(ps_fh_DP_down['A3'],ps_DP_strange,ps_DP_up, corr, spin) 
    #         fh_DD_V4+=contractions.isospin_one_spin_contract(ps_fh_DP_down['V4'],ps_DP_strange,ps_DP_up, corr, spin) 
    #         fh_UU_A3+= contractions.isospin_one_spin_contract(ps_DP_down,ps_DP_strange,ps_fh_DP_up['A3'], corr, spin) 
    #         fh_UU_V4+= contractions.isospin_one_spin_contract(ps_DP_down,ps_DP_strange,ps_fh_DP_up['V4'], corr, spin)
    #         '''
    #         for t in range(Nt):
    #             print(t,baryon_up_time[t])
    #         '''
    #         # f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
    #         # known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
    #         known_fh = {} 
            
    #         known_fh['A3_UU'] = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['A3_DD'] = f_fh['PS/fh_'+corr+'_A3_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['V4_DD'] = f_fh['PS/fh_'+corr+'_V4_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['V4_UU'] = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         fh_baryon_time = {} 
    #         fh_baryon_time['A3_UU'] = np.einsum('tzyx->t',fh_UU_A3)
    #         fh_baryon_time['V4_UU'] = np.einsum('tzyx->t',fh_UU_V4)
    #         fh_baryon_time['A3_DD'] = np.einsum('tzyx->t',fh_DD_A3)
    #         fh_baryon_time['V4_DD'] = np.einsum('tzyx->t',fh_DD_V4)
    #         f.close()

    #         for curr_contract in ['A3_UU','V4_UU','A3_DD','V4_DD']:
    #             if np.any(abs(np.real(fh_baryon_time[curr_contract] - known_fh[curr_contract])/np.real(fh_baryon_time[curr_contract] + known_fh[curr_contract])) > 1.e-7):
    #                 print('FH',curr_contract,'    FAIL')
    #             else:
    #                 print('FH',curr_contract,'    PASS')
    #         # if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
    #         #     print('    FAIL')
    #         # else:
    #         #     print('    PASS')


    # print('\nPS')
    # print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
    # for corr in ['sigma_to_lambda','sigma_to_lambda_np']:#
    #     for spin in ['up','dn']:
    #         print(corr,spin)
    #         # baryon = contractions.isospin_one_spin_contract(ps_DP_up,ps_DP_up,ps_DP_strange,corr,spin)
    #         # baryon_time = np.einsum('tzyx->t',baryon)

    #         fh_UU_A3 = np.complex128(0)
    #         fh_UU_V4 = np.complex128(0)

    #         fh_UU_A3+= contractions.sigma_lambda_spin_contract(ps_fh_DP_up['A3'],ps_DP_up,ps_DP_strange, corr, spin) 
    #         fh_UU_V4+= contractions.sigma_lambda_spin_contract(ps_DP_up,ps_fh_DP_up['V4'],ps_DP_strange, corr, spin)

    #         fh_baryon_time = {} 
    #         fh_baryon_time['A3_UU'] = np.einsum('tzyx->t',fh_UU_A3)
    #         fh_baryon_time['V4_UU'] = np.einsum('tzyx->t',fh_UU_V4)
            
    #         '''
    #         for t in range(Nt):
    #             print(t,baryon_up_time[t])
    #         '''
    #         # f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
    #         # known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
    #         known_fh = {} 
    #         known_fh['A3_UU'] = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         # known_fh_DD = f_fh['PS/fh_'+corr+'_A3_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         known_fh['V4_UU'] = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         for curr_contract in ['A3_UU','V4_UU']:
    #             if np.any(abs(np.real(fh_baryon_time[curr_contract] - known_fh[curr_contract])/np.real(fh_baryon_time[curr_contract] + known_fh[curr_contract])) > 1.e-7):
    #                 print('FH',curr_contract,'    FAIL')
    #             else:
    #                 print('FH',curr_contract,'    PASS')

    #         f.close()
    #         # if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
    #         #     print('    FAIL')
    #         # else:
    #         #     print('    PASS')

    # print('\nPS')
    # print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
    # for corr in ['sigma_z','sigma_z_np']: #dsu
    #     for spin in ['up','dn']:
    #         print(corr,spin)
    #         # baryon = contractions.isospin_one_spin_contract(ps_DP_up,ps_DP_up,ps_DP_strange,corr,spin)
    #         # baryon_time = np.einsum('tzyx->t',baryon)

    #         fh_UU_A3 = np.complex128(0)
    #         fh_UU_V4 = np.complex128(0)
    #         fh_DD_A3 = np.complex128(0)
    #         fh_DD_V4 = np.complex128(0)

    #         fh_DD_A3+= contractions.isospin_one_spin_contract(ps_fh_DP_down['A3'],ps_DP_strange,ps_DP_up, corr, spin) 
    #         fh_DD_V4+=contractions.isospin_one_spin_contract(ps_fh_DP_down['V4'],ps_DP_strange,ps_DP_up, corr, spin) 
    #         fh_UU_A3+= contractions.isospin_one_spin_contract(ps_DP_down,ps_DP_strange,ps_fh_DP_up['A3'], corr, spin) 
    #         fh_UU_V4+= contractions.isospin_one_spin_contract(ps_DP_down,ps_DP_strange,ps_fh_DP_up['V4'], corr, spin)

    #         fh_baryon_time = {} 
    #         fh_baryon_time['A3_UU'] = np.einsum('tzyx->t',fh_UU_A3)
    #         fh_baryon_time['V4_UU'] = np.einsum('tzyx->t',fh_UU_V4)
    #         fh_baryon_time['A3_DD'] = np.einsum('tzyx->t',fh_DD_A3)
    #         fh_baryon_time['V4_DD'] = np.einsum('tzyx->t',fh_DD_V4)
    #         '''
    #         for t in range(Nt):
    #             print(t,baryon_up_time[t])
    #         '''
    #         # f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
    #         # known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
    #         known_fH = {} 
    #         known_fh['A3_UU'] = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['A3_DD'] = f_fh['PS/fh_'+corr+'_A3_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['V4_DD'] = f_fh['PS/fh_'+corr+'_V4_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         known_fh['V4_UU'] = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         for curr_contract in ['A3_UU','V4_UU','A3_DD','V4_DD']:
    #             if np.any(abs(np.real(fh_baryon_time[curr_contract] - known_fh[curr_contract])/np.real(fh_baryon_time[curr_contract] + known_fh[curr_contract])) > 1.e-7):
    #                 print('FH',curr_contract,'    FAIL')
    #             else:
    #                 print('FH',curr_contract,'    PASS')
    #         # f.close()


    #         # if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
    #         #     print('    FAIL')
    #         # else:
    #         #     print('    PASS')


    # print('\nPS')
    # print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
    # for corr in ['omega_m','omega_m_np']: #sss
    #     for spin in ['up','dn']:
    #         print(corr,spin)
    #         baryon = contractions.isospin_zero_spin_contract(ps_DP_strange,ps_DP_strange,ps_DP_strange,corr,spin)
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

    # print('\nPS')
    # print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
    # for corr in ['xi_z','xi_z_np']: #ssu
    #     for spin in ['up','dn']:
    #         print(corr,spin)
    #         baryon = contractions.isospin_half_spin_contract(ps_DP_strange,ps_DP_strange,ps_DP_up,corr,spin)
    #         baryon_time = np.einsum('tzyx->t',baryon)

    #         fh_UU_A3 = np.complex128(0)
    #         fh_UU_V4 = np.complex128(0)
    #         fh_UU_A3+= contractions.isospin_half_spin_contract(ps_fh_DP_up['A3'],ps_DP_strange,ps_DP_strange, corr, spin) 
    #         fh_UU_V4+= contractions.isospin_half_spin_contract(ps_fh_DP_up['V4'],ps_DP_strange,ps_DP_strange, corr, spin) 

    #         fh_baryon_time = {} 
    #         fh_baryon_time['A3_UU'] = np.einsum('tzyx->t',fh_UU_A3)
    #         fh_baryon_time['V4_UU'] = np.einsum('tzyx->t',fh_UU_V4)
            
    #         '''
    #         for t in range(Nt):
    #             print(t,baryon_up_time[t])
    #         '''
    #         f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
    #         known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
    #         known_fh = {} 
    #         known_fh['A3_UU'] = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         known_fh['V4_UU'] = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         for curr_contract in ['A3_UU','V4_UU']:
    #             if np.any(abs(np.real(fh_baryon_time[curr_contract] - known_fh[curr_contract])/np.real(fh_baryon_time[curr_contract] + known_fh[curr_contract])) > 1.e-7):
    #                 print('FH',curr_contract,'    FAIL')
    #             else:
    #                 print('FH',curr_contract,'    PASS')
            
    #         f.close()
    #         if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
    #             print('    FAIL')
    #         else:
    #             print('    PASS')

    # print('\nPS')
    # print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
    # for corr in ['xi_star_m','xi_star_m_np']: #ssd
    #     for spin in ['up','dn','upup','dndn']:
    #         print(corr,spin)
    #         # baryon = contractions.isospin_half_spin_contract(ps_DP_strange,ps_DP_strange,ps_DP_down,corr,spin)

    #         # baryon_time = np.einsum('tzyx->t',baryon)

    #         fh_DD_A3 = np.complex128(0)
    #         fh_DD_V4 = np.complex128(0)
    #         fh_DD_A3+= contractions.isospin_half_spin_contract(ps_fh_DP_down['A3'],ps_DP_strange,ps_DP_strange, corr, spin) 
    #         fh_DD_V4+= contractions.isospin_half_spin_contract(ps_fh_DP_down['V4'],ps_DP_strange,ps_DP_strange, corr, spin) 

    #         fh_baryon_time = {} 
    #         fh_baryon_time['A3_DD'] = np.einsum('tzyx->t',fh_DD_A3)
    #         fh_baryon_time['V4_DD'] = np.einsum('tzyx->t',fh_DD_V4)
    #         '''
    #         for t in range(Nt):
    #             print(t,baryon_up_time[t])
    #         '''
    #         # f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
    #         # known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
    #         known_fh = {} 
    #         known_fh['A3_DD'] = f_fh['PS/fh_'+corr+'_A3_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         known_fh['V4_DD'] = f_fh['PS/fh_'+corr+'_V4_DD/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         for curr_contract in ['A3_DD','V4_DD']:
    #             if np.any(abs(np.real(fh_baryon_time[curr_contract] - known_fh[curr_contract])/np.real(fh_baryon_time[curr_contract] + known_fh[curr_contract])) > 1.e-7):
    #                 print('FH',curr_contract,'    FAIL')
    #             else:
    #                 print('FH',curr_contract,'    PASS')
    #         '''
    #         for t in range(Nt):
    #             print(t,baryon_up_time[t])
    #         '''
    #         # f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
    #         # known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         # f.close()
    #         # if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
    #         #     print('    FAIL')
    #         # else:
    #         #     print('    PASS')

    # print('\nPS')
    # print('(baryon_up - known) / (baryon_up + known) > 1.e-7?')
    # for corr in ['xi_star_z','xi_star_z_np']: #ssu
    #     for spin in ['up','dn','upup','dndn']:
    #         print(corr,spin)
    #         baryon = contractions.isospin_half_spin_contract(ps_DP_strange,ps_DP_strange,ps_DP_up,corr,spin)

    #         baryon_time = np.einsum('tzyx->t',baryon)

    #         fh_UU_A3 = np.complex128(0)
    #         fh_UU_V4 = np.complex128(0)
    #         fh_UU_A3+= contractions.isospin_half_spin_contract(ps_fh_DP_up['A3'],ps_DP_strange,ps_DP_strange, corr, spin) 
    #         fh_UU_V4+= contractions.isospin_half_spin_contract(ps_fh_DP_up['V4'],ps_DP_strange,ps_DP_strange, corr, spin) 

    #         fh_baryon_time = {} 
    #         fh_baryon_time['A3_UU'] = np.einsum('tzyx->t',fh_UU_A3)
    #         fh_baryon_time['V4_UU'] = np.einsum('tzyx->t',fh_UU_V4)
    #         '''
    #         for t in range(Nt):
    #             print(t,baryon_up_time[t])
    #         '''
    #         f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
    #         known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         f_fh = h5.File(known_path+'/lalibe_fh_baryons.h5')
    #         known_fh = {} 
    #         known_fh['A3_UU'] = f_fh['PS/fh_'+corr+'_A3_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         known_fh['V4_UU'] = f_fh['PS/fh_'+corr+'_V4_UU/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]

    #         for curr_contract in ['A3_UU','V4_UU']:
    #             if np.any(abs(np.real(fh_baryon_time[curr_contract] - known_fh[curr_contract])/np.real(fh_baryon_time[curr_contract] + known_fh[curr_contract])) > 1.e-7):
    #                 print('FH',curr_contract,'    FAIL')
    #             else:
    #                 print('FH',curr_contract,'    PASS')
            
    #         f = h5.File(known_path+'/lalibe_2pt_spectrum.h5')
    #         known_baryon = f['PS/'+corr+'/spin_'+spin+'/x0_y0_z0_t0/px0_py0_pz0'][()]
    #         f.close()
    #         if np.any(abs(np.real(baryon_time - known_baryon)/np.real(baryon_time + known_baryon)) > 1.e-7):
    #             print('    FAIL')
    #         else:
    #             print('    PASS')


    # if __name__ == '__main__':
    #     main()  


