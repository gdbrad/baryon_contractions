import numpy as np
import h5py as h5
import baryon_contractions_gb as contractions
import gamma
import sys
import warnings
warnings.filterwarnings("ignore")

'''
This routine computes a matrix elements of external currents
utilizing only two-point correlation functions. 
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
def spin_projection(ir):
    '''
    Define the quasi-local nucleon operators as in: 
      "Clebsch-Gordan construction of lattice interpolating fields for excited baryons" hep-lat/0508018
        Input:
        zero-momentum  Baryon interpolating field operator 
        there are 23 G_1g operators, 7 G_2g operators
        Notation follows that of Tablex XV,XVI in Basak.
        output:
        gauge invariant quark trilinear with appropiate flavor structure.
     '''
    IRs = {
    'G1g1u' : [1,2,1,1],
    'G1g1d' : [1,2,2,1],
    'G1u1u' : [3,4,3,1],
    'G1u1d' : [3,4,4,1],
      
    'G1g2u_1' : [1,4,3,1],
    'G1g2u_2' : [3,2,3,1],
    'G1g2u_3' : [3,4,1,1],
    'G1g2d_1' : [1,4,4,1],
    'G1g2d_2' : [3,2,4,1],
    'G1g2d_3' : [3,4,2,1],
    'G1u2u_1' : [1,2,3,1],
    'G1u2u_2' : [1,4,1,1],
    'G1u2u_3' : [3,2,1,1],
    'G1u2d_1' : [1,2,4,1],
    'G1u2d_2' : [1,4,2,1],
    'G1u2d_3' : [3,2,2,1],
        # basak 3
    'G1g3u_1' : [1,3,4,1],
    'G1g3u_2' : [3,2,3,1],
    'G1g3u_3' : [3,4,1,-1],
    'G1g3d_1' : [1,4,4,1],
    'G1g3d_2' : [4,2,3,1],
    'G1g3d_3' : [3,4,2,-1],
    'G1u3u_1' : [1,4,1,-1],
    'G1u3u_2' : [3,1,2,-1],
    'G1u3u_3' : [1,2,3,1],
    'G1u3d_1' : [3,2,2,-1],
    'G1u3d_2' : [2,4,1,-1],
    'G1u3d_3' : [1,2,4,-1],
    }

    U = gamma.U_DR_to_DP
    U_adj = gamma.U_DR_to_DP_adj





#ef bilinear_gamma(current_0, fh_prop_src, quark_propagator,u):

print(contractions.isospin_half_spin_contract(q1, q2, q3, corr, spin))

def create_fh_prop()

def flavor_conserving_fh_baryon_contraction(q1,q2,q3,corr,spin):
    '''
    3-pt contraction routine. Current is inserted on each line in the 2-point function 
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

    def levi():
        eijk = np.zeros((3, 3, 3))
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
        return eijk

    """ fh contract U """
    origin = [0,0,0,0]
    def fh_isospin_half_spin_contract_U(prop,fh_prop,P,P_2,G,G_2):
        FP = np.einsum('tzyxqlec,l->tzyxqec',fh_prop,P)
        FPP_2 = np.einsum('txyzqec,s->txyzqsec', FP, P_2)
        FPP_2G_2 = np.einsum('txyzqsec,qr->txyzrsec', FPP_2, G_2)
        UG = np.einsum('txyzsiga,ij->txyzsjga', prop, G)
        UGD = np.einsum('txyzsjga,txyzrjfb->txyzsrgafb', UG, prop)
        UGDFPP_2G_2 = np.einsum('txyzsrgafb,txyzrsec->txyzgafbec', UGD, FPP_2G_2)
        # color contract
        UGDFPP_2G_2 = np.einsum('abc,txyzgafbec->txyzgfe', levi(), UGDFPP_2G_2)
        UGDFPP_2G_2 = np.einsum('efg,txyzgfe->txyz', levi(), UGDFPP_2G_2)
        UGDFPP_2G_2 = 0.5*UGDFPP_2G_2
        # momentum projection
        term1 = UGDFPP_2G_2.sum(axis=3).sum(axis=2).sum(axis=1)
        # term 2
        # spin contract
        FG = np.einsum('txyzqiea,ij->txyzqjea', fhprop, G)
        FGG_2 = np.einsum('txyzqjea,qr->txyzrjea', FG, G_2)
        FGG_2D = np.einsum('txyzrjea,txyzrjfb->txyzeafb', FGG_2, prop)
        UP_2 = np.einsum('txyzslgc,s->txyzlgc', prop, P_2)
        UP_2P = np.einsum('txyzlgc,l->txyzgc', UP_2, P)
        FGG_2DUP_2P = np.einsum('txyzeafb,txyzgc->txyzeafbgc', FGG_2D, UP_2P)
        # color contract
        FGG_2DUP_2P = np.einsum('abc,txyzeafbgc->txyzefg', levi(), FGG_2DUP_2P)
        FGG_2DUP_2P = np.einsum('efg,txyzefg->txyz', levi(), FGG_2DUP_2P)
        FGG_2DUP_2P = -0.5*FGG_2DUP_2P
        # momentum projection
        term2 = FGG_2DUP_2P.sum(axis=3).sum(axis=2).sum(axis=1)
        # term 3
        # spin contract
        UG = np.einsum('txyzqiea,ij->txyzqjea', prop, G)
        UGG_2 = np.einsum('txyzqjea,qr->txyzrjea', UG, G_2)
        UGG_2D = np.einsum('txyzrjea,txyzrjfb->txyzeafb', UGG_2, prop)
        FP_2 = np.einsum('txyzslgc,s->txyzlgc', fhprop, P_2)
        FP_2P = np.einsum('txyzlgc,l->txyzgc', FP_2, P)
        UGG_2DFP_2P = np.einsum('txyzeafb,txyzgc->txyzeafbgc', UGG_2D, FP_2P)
        # color contract
        UGG_2DFP_2P = np.einsum('abc,txyzeafbgc->txyzefg', levi(), UGG_2DFP_2P)
        UGG_2DFP_2P = np.einsum('efg,txyzefg->txyz', levi(), UGG_2DFP_2P)
        UGG_2DFP_2P = -0.5*UGG_2DFP_2P
        # momentum projection
        term3 = UGG_2DFP_2P.sum(axis=3).sum(axis=2).sum(axis=1)
        # term 4
        # spin contract
        UG_2 = np.einsum('txyzqlec,qr->txyzrlec', prop, G_2)
        UG_2P = np.einsum('txyzrlec,l->txyzrec', UG_2, P)
        UG_2PP_2 = np.einsum('txyzrec,s->txyzrsec', UG_2P, P_2)
        DG = np.einsum('txyzrjfb,ij->txyzrifb', prop, G)
        DGF = np.einsum('txyzrifb,txyzsiga->txyzrsfbga', DG, fhprop)
        UG_2PP_2DGF = np.einsum('txyzrsec,txyzrsfbga->txyzecfbga', UG_2PP_2, DGF)
        # color contract
        UG_2PP_2DGF = np.einsum('abc,txyzecfbga->txyzefg', levi(), UG_2PP_2DGF)
        UG_2PP_2DGF = np.einsum('efg,txyzefg->txyz', levi(), UG_2PP_2DGF)
        UG_2PP_2DGF = 0.5*UG_2PP_2DGF
        # momentum projection
        term4 = UG_2PP_2DGF.sum(axis=3).sum(axis=2).sum(axis=1)
        # combine
        fhcorr = term1+term2+term3+term4
        fhcorr[:origin[-1]] = -1*fhcorr[:origin[-1]]
        fhcorr = np.roll(fhcorr,-1*origin[-1])
        return fhcorr

    """ fh contract D """
    def fh_isospin_half_spin_contract_D(prop,fh_prop,P,P_2,G,G_2):
        # F^ab_ij = D^ad_in J_nm D^db_mj
        # corr = 1/2 e_efg e_abc H_qr P_2_s G_ij P_l * [
        #      - U^ea_qi U^gc_sl F^fb_rj
        #      + U^ga_si U^ec_ql F^fb_rj
        #      + two disconnected diagrams]
        # term 1
        # spin contract
        UG = np.einsum('txyzqiea,ij->txyzqjea', prop, G)
        UGG_2 = np.einsum('txyzqjea,qr->txyzrjea', UG, G_2)
        UGG_2F = np.einsum('txyzrjea,txyzrjfb->txyzeafb', UGG_2, fhprop)
        UP_2 = np.einsum('txyzslgc,s->txyzlgc', prop, P_2)
        UP_2P = np.einsum('txyzlgc,l->txyzgc', UP_2, P)
        UGFG_2UP_2P = np.einsum('txyzeafb,txyzgc->txyzeafbgc', UGG_2F, UP_2P)
        # color contract
        UGFG_2UP_2P = np.einsum('abc,txyzeafbgc->txyzefg', levi(), UGFG_2UP_2P)
        UGFG_2UP_2P = np.einsum('efg,txyzefg->txyz', levi(), UGFG_2UP_2P)
        UGFG_2UP_2P = -0.5*UGFG_2UP_2P
        # momentum projection
        term1 = UGFG_2UP_2P.sum(axis=3).sum(axis=2).sum(axis=1)
        # term 2
        # spin contract
        UG = np.einsum('txyzsiga,ij->txyzsjga', prop, G)
        UGP_2 = np.einsum('txyzsjga,s->txyzjga', UG, P_2)
        UGP_2P = np.einsum('txyzjga,l->txyzljga', UGP_2, P)
        UG_2 = np.einsum('txyzqlec,qr->txyzrlec', prop, G_2)
        UG_2F = np.einsum('txyzrlec,txyzrjfb->txyzljecfb', UG_2, fhprop)
        UG_2FUGP_2P = np.einsum('txyzljecfb,txyzljga->txyzecfbga', UG_2F, UGP_2P)
        # color contract
        UG_2FUGP_2P = np.einsum('abc,txyzecfbga->txyzefg', levi(), UG_2FUGP_2P)
        UG_2FUGP_2P = np.einsum('efg,txyzefg->txyz', levi(), UG_2FUGP_2P)
        UG_2FUGP_2P = 0.5*UG_2FUGP_2P
        # momentum projection
        term2 = UG_2FUGP_2P.sum(axis=3).sum(axis=2).sum(axis=1)
        # combine
        fhcorr = term1+term2
        fhcorr[:origin[-1]] = -1*fhcorr[:origin[-1]]
        fhcorr = np.roll(fhcorr,-1*origin[-1])
        return fhcorr

    def allthreeptcontractD(a2aprop,P,P_2,G,G_2,O,origin):
        # derive from fhcontractD
        # F^ab_ij = D^ad_in J_nm D^db_mj
        # fhcorr = 1/2 e_efg e_abc G_2_qr P_2_s G_ij P_l * [
        #        - U^ea_qi U^gc_sl F^fb_rj
        #        + U^ga_si U^ec_ql F^fb_rj ]
        #        = 1/2 e_efg e_abc G_2_qr P_2_s G_ij P_l * [
        #        - U^ea_qi U^gc_sl D^fd_rn J_nm D^db_mj
        #        + U^ga_si U^ec_ql D^fd_rn J_nm D^db_mj ]
        #        = 1/3 e_efg e_abc G_ij P_l D^fd_rn J_nm D^db_mj U^ga_qi U^ec_sl * [
        #        + G_2_qr P_2_s + G_2_sr P_2_q ]
        srcprop = a2aprop[:,:,:,:,origin[0],origin[1],origin[2],origin[3],:,:,:,:]
        # FG_2 prop
        J = Operator(O)
        JD = np.einsum('nm,vqwamjdb->vqwanjdb',J,srcprop) # need to project momentum here if we want to
        DJD = np.einsum('tzyxvqwarnfd,vqwanjdb->tzyxvrjfb',a2aprop,JD) # v = t' # sum over current x y z
        # FH contract
        # term 1
        # spin contract
        prop = srcprop
        fhprop = DJD
        UG = np.einsum('txyzqiea,ij->txyzqjea', prop, G)
        UGG_2 = np.einsum('txyzqjea,qr->txyzrjea', UG, G_2)
        UGG_2F = np.einsum('txyzrjea,txyzvrjfb->txyzveafb', UGG_2, fhprop)
        UP_2 = np.einsum('txyzslgc,s->txyzlgc', prop, P_2)
        UP_2P = np.einsum('txyzlgc,l->txyzgc', UP_2, P)
        UGFG_2UP_2P = np.einsum('txyzveafb,txyzgc->txyzveafbgc', UGG_2F, UP_2P)
        # color contract
        UGFG_2UP_2P = np.einsum('abc,txyzveafbgc->txyzvefg', levi(), UGFG_2UP_2P)
        UGFG_2UP_2P = np.einsum('efg,txyzvefg->txyzv', levi(), UGFG_2UP_2P)
        UGFG_2UP_2P = -0.5*UGFG_2UP_2P
        # momentum projection
        term1 = UGFG_2UP_2P.sum(axis=3).sum(axis=2).sum(axis=1)
        # term 2
        # spin contract
        UG = np.einsum('txyzsiga,ij->txyzsjga', prop, G)
        UGP_2 = np.einsum('txyzsjga,s->txyzjga', UG, P_2)
        UGP_2P = np.einsum('txyzjga,l->txyzljga', UGP_2, P)
        UG_2 = np.einsum('txyzqlec,qr->txyzrlec', prop, G_2)
        UG_2F = np.einsum('txyzrlec,txyzvrjfb->txyzvljecfb', UG_2, fhprop)
        UG_2FUGP_2P = np.einsum('txyzvljecfb,txyzljga->txyzvecfbga', UG_2F, UGP_2P)
        # color contract
        UG_2FUGP_2P = np.einsum('abc,txyzvecfbga->txyzvefg', levi(), UG_2FUGP_2P)
        UG_2FUGP_2P = np.einsum('efg,txyzvefg->txyzv', levi(), UG_2FUGP_2P)
        UG_2FUGP_2P = 0.5*UG_2FUGP_2P
        # momentum projection
        term2 = UG_2FUGP_2P.sum(axis=3).sum(axis=2).sum(axis=1)
        # combine
        fhcorr = term1+term2
        fhcorr[:origin[-1]] = -1*fhcorr[:origin[-1]]
        fhcorr = np.roll(fhcorr,-1*origin[-1])
        return fhcorr

    def main():
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



        # make UU three point
        fhcorr = []
        for i in src:
            for j in snk:
                G, P = Gamma_Proj(i)
                H, Q = Gamma_Proj(j)
                fhcorr.append(fhcontractU(prop,fhprop,P,Q,G,H,origin).imag)
        fhcorrU = np.array(fhcorr).sum(axis=0)/np.sqrt(len(src)*len(snk))
        # make DD three point
        fhcorr = []
        for i in src:
            for j in snk:
                G, P = Gamma_Proj(i)
                H, Q = Gamma_Proj(j)
                fhcorr.append(fhcontractD(prop,fhprop,P,Q,G,H,origin).imag)
        fhcorrD = np.array(fhcorr).sum(axis=0)/np.sqrt(len(src)*len(snk))

    
        
        

