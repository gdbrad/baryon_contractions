# do contractions with python
# takes in propagator from h5

import numpy as np
import h5py as h5
import spin_stuff as ss
import time
import argparse as ap

def levi():
    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    return eijk

def Gamma_Proj(basak):
    bterms = dict()
    # basak 1
    bterms['G1g1u'] = [1,2,1,1]
    bterms['G1g1d'] = [1,2,2,1]
    bterms['G1u1u'] = [3,4,3,1]
    bterms['G1u1d'] = [3,4,4,1]
    # basak 2
    bterms['G1g2u_1'] = [1,4,3,1]
    bterms['G1g2u_2'] = [3,2,3,1]
    bterms['G1g2u_3'] = [3,4,1,1]
    bterms['G1g2d_1'] = [1,4,4,1]
    bterms['G1g2d_2'] = [3,2,4,1]
    bterms['G1g2d_3'] = [3,4,2,1]
    bterms['G1u2u_1'] = [1,2,3,1]
    bterms['G1u2u_2'] = [1,4,1,1]
    bterms['G1u2u_3'] = [3,2,1,1]
    bterms['G1u2d_1'] = [1,2,4,1]
    bterms['G1u2d_2'] = [1,4,2,1]
    bterms['G1u2d_3'] = [3,2,2,1]
    # basak 3
    bterms['G1g3u_1'] = [1,3,4,1]
    bterms['G1g3u_2'] = [3,2,3,1]
    bterms['G1g3u_3'] = [3,4,1,-1]
    bterms['G1g3d_1'] = [1,4,4,1]
    bterms['G1g3d_2'] = [4,2,3,1]
    bterms['G1g3d_3'] = [3,4,2,-1]
    bterms['G1u3u_1'] = [1,4,1,-1]
    bterms['G1u3u_2'] = [3,1,2,-1]
    bterms['G1u3u_3'] = [1,2,3,1]
    bterms['G1u3d_1'] = [3,2,2,-1]
    bterms['G1u3d_2'] = [2,4,1,-1]
    bterms['G1u3d_3'] = [1,2,4,-1]
    g = Gamma(bterms[basak])
    p = Proj(bterms[basak])
    u, ud = ss.basis_transform()
    up = np.einsum('ij,j->i', u, p)
    gud = np.einsum('ij,jk->ik', g, ud)
    ugud = np.einsum('ij,jk->ik', u, gud)
    return ugud, up

def Proj(bterms): #, bterms_transition):
    p = np.zeros((4))
    i = bterms[2]-1
    #j = bterms_transition[2]-1
    phase = bterms[3]
    p[i] = phase
    return p

def Gamma(bterms):
    g = np.zeros((4,4))
    i = bterms[0]-1
    j = bterms[1]-1
    g[i,j] = 1
    g[j,i] = -1
    return g

def Operator(O):
    g = dict()
    g['4'] = np.array([[complex( 0., 0.),complex( 0., 0.),complex( 0., 1.),complex( 0., 0.)],\
                       [complex( 0., 0.),complex( 0., 0.),complex( 0., 0.),complex( 0., 1.)],\
                       [complex( 0., 1.),complex( 0., 0.),complex( 0., 0.),complex( 0., 0.)],\
                       [complex( 0., 0.),complex( 0., 1.),complex( 0., 0.),complex( 0., 0.)]])
    g['1'] = np.array([[complex( 0., 0.),complex( 0., 0.),complex( 0., 0.),complex( 0., 1.)],\
                       [complex( 0., 0.),complex( 0., 0.),complex( 0., 1.),complex( 0., 0.)],\
                       [complex( 0., 0.),complex( 0.,-1.),complex( 0., 0.),complex( 0., 0.)],\
                       [complex( 0.,-1.),complex( 0., 0.),complex( 0., 0.),complex( 0., 0.)]])
    g['2'] = np.array([[complex( 0., 0.),complex( 0., 0.),complex( 0., 0.),complex(-1, 0.)],\
                       [complex( 0., 0.),complex( 0., 0.),complex( 1., 0.),complex( 0., 0.)],\
                       [complex( 0., 0.),complex( 1., 0.),complex( 0., 0.),complex( 0., 0.)],\
                       [complex(-1., 0.),complex( 0., 0.),complex( 0., 0.),complex( 0., 0.)]])
    g['3'] = np.array([[complex( 0., 0.),complex( 0., 0.),complex( 0., 1.),complex( 0., 0.)],\
                       [complex( 0., 0.),complex( 0., 0.),complex( 0., 0.),complex( 0.,-1.)],\
                       [complex( 0.,-1.),complex( 0., 0.),complex( 0., 0.),complex( 0., 0.)],\
                       [complex( 0., 0.),complex( 0., 1.),complex( 0., 0.),complex( 0., 0.)]])
    g['5'] = np.array([[complex( 1., 0.),complex( 0., 0.),complex( 0., 0.),complex( 0., 0.)],\
                       [complex( 0., 0.),complex( 1., 0.),complex( 0., 0.),complex( 0., 0.)],\
                       [complex( 0., 0.),complex( 0., 0.),complex(-1., 0.),complex( 0., 0.)],\
                       [complex( 0., 0.),complex( 0., 0.),complex( 0., 0.),complex(-1., 0.)]])
    if O[0] in ['V']:
        J = g[O[1]]
    elif O[0] in ['A']:
        J = np.einsum('ij,jk->ik',g[O[1]],g['5'])
    elif O[0] in ['g']:
        J = g[O[1]]
    return J

def read_prop(filename,datapath):
    f = h5.File(filename,'r')
    data = f[datapath][()]
    #data = data[:,:,src,snk] #00 is particle particle smear smear
    #f.close()
    return data

def contract(prop,P,Q,G,H,origin):
    # 1/2 e_efg e_abc Q_s P_l H_qr G_ij [ U^ec_ql D^fb_rj U^ga_si - U^ea_qi D^fb_rj U^gc_sl ]
    # term 1
    # spin contract
    UP = np.einsum('txyzqlec,l->txyzqec', prop, P)
    UPQ = np.einsum('txyzqec,s->txyzqsec', UP, Q)
    UG = np.einsum('txyzsiga,ij->txyzsjga', prop, G)
    DH = np.einsum('txyzrjfb,qr->txyzqjfb', prop, H)
    UGDH = np.einsum('txyzsjga,txyzqjfb->txyzsqgafb', UG, DH)
    UGDHUPQ = np.einsum('txyzsqgafb,txyzqsec->txyzgafbec', UGDH, UPQ)
    # color contract
    UGDHUPQ = np.einsum('abc,txyzgafbec->txyzgfe', levi(), UGDHUPQ)
    UGDHUPQ = np.einsum('efg,txyzgfe->txyz', levi(), UGDHUPQ)
    UGDHUPQ = 0.5*UGDHUPQ
    # momentum projection
    term1 = UGDHUPQ.sum(axis=3).sum(axis=2).sum(axis=1)
    # term 2
    # spin contract
    UG = np.einsum('txyzqiea,ij->txyzqjea', prop, G)
    DH = np.einsum('txyzrjfb,qr->txyzqjfb', prop, H)
    UP = np.einsum('txyzslgc,l->txyzsgc', prop, P)
    UPQ = np.einsum('txyzsgc,s->txyzgc', UP, Q)
    UGDH = np.einsum('txyzqjea,txyzqjfb->txyzeafb', UG, DH)
    # color contract
    UPQ = np.einsum('abc,txyzgc->txyzabg', levi(), UPQ)
    UPQ = np.einsum('efg,txyzabg->txyzabef', levi(), UPQ)
    UPQUGDH = np.einsum('txyzabef,txyzeafb->txyz', UPQ, UGDH)
    UPQUGDH = 0.5*UPQUGDH
    # momentum projection
    term2 = UPQUGDH.sum(axis=3).sum(axis=2).sum(axis=1)
    # difference
    corr = term1-term2
    corr[:origin[-1]] = -1*corr[:origin[-1]]
    corr = np.roll(corr,-1*origin[-1])
    return corr

def fhcontractU(prop,fhprop,P,Q,G,H,origin):
    # F^ab_ij = U^ad_in J_nm U^db_mj
    # corr = 1/2 e_efg e_abc Q_s P_l * [
    #      + F^ec_ql U^ga_si G_ij D^fb_rj H_qr
    #      - F^ea_qi G_ij D^fb_rj H_qr U^gc_sl
    #      - U^ea_qi G_ij D^fb_rj H_qr F^gc_sl
    #      + U^ec_ql F^ga_si G_ij D^fb_rj H_qr
    #      + two disconnected diagrams]
    # term 1
    # spin contract
    FP = np.einsum('txyzqlec,l->txyzqec', fhprop, P)
    FPQ = np.einsum('txyzqec,s->txyzqsec', FP, Q)
    FPQH = np.einsum('txyzqsec,qr->txyzrsec', FPQ, H)
    UG = np.einsum('txyzsiga,ij->txyzsjga', prop, G)
    UGD = np.einsum('txyzsjga,txyzrjfb->txyzsrgafb', UG, prop)
    UGDFPQH = np.einsum('txyzsrgafb,txyzrsec->txyzgafbec', UGD, FPQH)
    # color contract
    UGDFPQH = np.einsum('abc,txyzgafbec->txyzgfe', levi(), UGDFPQH)
    UGDFPQH = np.einsum('efg,txyzgfe->txyz', levi(), UGDFPQH)
    UGDFPQH = 0.5*UGDFPQH
    # momentum projection
    term1 = UGDFPQH.sum(axis=3).sum(axis=2).sum(axis=1)
    # term 2
    # spin contract
    FG = np.einsum('txyzqiea,ij->txyzqjea', fhprop, G)
    FGH = np.einsum('txyzqjea,qr->txyzrjea', FG, H)
    FGHD = np.einsum('txyzrjea,txyzrjfb->txyzeafb', FGH, prop)
    UQ = np.einsum('txyzslgc,s->txyzlgc', prop, Q)
    UQP = np.einsum('txyzlgc,l->txyzgc', UQ, P)
    FGHDUQP = np.einsum('txyzeafb,txyzgc->txyzeafbgc', FGHD, UQP)
    # color contract
    FGHDUQP = np.einsum('abc,txyzeafbgc->txyzefg', levi(), FGHDUQP)
    FGHDUQP = np.einsum('efg,txyzefg->txyz', levi(), FGHDUQP)
    FGHDUQP = -0.5*FGHDUQP
    # momentum projection
    term2 = FGHDUQP.sum(axis=3).sum(axis=2).sum(axis=1)
    # term 3
    # spin contract
    UG = np.einsum('txyzqiea,ij->txyzqjea', prop, G)
    UGH = np.einsum('txyzqjea,qr->txyzrjea', UG, H)
    UGHD = np.einsum('txyzrjea,txyzrjfb->txyzeafb', UGH, prop)
    FQ = np.einsum('txyzslgc,s->txyzlgc', fhprop, Q)
    FQP = np.einsum('txyzlgc,l->txyzgc', FQ, P)
    UGHDFQP = np.einsum('txyzeafb,txyzgc->txyzeafbgc', UGHD, FQP)
    # color contract
    UGHDFQP = np.einsum('abc,txyzeafbgc->txyzefg', levi(), UGHDFQP)
    UGHDFQP = np.einsum('efg,txyzefg->txyz', levi(), UGHDFQP)
    UGHDFQP = -0.5*UGHDFQP
    # momentum projection
    term3 = UGHDFQP.sum(axis=3).sum(axis=2).sum(axis=1)
    # term 4
    # spin contract
    UH = np.einsum('txyzqlec,qr->txyzrlec', prop, H)
    UHP = np.einsum('txyzrlec,l->txyzrec', UH, P)
    UHPQ = np.einsum('txyzrec,s->txyzrsec', UHP, Q)
    DG = np.einsum('txyzrjfb,ij->txyzrifb', prop, G)
    DGF = np.einsum('txyzrifb,txyzsiga->txyzrsfbga', DG, fhprop)
    UHPQDGF = np.einsum('txyzrsec,txyzrsfbga->txyzecfbga', UHPQ, DGF)
    # color contract
    UHPQDGF = np.einsum('abc,txyzecfbga->txyzefg', levi(), UHPQDGF)
    UHPQDGF = np.einsum('efg,txyzefg->txyz', levi(), UHPQDGF)
    UHPQDGF = 0.5*UHPQDGF
    # momentum projection
    term4 = UHPQDGF.sum(axis=3).sum(axis=2).sum(axis=1)
    # combine
    fhcorr = term1+term2+term3+term4
    fhcorr[:origin[-1]] = -1*fhcorr[:origin[-1]]
    fhcorr = np.roll(fhcorr,-1*origin[-1])
    return fhcorr

def fhcontractD(prop,fhprop,P,Q,G,H,origin):
    # F^ab_ij = D^ad_in J_nm D^db_mj
    # corr = 1/2 e_efg e_abc H_qr Q_s G_ij P_l * [
    #      - U^ea_qi U^gc_sl F^fb_rj
    #      + U^ga_si U^ec_ql F^fb_rj
    #      + two disconnected diagrams]
    # term 1
    # spin contract
    UG = np.einsum('txyzqiea,ij->txyzqjea', prop, G)
    UGH = np.einsum('txyzqjea,qr->txyzrjea', UG, H)
    UGHF = np.einsum('txyzrjea,txyzrjfb->txyzeafb', UGH, fhprop)
    UQ = np.einsum('txyzslgc,s->txyzlgc', prop, Q)
    UQP = np.einsum('txyzlgc,l->txyzgc', UQ, P)
    UGFHUQP = np.einsum('txyzeafb,txyzgc->txyzeafbgc', UGHF, UQP)
    # color contract
    UGFHUQP = np.einsum('abc,txyzeafbgc->txyzefg', levi(), UGFHUQP)
    UGFHUQP = np.einsum('efg,txyzefg->txyz', levi(), UGFHUQP)
    UGFHUQP = -0.5*UGFHUQP
    # momentum projection
    term1 = UGFHUQP.sum(axis=3).sum(axis=2).sum(axis=1)
    # term 2
    # spin contract
    UG = np.einsum('txyzsiga,ij->txyzsjga', prop, G)
    UGQ = np.einsum('txyzsjga,s->txyzjga', UG, Q)
    UGQP = np.einsum('txyzjga,l->txyzljga', UGQ, P)
    UH = np.einsum('txyzqlec,qr->txyzrlec', prop, H)
    UHF = np.einsum('txyzrlec,txyzrjfb->txyzljecfb', UH, fhprop)
    UHFUGQP = np.einsum('txyzljecfb,txyzljga->txyzecfbga', UHF, UGQP)
    # color contract
    UHFUGQP = np.einsum('abc,txyzecfbga->txyzefg', levi(), UHFUGQP)
    UHFUGQP = np.einsum('efg,txyzefg->txyz', levi(), UHFUGQP)
    UHFUGQP = 0.5*UHFUGQP
    # momentum projection
    term2 = UHFUGQP.sum(axis=3).sum(axis=2).sum(axis=1)
    # combine
    fhcorr = term1+term2
    fhcorr[:origin[-1]] = -1*fhcorr[:origin[-1]]
    fhcorr = np.roll(fhcorr,-1*origin[-1])
    return fhcorr

def seqtwoptcontractU(seqprop,origin):
    # derive from fhcontractU
    # fhcorr = 1/2 e_efg e_abc Q_s P_l * [
    #        + F^ec_ql U^ga_si G_ij D^fb_rj H_qr
    #        - F^ea_qi G_ij D^fb_rj H_qr U^gc_sl
    #        - U^ea_qi G_ij D^fb_rj H_qr F^gc_sl
    #        + U^ec_ql F^ga_si G_ij D^fb_rj H_qr ]
    #
    #        = 1/2 e_efg e_abc Q_s P_l G_ij D^fb_rj H_qr * [
    #        + U^ed_qn J_nm U^dc_ml U^ga_si - U^gc_sl U^ed_qn J_nm U^da_mi
    #        - U^eq_qi U^gd_sn J_nm U^dc_ml + U^ec_ql U^gd_sn J_nm U^da_mi ]
    #
    # seqprop drops J_mn U^dc_ml and J_nm U^da_mi
    # anti-commute color indices
    # seqprop = 1/2 e_efg e_abc Q_s P_l G_ij D^fb_rj H_qr * [
    #         + U^gd_qn U^ec_si + U^ec_sl U^gd_qn + U^ec_qi U^gd_sn + U^ec_ql U^gd_sn ]
    # swap indices s <--> q
    #         = 1/2 e_efg e_abc P_l G_ij D^fb_rj U^gd_qn * [
    #         + Q_s H_qr * (U^ec_si U^ec_sl) + Q_q H_sr * (U^ec_si + U^ec_sl) ]
    # swap indices i <--> l
    #         = 1/2 e_efg e_abc D^fb_rj U^gd_qn U^ec_si
    #         * (P_l G_ij + P_i G_lj) * (Q_s H_qr + Q_q H_sr)
    # seqprop is inverted off the snk however
    #         = 1/2 e_efg e_abc D^fb_rj U^ec_si gamma5_qx U+^gd_xn gamma5_ny
    #         * (P_l G_ij + P_i G_lj) * (Q_s H_qr + Q_q H_sr)
    # the seqprop doesn't multiply by the final gamma5_ny
    #         = 1/2 e_efg e_abc D^fb_rj U^ec_si gamma5_qx U+^gd_xn
    #         * (P_l G_ij + P_i G_lj) * (Q_s H_qr + Q_q H_sr)
    # make two point
    # seqprop^ad_ln gamma5_ny delta_ad delta_yl
    gamma5 = np.array([[ 1.,  0.,  0.,  0.], \
                       [ 0.,  1.,  0.,  0.], \
                       [ 0.,  0., -1.,  0.], \
                       [ 0.,  0.,  0., -1.]])
    #gamma5 = np.array([[  0.,  0.,  1.,  0.], \
    #                   [  0.,  0.,  0.,  1.], \
    #                   [  1.,  0.,  0.,  0.], \
    #                   [  0.,  1.,  0.,  0.]])
    # seqprop[t0,x1,y2,z3,s4,s5,c6,c7]
    
    seqprop1 = np.einsum('kl,txyzlnad,nm->txyzkmad', gamma5,seqprop, gamma5)
    term1 = 0.5*np.trace(np.trace(seqprop1, axis1=6, axis2=7), axis1=4, axis2=5)
    # momentum projection
    seqtwoptcorr = term1 #.sum(axis=3).sum(axis=2).sum(axis=1)
    return seqtwoptcorr

def seqtwoptcontractD(seqprop,origin):
    # derive from fhcontractD
    # F^ab_ij = D^ad_in J_nm D^db_mj
    # fhcorr = 1/2 e_efg e_abc H_qr Q_s G_ij P_l * [
    #        - U^ea_qi U^gc_sl F^fb_rj
    #        + U^ga_si U^ec_ql F^fb_rj ]
    #        = 1/2 e_efg e_abc H_qr Q_s G_ij P_l * [
    #        - U^ea_qi U^gc_sl D^fd_rn J_nm D^db_mj
    #        + U^ga_si U^ec_ql D^fd_rn J_nm D^db_mj ]
    # seqcorr = 1/2 e_efg e_abc H_qr Q_s G_ij P_l * [
    #         - U^ea_qi U^gc_sl D^fd_rn
    #         + U^ga_si U^ec_ql D^fd_rn ]
    #         = 1/2 e_efg e_abc H_qr Q_s G_ij P_l D^fd_rn * [
    #         + U^ga_qi U^ec_sl
    #         + U^ec_si U^ga_ql ]
    #         = 1/2 e_efg e_abc H_qr Q_s U^ga_ql D^fd_rn U^ec_si * [ P_l G_ij + P_i G_lj ]
    seqprop1 = np.einsum('kl,txyzlnad,nm->txyzkmad', gamma5,seqprop, gamma5)
    term1 = np.trace(np.trace(seqprop1, axis1=6, axis2=7), axis1=4, axis2=5)
    # momentum projection
    seqtwoptcorr = term1 #.sum(axis=3).sum(axis=2).sum(axis=1)
    return seqtwoptcorr

def allthreeptcontractD(a2aprop,P,Q,G,H,O,origin):
    # derive from fhcontractD
    # F^ab_ij = D^ad_in J_nm D^db_mj
    # fhcorr = 1/2 e_efg e_abc H_qr Q_s G_ij P_l * [
    #        - U^ea_qi U^gc_sl F^fb_rj
    #        + U^ga_si U^ec_ql F^fb_rj ]
    #        = 1/2 e_efg e_abc H_qr Q_s G_ij P_l * [
    #        - U^ea_qi U^gc_sl D^fd_rn J_nm D^db_mj
    #        + U^ga_si U^ec_ql D^fd_rn J_nm D^db_mj ]
    #        = 1/3 e_efg e_abc G_ij P_l D^fd_rn J_nm D^db_mj U^ga_qi U^ec_sl * [
    #        + H_qr Q_s + H_sr Q_q ]
    srcprop = a2aprop[:,:,:,:,origin[0],origin[1],origin[2],origin[3],:,:,:,:]
    # FH prop
    J = Operator(O)
    JD = np.einsum('nm,vqwamjdb->vqwanjdb',J,srcprop) # need to project momentum here if we want to
    DJD = np.einsum('tzyxvqwarnfd,vqwanjdb->tzyxvrjfb',a2aprop,JD) # v = t' # sum over current x y z
    # FH contract
    # term 1
    # spin contract
    prop = srcprop
    fhprop = DJD
    UG = np.einsum('txyzqiea,ij->txyzqjea', prop, G)
    UGH = np.einsum('txyzqjea,qr->txyzrjea', UG, H)
    UGHF = np.einsum('txyzrjea,txyzvrjfb->txyzveafb', UGH, fhprop)
    UQ = np.einsum('txyzslgc,s->txyzlgc', prop, Q)
    UQP = np.einsum('txyzlgc,l->txyzgc', UQ, P)
    UGFHUQP = np.einsum('txyzveafb,txyzgc->txyzveafbgc', UGHF, UQP)
    # color contract
    UGFHUQP = np.einsum('abc,txyzveafbgc->txyzvefg', levi(), UGFHUQP)
    UGFHUQP = np.einsum('efg,txyzvefg->txyzv', levi(), UGFHUQP)
    UGFHUQP = -0.5*UGFHUQP
    # momentum projection
    term1 = UGFHUQP.sum(axis=3).sum(axis=2).sum(axis=1)
    # term 2
    # spin contract
    UG = np.einsum('txyzsiga,ij->txyzsjga', prop, G)
    UGQ = np.einsum('txyzsjga,s->txyzjga', UG, Q)
    UGQP = np.einsum('txyzjga,l->txyzljga', UGQ, P)
    UH = np.einsum('txyzqlec,qr->txyzrlec', prop, H)
    UHF = np.einsum('txyzrlec,txyzvrjfb->txyzvljecfb', UH, fhprop)
    UHFUGQP = np.einsum('txyzvljecfb,txyzljga->txyzvecfbga', UHF, UGQP)
    # color contract
    UHFUGQP = np.einsum('abc,txyzvecfbga->txyzvefg', levi(), UHFUGQP)
    UHFUGQP = np.einsum('efg,txyzvefg->txyzv', levi(), UHFUGQP)
    UHFUGQP = 0.5*UHFUGQP
    # momentum projection
    term2 = UHFUGQP.sum(axis=3).sum(axis=2).sum(axis=1)
    # combine
    fhcorr = term1+term2
    fhcorr[:origin[-1]] = -1*fhcorr[:origin[-1]]
    fhcorr = np.roll(fhcorr,-1*origin[-1])
    return fhcorr

def main(prop='./test_propagator_PP.h5',
         propdir='/pt/PP_prop_0',
         fhprop=False,
         fhpropdir=False,
         seqprop='./test_seqprop_PP.h5',
         seqpropdir='/PP_seqprop_proton_DD_up_up_tsrc_0_tsep_4',
         seqsrc='./test_seqsource_PP.h5',
         seqsrcdir='/PP_sink_proton_UU_up_up_tsrc_0_tsep_4',
         origin=[0,0,0,0],
         src=['G1g1u'],
         snk=['G1g1u'],
         allprop='./all_pt_props.h5'):
    # prop      : propagator file path
    # fhprop    : fhprop file path
    # seqprop   : seqprop file path
    # propdir   : h5 path to propagator
    # fhpropdir : h5 path to fhprop
    # seqpropdir: h5 path to seqprop
    # origin    : x,y,z,t of origin position
    # src / snk : 'G1g1' positive parity, 'G1u1' negative parity / 'u' spin up, 'd' spin down
    # other src / snk options in Gamma_Proj()
    # flags and read data
    if prop is not False and propdir is not False:
        print("making two point correlator from propagator")
        twopt = True
        # read propagator
        prop = read_prop(prop, propdir)
        print("len(prop)  :", np.shape(prop))
    else:
        twopt = False
        print("no propagator")
    if prop is not False and propdir is not False and fhprop is not False and fhpropdir is not False:
        print("making UU and DD fh correlator from prop and fhprop")
        fhthreept = True
        # read fh propagator
        fhprop = read_prop(fhprop, fhpropdir)
        print("len(fhprop):", np.shape(fhprop))
    else:
        fhthreept = False
        print("no fhprop")
    if seqprop is not False and seqpropdir is not False:
        print("making two point correlator from sequential propagator")
        seqtwopt = True
        # read seq propagator
        seqprop = read_prop(seqprop, seqpropdir)
        print("len(seqprop):", np.shape(seqprop))
        # read seq source
        seqsrc = read_prop(seqsrc, seqsrcdir)
        print("len(seqsrc):", np.shape(seqsrc))
    else:
        seqtwopt = False
        print("no seqprop")
    if allprop is not False:
        print("making three point correlator from all to all prop")
        allthreept = True
        # read all to all prop
        a2aprop = np.zeros((8,4,4,4,8,4,4,4,4,4,3,3),dtype=np.complex128)
        for t in range(8):
            for z in range(4):
                for y in range(4):
                    for x in range(4):
                        a2aprop[:,:,:,:,t,z,y,x,:,:] = h5.File(allprop)['/props/pt_prop_x%s_y%s_z%s_t%s' %(x,y,z,t)][()]
        print("len(a2aprop):", np.shape(a2aprop))
    else:
        allthreept = False
        print("no all to all prop")
    # make correlation functions
    if twopt:
        # make two point
        corr = []
        for i in src:
            for j in snk:
                G, P = Gamma_Proj(i)
                H, Q = Gamma_Proj(j)
                corr.append(contract(prop,P,Q,G,H,origin).real)
        corr = np.array(corr).sum(axis=0)/np.sqrt(len(src)*len(snk))
    if fhthreept:
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
    if seqtwopt:
        seqtwoptcorr = seqtwoptcontractU(seqprop,origin).real
        a = np.array(seqtwoptcorr-corr[4]).flatten()
        a = [i for i in a if abs(i) < 1E-7]
        print(seqtwoptcorr[0,0,0,0])
        print(corr)
    if allthreept:
        allthreeptcorr = allthreeptcontractD(a2aprop,P,Q,G,H,'A3',origin) 
        print(allthreeptcorr)
if __name__=='__main__':
    main()

