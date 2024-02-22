
# Encoding the ODE system
def deriv(y, t, k_tr, K_pol, k_rdeg, k_tl, K_rib, k_exc, K_T_rep, k_tp, k_cat, k_M):
    
    R_e, R_t, R_r, R_p, P_e, P_t, P_r, P_p, M_a, M_b = y

    dR_edt = k_tr * ( 1/ ( 1 + (K_pol/P_p)*(1+P_t/K_T_rep) ) ) - R_e
    dR_tdt = k_tr * ( 1/ ( 1 + (K_pol/P_p) ) ) - k_rdeg*R_t
    dR_rdt = k_tr * ( 1/ ( 1 + (K_pol/P_p) ) ) - k_rdeg*R_r
    dR_pdt = k_tr * ( 1/ ( 1 + (K_pol/P_p) ) ) - k_rdeg*R_p
    
    dP_edt = k_tl * ( 1/ ( 1 + (K_rib/P_r) ) ) * R_e 
    dP_tdt = k_tl * ( 1/ ( 1 + (K_rib/P_r) ) ) * R_t 
    dP_rdt = k_tl * ( 1/ ( 1 + (K_rib/P_r) ) ) * R_r 
    dP_pdt = k_tl * ( 1/ ( 1 + (K_rib/P_r) ) ) * R_p 

    dM_adt = k_tp - ( (k_cat*P_e*M_a)*(k_M+M_a+P_e) ) - k_exc * M_a
    dM_bdt = ( (k_cat*P_e*M_a)*(k_M+M_a+P_e) ) - k_exc * M_b
    

    return dR_edt, dR_tdt, dR_rdt, dR_pdt, dP_edt, dP_tdt, dP_rdt, dP_pdt, dM_adt, dM_bdt