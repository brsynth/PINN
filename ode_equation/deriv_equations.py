
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

    dM_adt = k_tp - ( (k_cat*P_e*M_a)/(k_M+M_a+P_e) ) - k_exc * M_a
    dM_bdt = ( (k_cat*P_e*M_a)/(k_M+M_a+P_e) ) - k_exc * M_b

    return dR_edt, dR_tdt, dR_rdt, dR_pdt, dP_edt, dP_tdt, dP_rdt, dP_pdt, dM_adt, dM_bdt


ODE_residual_dict = {
                     "ode_1" : 
                     lambda var_dict,d_dt_var_dict,params_dict,min_var_dict,max_var_dict :
                        d_dt_var_dict["R_e"] - (params_dict["k_tr"]* (1/ (1 + (params_dict["K_pol"]/var_dict["P_p"]) * \
                                                                              (1+var_dict["P_t"]/params_dict["K_T_rep"])
                                                                         ))- var_dict["R_e"]
                                                ) /(max_var_dict["R_e"] - min_var_dict["R_e"]),
                     "ode_2" : 
                     lambda var_dict,d_dt_var_dict,params_dict,min_var_dict,max_var_dict :
                        d_dt_var_dict["R_t"] - (params_dict["k_tr"]* (1/ (1 + (params_dict["K_pol"]/var_dict["P_p"])
                                                                         ))- params_dict["k_rdeg"] * var_dict["R_t"]
                                                ) /(max_var_dict["R_t"] - min_var_dict["R_t"]),
                     "ode_3" : 
                     lambda var_dict,d_dt_var_dict,params_dict,min_var_dict,max_var_dict :
                        d_dt_var_dict["R_r"] - (params_dict["k_tr"]* (1/ (1 + (params_dict["K_pol"]/var_dict["P_p"])
                                                                         ))- params_dict["k_rdeg"] * var_dict["R_r"]
                                                ) /(max_var_dict["R_r"] - min_var_dict["R_r"]),
                     "ode_4" : 
                     lambda var_dict,d_dt_var_dict,params_dict,min_var_dict,max_var_dict :
                        d_dt_var_dict["R_p"] - (params_dict["k_tr"]* (1/ (1 + (params_dict["K_pol"]/var_dict["P_p"])
                                                                         ))- params_dict["k_rdeg"] * var_dict["R_p"]
                                                ) /(max_var_dict["R_p"] - min_var_dict["R_p"]),
                     "ode_5" : 
                     lambda var_dict,d_dt_var_dict,params_dict,min_var_dict,max_var_dict :
                        d_dt_var_dict["P_e"] - (params_dict["k_tl"]* (1/ (1 + (params_dict["K_rib"]/var_dict["P_r"])
                                                                         )) * var_dict["R_e"]
                                                ) /(max_var_dict["P_e"] - min_var_dict["P_e"]),
                     "ode_6" : 
                     lambda var_dict,d_dt_var_dict,params_dict,min_var_dict,max_var_dict :
                        d_dt_var_dict["P_t"] - (params_dict["k_tl"]* (1/ (1 + (params_dict["K_rib"]/var_dict["P_r"])
                                                                         )) * var_dict["R_t"]
                                                ) /(max_var_dict["P_t"] - min_var_dict["P_t"]),
                     "ode_7" : 
                     lambda var_dict,d_dt_var_dict,params_dict,min_var_dict,max_var_dict :                        
                        d_dt_var_dict["P_r"] - (params_dict["k_tl"]* (1/ (1 + (params_dict["K_rib"]/var_dict["P_r"]))) * var_dict["R_r"]
                                            ) /(max_var_dict["P_r"] - min_var_dict["P_r"]),
                     "ode_8" : 
                     lambda var_dict,d_dt_var_dict,params_dict,min_var_dict,max_var_dict :                        
                        d_dt_var_dict["P_p"] - (params_dict["k_tl"]* (1/ (1 + (params_dict["K_rib"]/var_dict["P_r"])
                                                                         )) * var_dict["R_p"]
                                                ) /(max_var_dict["P_p"] - min_var_dict["P_p"]),

                     "ode_9" : 
                     lambda var_dict,d_dt_var_dict,params_dict,min_var_dict,max_var_dict :        
                        d_dt_var_dict["M_a"] -  ( params_dict["k_tp"] - ( (params_dict["k_cat"] * var_dict["P_e"] * var_dict["M_a"])/ \
                                                                          (params_dict["k_M"] + var_dict["M_a"] + var_dict["P_e"]) ) \
                                                                      -params_dict["k_exc"] * var_dict["M_a"]
                                              )  / (max_var_dict["M_a"] - min_var_dict["M_a"]),

                     "ode_10" : 
                     lambda var_dict,d_dt_var_dict,params_dict,min_var_dict,max_var_dict :
                        d_dt_var_dict["M_b"] - ( ((params_dict["k_cat"] * var_dict["P_e"] * var_dict["M_a"]) /\
                                                  (params_dict["k_M"] + var_dict["M_a"] + var_dict["P_e"]))\
                                                 -params_dict["k_exc"] * var_dict["M_b"]
                                    )/ (max_var_dict["M_b"] - min_var_dict["M_b"]),
                                                }

