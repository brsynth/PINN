r1 = lambda V, S, Km, I, Ki : V*S/((Km+S)*(1+I/Ki))

v_glycolysis = lambda v_max_glycolysis, GLC, Km_GLC, ACE_env, Ki_ACE_glycolysis : r1(v_max_glycolysis, GLC, Km_GLC, ACE_env, Ki_ACE_glycolysis)

v_TCA_cycle = lambda v_max_TCA_cycle, ACCOA, Km_ACCOA_TCA_cycle, ACE_env, Ki_ACE_TCA_cycle : r1(v_max_TCA_cycle, ACCOA, Km_ACCOA_TCA_cycle, ACE_env, Ki_ACE_TCA_cycle)

v_AckA = lambda v_max_AckA, ACP, ADP, ACE_cell, ATP, Keq_AckA, Km_ACP_AckA, Km_ADP, Km_ATP, Km_ACE_AckA : \
   v_max_AckA*(ACP*ADP-ACE_cell*ATP/Keq_AckA)/(Km_ACP_AckA*Km_ADP)/((1+ACP/Km_ACP_AckA+ACE_cell/Km_ACE_AckA)*(1+ADP/Km_ADP+ATP/Km_ATP))

v_Pta = lambda v_max_Pta, ACCOA, P, ACP, COA, Keq_Pta, Km_ACP_Pta, Km_P, Km_ACCOA_Pta, Ki_P, Ki_ACP, Km_COA : \
   v_max_Pta*(ACCOA*P-ACP*COA/Keq_Pta)/(Km_ACCOA_Pta*Km_P)/(1+ACCOA/Km_ACCOA_Pta+P/Ki_P+ACP/Ki_ACP+COA/Km_COA+ACCOA*P/(Km_ACCOA_Pta*Km_P)+ACP*COA/(Km_ACP_Pta*Km_COA))

v_acetate_exchange = lambda v_max_acetate_exchange, ACE_cell, ACE_env, Keq_acetate_exchange, Km_ACE_acetate_exchange : \
   v_max_acetate_exchange*(ACE_cell-ACE_env/Keq_acetate_exchange)/Km_ACE_acetate_exchange/(1+ACE_cell/Km_ACE_acetate_exchange+ACE_env/Km_ACE_acetate_exchange)


ODE_residual_dict_Millard = {
                     "ode_1" : 
                     lambda var_dict,d_dt_var_dict,value,min_var_dict,max_var_dict :
                        d_dt_var_dict["GLC"] - v_glycolysis(v_max_glycolysis=value["v_max_glycolysis"],
                                                            GLC=var_dict["GLC"],
                                                            Km_GLC=value["Km_GLC"],
                                                            ACE_env=var_dict["ACE_env"],
                                                            Ki_ACE_glycolysis=value["Ki_ACE_glycolysis"])
                                             *var_dict["X"]
                                             *(value["volume"])
                                             *(value["v_feed"]-value["D"]*var_dict["GLC"])
                                             /(max_var_dict["GLC"] - min_var_dict["GLC"]),
                     "ode_2" : 
                     lambda var_dict,d_dt_var_dict,value,min_var_dict,max_var_dict :
                        d_dt_var_dict["ACE_env"] - v_acetate_exchange(v_max_acetate_exchange=value["v_max_acetate_exchange"],
                                                                      ACE_cell=var_dict["ACE_cell"],
                                                                      ACE_env=var_dict["ACE_env"],
                                                                      Keq_acetate_exchange=value["Keq_acetate_exchange"],
                                                                      Km_ACE_acetate_exchange=value["Km_ACE_acetate_exchange"])
                                                   *var_dict["X"]
                                                   *(value["volume"])
                                                   *(-value["D"]
                                                   *var_dict["ACE_env"])
                                                   /(max_var_dict["ACE_env"] - min_var_dict["ACE_env"]),
                     "ode_3" : 
                     lambda var_dict,d_dt_var_dict,value,min_var_dict,max_var_dict :
                        d_dt_var_dict["X"] - var_dict["X"]
                                             *v_TCA_cycle(v_max_TCA_cycle=value["v_max_TCA_cycle"],
                                                          ACCOA=var_dict["ACCOA"],
                                                          Km_ACCOA_TCA_cycle=value["Km_ACCOA_TCA_cycle"],
                                                          ACE_env=var_dict["ACE_env"],
                                                          Ki_ACE_TCA_cycle=value["Ki_ACE_TCA_cycle"])
                                             *value["Y"]
                                             *(-value["D"]*var_dict["X"])
                                             /(max_var_dict["X"] - min_var_dict["X"]),
                     "ode_4" : 
                     lambda var_dict,d_dt_var_dict,value,min_var_dict,max_var_dict :
                        d_dt_var_dict["ACCOA"] - (1.4*v_glycolysis(v_max_glycolysis=value["v_max_glycolysis"],
                                                                   GLC=var_dict["GLC"],
                                                                   Km_GLC=value["Km_GLC"],
                                                                   ACE_env=var_dict["ACE_env"],
                                                                   Ki_ACE_glycolysis=value["Ki_ACE_glycolysis"]) 
                                                   - v_Pta(v_max_Pta=value["v_max_Pta"],
                                                           ACCOA=var_dict["ACCOA"],
                                                           P=value["P"],
                                                           ACP=var_dict["ACP"],
                                                           COA=value["COA"],
                                                           Keq_Pta=value["Keq_Pta"],
                                                           Km_ACP_Pta=value["Km_ACP_Pta"],
                                                           Km_P=value["Km_P"],
                                                           Km_ACCOA_Pta=value["Km_ACCOA_Pta"],
                                                           Ki_P=value["Ki_P"],
                                                           Ki_ACP=value["Ki_ACP"],
                                                           Km_COA=value["Km_COA"]) 
                                                   - v_TCA_cycle(v_max_TCA_cycle=value["v_max_TCA_cycle"],
                                                                 ACCOA=var_dict["ACCOA"], 
                                                                 Km_ACCOA_TCA_cycle=value["Km_ACCOA_TCA_cycle"],
                                                                 ACE_env=var_dict["ACE_env"],
                                                                 Ki_ACE_TCA_cycle=value["Ki_ACE_TCA_cycle"])
                                                   )/(max_var_dict["ACCOA"] - min_var_dict["ACCOA"]),
                     "ode_5" : 
                     lambda var_dict,d_dt_var_dict,value,min_var_dict,max_var_dict :
                        d_dt_var_dict["ACP"] - (v_Pta(v_max_Pta=value["v_max_Pta"],
                                                      ACCOA=var_dict["ACCOA"],
                                                      P=value["P"],
                                                      ACP=var_dict["ACP"],
                                                      COA=value["COA"],
                                                      Keq_Pta=value["Keq_Pta"],
                                                      Km_ACP_Pta=value["Km_ACP_Pta"],
                                                      Km_P=value["Km_P"], 
                                                      Km_ACCOA_Pta=value["Km_ACCOA_Pta"],
                                                      Ki_P=value["Ki_P"],
                                                      Ki_ACP=value["Ki_ACP"],
                                                      Km_COA=value["Km_COA"]) 
                                                - v_AckA(v_max_AckA=value["v_max_AckA"],
                                                         ACP=var_dict["ACP"],
                                                         ADP=value["ADP"],
                                                         ACE_cell=var_dict["ACE_cell"],
                                                         ATP=value["ATP"],
                                                         Keq_AckA=value["Keq_AckA"],
                                                         Km_ACP_AckA=value["Km_ACP_AckA"],
                                                         Km_ADP=value["Km_ADP"],
                                                         Km_ATP=value["Km_ATP"],
                                                         Km_ACE_AckA=value["Km_ACE_AckA"])
                                                )/(max_var_dict["ACP"] - min_var_dict["ACP"]),
                     "ode_6" : 
                     lambda var_dict,d_dt_var_dict,value,min_var_dict,max_var_dict :
                        d_dt_var_dict["ACE_cell"] - (v_AckA(v_max_AckA=value["v_max_AckA"],
                                                            ACP=var_dict["ACP"],
                                                            ADP=value["ADP"],
                                                            ACE_cell=var_dict["ACE_cell"],
                                                            ATP=value["ATP"],
                                                            Keq_AckA=value["Keq_AckA"],
                                                            Km_ACP_AckA=value["Km_ACP_AckA"], 
                                                            Km_ADP=value["Km_ADP"],
                                                            Km_ATP=value["Km_ATP"],
                                                            Km_ACE_AckA=value["Km_ACE_AckA"]) 
                                                   - v_acetate_exchange(v_max_acetate_exchange=value["v_max_acetate_exchange"],
                                                                        ACE_cell=var_dict["ACE_cell"],
                                                                        ACE_env=var_dict["ACE_env"],
                                                                        Keq_acetate_exchange=value["Keq_acetate_exchange"],
                                                                        Km_ACE_acetate_exchange=value["Km_ACE_acetate_exchange"])
                                                   )/(max_var_dict["ACE_cell"] - min_var_dict["ACE_cell"]),
                    }