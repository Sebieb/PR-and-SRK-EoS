Simple implementation of the Peng-Robinson and Soave-Redlich-Kwong equation of states, 
with the classical mixing rule and a group contribution method for the binary interaction 
parameter "k_ij" by Jaubert and Mutelet.




exemplary input into Volume_SRK(T, P, Tc, Pc, w, R, mol_fractions, Components):



T=100 #K
P= 1000 # Pa
Tc= np.asarray([Tc_first_component, Tc_second_component, ....]) #find in NIST repository (or somewhere else)
Pc= np.asarray([Pc_first_component, Pc_second_component, ....]) #find in NIST repository (or somewhere else)
w = np.asarray([w_first_component, w_second_component, ....]) #find in NIST repository (or somewhere else)
z_example=[0.5,0.475,0.016,0.0035,0.0015,0.0015,0.0025] # 50%NG (average composition "Ingo_2022") 50%H2
Components_example=[[(6,1)],[(2,1)],[(3,1)],[(0,2),(1,1)],[(0,2),(1,2)],[(4,1)],[(5,1)]] #example for NG-H2 mixture

Volume_PR(T, P, Tc, Pc, w, C_R(), z_example, Components_example)
