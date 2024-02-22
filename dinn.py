import torch
# from torch.autograd import grad
import torch.nn as nn
# from numpy import genfromtxt
# import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import r2_score

# Define class
class DINN(nn.Module):

    def __init__(self,true_stuff, ranges, t,
                 R_e_data, R_t_data, R_r_data, R_p_data, P_e_data, P_t_data, P_r_data, P_p_data, M_a_data, M_b_data
                 ): #[t,S,I,D,R]
        super(DINN, self).__init__()

        self.ranges = ranges
        self.true_stuff=true_stuff
        
        #for the time steps, we need to convert them to a tensor, a float, and eventually to reshape it so it can be used as a batch
        self.t = torch.tensor(t, requires_grad=True)
        self.t_float = self.t.float()
        self.t_batch = torch.reshape(self.t_float, (len(self.t),1)) #reshape for batch 

        #for the compartments we just need to convert them into tensors
        self.R_e = torch.tensor(R_e_data)
        self.R_t = torch.tensor(R_t_data)
        self.R_r = torch.tensor(R_r_data)
        self.R_p = torch.tensor(R_p_data)
        self.P_e = torch.tensor(P_e_data)
        self.P_t = torch.tensor(P_t_data)
        self.P_r = torch.tensor(P_r_data)
        self.P_p = torch.tensor(P_p_data)
        self.M_a = torch.tensor(M_a_data)
        self.M_b = torch.tensor(M_b_data)

        self.losses = [] # here I saved the model's losses per epoch

        #setting the parameters
        self.k_tr_tilda = torch.nn.Parameter(torch.rand(1, requires_grad=True))
        self.K_pol_tilda = torch.nn.Parameter(torch.rand(1, requires_grad=True))
        self.k_rdeg_tilda = torch.nn.Parameter(torch.rand(1, requires_grad=True))
        self.k_tl_tilda = torch.nn.Parameter(torch.rand(1, requires_grad=True))
        self.K_rib_tilda = torch.nn.Parameter(torch.rand(1, requires_grad=True))
        self.k_exc_tilda = torch.nn.Parameter(torch.rand(1, requires_grad=True))
        self.K_T_rep_tilda = torch.nn.Parameter(torch.rand(1, requires_grad=True))
        self.k_tp_tilda = torch.nn.Parameter(torch.rand(1, requires_grad=True))
        self.k_cat_tilda = torch.nn.Parameter(torch.rand(1, requires_grad=True))
        self.k_M_tilda = torch.nn.Parameter(torch.rand(1, requires_grad=True))

        #find values for normalization
        
        self.R_e_max = max(self.R_e)
        self.R_t_max = max(self.R_t)
        self.R_r_max = max(self.R_r)
        self.R_p_max = max(self.R_p)
        self.P_e_max = max(self.P_e)
        self.P_t_max = max(self.P_t)
        self.P_r_max = max(self.P_r)
        self.P_p_max = max(self.P_p)
        self.M_a_max = max(self.M_a)
        self.M_b_max = max(self.M_b)

        self.R_e_min = min(self.R_e)
        self.R_t_min = min(self.R_t)
        self.R_r_min = min(self.R_r)
        self.R_p_min = min(self.R_p)
        self.P_e_min = min(self.P_e)
        self.P_t_min = min(self.P_t)
        self.P_r_min = min(self.P_r)
        self.P_p_min = min(self.P_p)
        self.M_a_min = min(self.M_a)
        self.M_b_min = min(self.M_b)

        #normalize
        self.R_e_hat = (self.R_e - self.R_e_min) / (self.R_e_max - self.R_e_min)
        self.R_t_hat = (self.R_t - self.R_t_min) / (self.R_t_max - self.R_t_min)
        self.R_r_hat = (self.R_r - self.R_r_min) / (self.R_r_max - self.R_r_min)
        self.R_p_hat = (self.R_p - self.R_p_min) / (self.R_p_max - self.R_p_min)
        self.P_e_hat = (self.P_e - self.P_e_min) / (self.P_e_max - self.P_e_min)
        self.P_t_hat = (self.P_t - self.P_t_min) / (self.P_t_max - self.P_t_min)
        self.P_r_hat = (self.P_r - self.P_r_min) / (self.P_r_max - self.P_r_min)
        self.P_p_hat = (self.P_p - self.P_p_min) / (self.P_p_max - self.P_p_min)
        self.M_a_hat = (self.M_a - self.M_a_min) / (self.M_a_max - self.M_a_min)
        self.M_b_hat = (self.M_b - self.M_b_min) / (self.M_b_max - self.M_b_min)


        #matrices (x4 for S,I,D,R) for the gradients
        self.m1 = torch.zeros((len(self.t), 10)); self.m1[:, 0] = 1
        self.m2 = torch.zeros((len(self.t), 10)); self.m2[:, 1] = 1
        self.m3 = torch.zeros((len(self.t), 10)); self.m3[:, 2] = 1
        self.m4 = torch.zeros((len(self.t), 10)); self.m4[:, 3] = 1
        self.m5 = torch.zeros((len(self.t), 10)); self.m5[:, 4] = 1
        self.m6 = torch.zeros((len(self.t), 10)); self.m6[:, 5] = 1
        self.m7 = torch.zeros((len(self.t), 10)); self.m7[:, 6] = 1
        self.m8 = torch.zeros((len(self.t), 10)); self.m8[:, 7] = 1
        self.m9 = torch.zeros((len(self.t), 10)); self.m9[:, 8] = 1
        self.m10 = torch.zeros((len(self.t), 10)); self.m10[:, 9] = 1
        
        #NN
        self.net_sidr = self.Net_sidr()
        self.params = list(self.net_sidr.parameters())
        self.params.extend(list([
            self.k_tr_tilda,
            self.K_pol_tilda,
            self.k_rdeg_tilda,
            self.k_tl_tilda,
            self.K_rib_tilda,
            self.k_exc_tilda,
            self.K_T_rep_tilda,
            self.k_tp_tilda,
            self.k_cat_tilda,
            self.k_M_tilda
            ]))


    @property
    def k_tr(self):
        range = self.ranges[0]
        return (torch.tanh(self.k_tr_tilda) + 1)/2 * (range[1] - range[0]) + range[0] #* 0.1 + 0.2
    @property
    def K_pol(self):
        range = self.ranges[1]
        return (torch.tanh(self.K_pol_tilda) + 1) / 2 * (range[1] - range[0]) + range[0] #* 0.01 + 0.05
    @property
    def k_rdeg(self):
        range = self.ranges[2]
        return (torch.tanh(self.k_rdeg_tilda) + 1) / 2 * (range[1] - range[0]) + range[0] #* 0.01 + 0.03
    @property
    def k_tl(self):
        range = self.ranges[3]
        return (torch.tanh(self.k_tl_tilda) + 1) / 2 * (range[1] - range[0]) + range[0] #* 0.01 + 0.03
    @property
    def K_rib(self):
        range = self.ranges[4]
        return (torch.tanh(self.K_rib_tilda) + 1) / 2 * (range[1] - range[0]) + range[0] #* 0.01 + 0.03
    @property
    def k_exc(self):
        range = self.ranges[5]
        return (torch.tanh(self.k_exc_tilda) + 1) / 2 * (range[1] - range[0]) + range[0] #* 0.1 + 0.2
    @property
    def K_T_rep(self):
        range = self.ranges[6]
        return (torch.tanh(self.K_T_rep_tilda) + 1) / 2 * (range[1] - range[0]) + range[0] #* 0.01 + 0.05
    @property
    def k_tp(self):
        range = self.ranges[7]
        return (torch.tanh(self.k_tp_tilda) + 1) / 2 * (range[1] - range[0]) + range[0] #* 0.01 + 0.03
    @property
    def k_cat(self):
        range = self.ranges[8]
        return (torch.tanh(self.k_cat_tilda) + 1) / 2 * (range[1] - range[0]) + range[0] #* 0.01 + 0.03
    @property
    def k_M(self):
        range = self.ranges[9]
        return (torch.tanh(self.k_M_tilda) + 1) / 2 * (range[1] - range[0]) + range[0] #* 0.01 + 0.03
    

    class Net_sidr(nn.Module): # input = [[t1], [t2]...[t100]] -- that is, a batch of timesteps 
        def __init__(self):
            super(DINN.Net_sidr, self).__init__()

            self.fc1=nn.Linear(1, 20) #takes 100 t's
            self.fc2=nn.Linear(20, 20)
            self.fc3=nn.Linear(20, 20)
            self.fc4=nn.Linear(20, 20)
            self.fc5=nn.Linear(20, 20)
            self.fc6=nn.Linear(20, 20)
            self.fc7=nn.Linear(20, 20)
            self.fc8=nn.Linear(20, 20)
            self.out=nn.Linear(20, 10) #outputs S, I, D, R (100 S, 100 I, 100 D, 100 R --- since we have a batch of 100 timesteps)

        def forward(self, t_batch):
            sidr=F.relu(self.fc1(t_batch))
            sidr=F.relu(self.fc2(sidr))
            sidr=F.relu(self.fc3(sidr))
            sidr=F.relu(self.fc4(sidr))
            sidr=F.relu(self.fc5(sidr))
            sidr=F.relu(self.fc6(sidr))
            sidr=F.relu(self.fc7(sidr))
            sidr=F.relu(self.fc8(sidr))
            sidr=self.out(sidr)
            return sidr

    def net_f(self, t_batch):
            
            #pass the timesteps batch to the neural network
            sidr_hat = self.net_sidr(t_batch)
            
            #organize S,I,D,R from the neural network's output -- note that these are normalized values -- hence the "hat" part
            R_e_hat = sidr_hat[:,0]
            R_t_hat = sidr_hat[:,1]
            R_r_hat = sidr_hat[:,2]
            R_p_hat = sidr_hat[:,3]
            P_e_hat = sidr_hat[:,4]
            P_t_hat = sidr_hat[:,5]
            P_r_hat = sidr_hat[:,6]
            P_p_hat = sidr_hat[:,7]
            M_a_hat = sidr_hat[:,8]
            M_b_hat = sidr_hat[:,9]

            #S_t
            sidr_hat.backward(self.m1, retain_graph=True)
            R_e_hat_t = self.t.grad.clone()
            self.t.grad.zero_()

            #I_t
            sidr_hat.backward(self.m2, retain_graph=True)
            R_t_hat_t = self.t.grad.clone()
            self.t.grad.zero_()

            #D_t
            sidr_hat.backward(self.m3, retain_graph=True)
            R_r_hat_t = self.t.grad.clone()
            self.t.grad.zero_()

            #R_t
            sidr_hat.backward(self.m4, retain_graph=True)
            R_p_hat_t = self.t.grad.clone()
            self.t.grad.zero_() 

            sidr_hat.backward(self.m5, retain_graph=True)
            P_e_hat_t = self.t.grad.clone()
            self.t.grad.zero_() 

            sidr_hat.backward(self.m6, retain_graph=True)
            P_t_hat_t = self.t.grad.clone()
            self.t.grad.zero_() 

            sidr_hat.backward(self.m7, retain_graph=True)
            P_r_hat_t = self.t.grad.clone()
            self.t.grad.zero_() 

            sidr_hat.backward(self.m8, retain_graph=True)
            P_p_hat_t = self.t.grad.clone()
            self.t.grad.zero_() 

            sidr_hat.backward(self.m9, retain_graph=True)
            M_a_hat_t = self.t.grad.clone()
            self.t.grad.zero_() 

            sidr_hat.backward(self.m10, retain_graph=True)
            M_b_hat_t = self.t.grad.clone()
            self.t.grad.zero_() 

            #unnormalize
            R_e  = self.R_e_min + (self.R_e_max - self.R_e_min) * R_e_hat
            R_t  = self.R_t_min + (self.R_t_max - self.R_t_min) * R_t_hat
            R_r  = self.R_r_min + (self.R_r_max - self.R_r_min) * R_r_hat
            R_p  = self.R_p_min + (self.R_p_max - self.R_p_min) * R_p_hat
            P_e  = self.P_e_min + (self.P_e_max - self.P_e_min) * P_e_hat
            P_t  = self.P_t_min + (self.P_t_max - self.P_t_min) * P_t_hat
            P_r  = self.P_r_min + (self.P_r_max - self.P_r_min) * P_r_hat
            P_p  = self.P_p_min + (self.P_p_max - self.P_p_min) * P_p_hat
            M_a  = self.M_a_min + (self.M_a_max - self.M_a_min) * M_a_hat
            M_b  = self.M_b_min + (self.M_b_max - self.M_b_min) * M_b_hat

            # f1_hat = S_hat_t - ODE(with self.param but normal variable names) / (self.S_max-self.S_mmin)

            f1_hat  = R_e_hat_t -  ( self.k_tr * ( 1/ ( 1 + (self.K_pol/P_p)*(1+P_t/self.K_T_rep) ) ) - R_e ) / (self.R_e_max - self.R_e_min)
            f2_hat  = R_t_hat_t -  ( self.k_tr * ( 1/ ( 1 + (self.K_pol/P_p) ) ) - self.k_rdeg*R_t )          / (self.R_t_max - self.R_t_min)
            f3_hat  = R_r_hat_t -  ( self.k_tr * ( 1/ ( 1 + (self.K_pol/P_p) ) ) - self.k_rdeg*R_r )          / (self.R_r_max - self.R_r_min)
            f4_hat  = R_p_hat_t -  ( self.k_tr * ( 1/ ( 1 + (self.K_pol/P_p) ) ) - self.k_rdeg*R_p )          / (self.R_p_max - self.R_p_min)
            f5_hat  = P_e_hat_t -  ( self.k_tl * ( 1/ ( 1 + (self.K_rib/P_r) ) ) * R_e )                 / (self.P_e_max - self.P_e_min)
            f6_hat  = P_t_hat_t -  ( self.k_tl * ( 1/ ( 1 + (self.K_rib/P_r) ) ) * R_t )                 / (self.P_t_max - self.P_t_min)
            f7_hat  = P_r_hat_t -  ( self.k_tl * ( 1/ ( 1 + (self.K_rib/P_r) ) ) * R_r )                 / (self.P_r_max - self.P_r_min)
            f8_hat  = P_p_hat_t -  ( self.k_tl * ( 1/ ( 1 + (self.K_rib/P_r) ) ) * R_p )                 / (self.P_p_max - self.P_p_min)
            f9_hat  = M_a_hat_t -  ( self.k_tp - ( (self.k_cat*P_e*M_a)*(self.k_M+M_a+P_e) ) - self.k_exc * M_a )  / (self.M_a_max - self.M_a_min)
            f10_hat = M_b_hat_t -  ( ( (self.k_cat*P_e*M_a)*(self.k_M+M_a+P_e) ) - self.k_exc * M_b )         / (self.M_b_max - self.M_b_min)   

            return f1_hat, f2_hat, f3_hat, f4_hat, f5_hat, f6_hat, f7_hat, f8_hat, f9_hat, f10_hat, R_e_hat, R_t_hat, R_r_hat, R_p_hat, P_e_hat, P_t_hat, P_r_hat, P_p_hat, M_a_hat, M_b_hat

    

    def train(self, n_epochs):
        # train
        print('\nstarting training...\n')
        
        # initialize r2_store
        r2_store = []

        for epoch in range(n_epochs):
            # lists to hold the output (maintain only the final epoch)
            R_e_pred_list = []
            R_t_pred_list = []
            R_r_pred_list = []
            R_p_pred_list = []
            P_e_pred_list = []
            P_t_pred_list = []
            P_r_pred_list = []
            P_p_pred_list = []
            M_a_pred_list = []
            M_b_pred_list = []

            # we pass the timesteps batch into net_f
            f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, R_e_pred, R_t_pred, R_r_pred, R_p_pred, P_e_pred, P_t_pred, P_r_pred, P_p_pred, M_a_pred, M_b_pred = self.net_f(self.t_batch) # net_f outputs f1_hat, f2_hat, f3_hat, f4_hat, S_hat, I_hat, D_hat, R_hat
            
            self.optimizer.zero_grad() #zero grad
            
            #append the values to plot later (note that we unnormalize them here for plotting)
            R_e_pred_list.append(self.R_e_min + (self.R_e_max - self.R_e_min) * R_e_pred)
            R_t_pred_list.append(self.R_t_min + (self.R_t_max - self.R_t_min) * R_t_pred)
            R_r_pred_list.append(self.R_r_min + (self.R_r_max - self.R_r_min) * R_r_pred)
            R_p_pred_list.append(self.R_p_min + (self.R_p_max - self.R_p_min) * R_p_pred)
            P_e_pred_list.append(self.P_e_min + (self.P_e_max - self.P_e_min) * P_e_pred)
            P_t_pred_list.append(self.P_t_min + (self.P_t_max - self.P_t_min) * P_t_pred)
            P_r_pred_list.append(self.P_r_min + (self.P_r_max - self.P_r_min) * P_r_pred)
            P_p_pred_list.append(self.P_p_min + (self.P_p_max - self.P_p_min) * P_p_pred)
            M_a_pred_list.append(self.M_a_min + (self.M_a_max - self.M_a_min) * M_a_pred)
            M_b_pred_list.append(self.M_b_min + (self.M_b_max - self.M_b_min) * M_b_pred)

            #calculate the loss --- MSE of the neural networks output and each compartment
            
            R_e_pred, R_t_pred, R_r_pred, R_p_pred, P_e_pred, P_t_pred, P_r_pred, P_p_pred, M_a_pred, M_b_pred

            f1, f2, f3, f4, f5, f6, f7, f8, f9, f10
            
            loss = (
                
                torch.mean(torch.square(self.R_e_hat-R_e_pred    ))+
                torch.mean(torch.square(self.R_t_hat-R_t_pred    ))+
                torch.mean(torch.square(self.R_r_hat-R_r_pred    ))+
                torch.mean(torch.square(self.R_p_hat-R_p_pred    ))+
                torch.mean(torch.square(self.P_e_hat-P_e_pred    ))+
                torch.mean(torch.square(self.P_t_hat-P_t_pred    ))+
                torch.mean(torch.square(self.P_r_hat-P_r_pred    ))+
                torch.mean(torch.square(self.P_p_hat-P_p_pred    ))+
                torch.mean(torch.square(self.M_a_hat-M_a_pred    ))+
                torch.mean(torch.square(self.M_b_hat-M_b_pred    ))+
                torch.mean(torch.square(f1))+
                torch.mean(torch.square(f2))+
                torch.mean(torch.square(f3))+
                torch.mean(torch.square(f4))+
                torch.mean(torch.square(f5))+
                torch.mean(torch.square(f6))+
                torch.mean(torch.square(f7))+
                torch.mean(torch.square(f8))+
                torch.mean(torch.square(f9))+
                torch.mean(torch.square(f10))
                    ) 

            loss.backward()
            self.optimizer.step()
            self.scheduler.step() 

            # append the loss value (we call "loss.item()" because we just want the value of the loss and not the entire computational graph)
            self.losses.append(loss.item())

            # compute r2
            learned_stuff = [self.k_tr.item(),
                             self.K_pol.item(),
                             self.k_rdeg.item(),
                             self.k_tl.item(),
                             self.K_rib.item(),
                             self.k_exc.item(),
                             self.K_T_rep.item(),
                             self.k_tp.item(),
                             self.k_cat.item(),
                             self.k_M.item()]
            
            r2_store.append(r2_score(self.true_stuff, learned_stuff))
            # r2_store.append(0)

            if epoch % 1000 == 0:          
                print('\nEpoch ', epoch)
                print('#################################')                

        return r2_store, R_e_pred_list, R_t_pred_list, R_r_pred_list, R_p_pred_list, P_e_pred_list, P_t_pred_list, P_r_pred_list, P_p_pred_list, M_a_pred_list, M_b_pred_list