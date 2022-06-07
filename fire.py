import torch,os,pickle,math,random
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from numba import njit
from mpi4py import MPI
from MCMC import MC_sequence
#torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled = False

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

L=12 #define the side length of the square lattice
K=5 #kernel size of the first convolution layer
site_number=int(L**2)
Case="tJ"#which model? "J1J2" or "tJ"

# build neural network
from network_structures import Net_J1J2,Net_tJ
def get_n_params(model):#obtain network's total parameter number
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

if rank == 0:
    if Case=="J1J2":
        net=Net_J1J2().double()
    elif Case=="tJ":
        net = Net_tJ().double()
    print("Total parameter number is {}".format(get_n_params(net)))
    params_amp_nn=list(net.parameters())[:]
    param_number_list=[len(params_amp_nn[i].reshape(-1)) for i in range(len(params_amp_nn))]
    # load model checkpoints and initial spin lattice
    with open('{}/model_chk'.format(Case),"rb") as f:
        weights=pickle.load(f)
    with open('{}/spin_lattice'.format(Case), 'rb') as f:
        spin_lattice = pickle.load(f)
    # assign pre-trained weights
    for i in range(len(weights)):
        params_amp_nn[i].data=torch.from_numpy(weights[i])
    send_to_process = [[12, net, spin_lattice] for _ in range(size)]
else:
    send_to_process = None

on_each_process=comm.scatter(send_to_process, root=0)
#MC_sequence(Nsweep,net,spin_lattice,model_type:str,rank)
Es_list,spin_lattice=MC_sequence(945,on_each_process[1],on_each_process[2],Case,rank)
send_to_root=comm.gather(np.mean(Es_list),root=0)
if rank == 0:
    Energy=np.mean(send_to_root) # average energy from each MPI rank on rank0
    with open("energy","a") as f: # write down energy value
        f.write("Case is : {}\tEnergy per site : {}\n".format(Case,Energy))
