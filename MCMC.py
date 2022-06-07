import random,pickle
from numba import njit
import torch
from torch.autograd import Variable
import numpy as np

L=12 #square lattice side length is L
site_number=int(L**2)
K=5 #filter side length of first convolution layer K

#@njit
def make_PBC_spin_lattice(spin_lattice,model_type:str):
    batchsize = len(spin_lattice)
    PBC=np.zeros(shape=(batchsize,L+K-1,L+K-1))
    k=int((K-1)/2)
    PBC[:,k:L+k,k:L+k]=spin_lattice[:,:,:]
    PBC[:,:k,:k]=spin_lattice[:,-k:,-k:]
    PBC[:,:k, -k:]=spin_lattice[:,-k:, :k]
    PBC[:,-k:, :k]=spin_lattice[:,:k, -k:]
    PBC[:,-k:, -k:]=spin_lattice[:,:k, :k]
    PBC[:,:k, k:L+k]=spin_lattice[:,-k:, :]
    PBC[:,-k:, k:L+k]=spin_lattice[:,:k, :]
    PBC[:,k:L+k, :k]=spin_lattice[:,:, -k:]
    PBC[:,k:L+k, -k:]=spin_lattice[:,:, :k]
    if model_type=="tJ":
        return PBC+1 #[batchsize,L+K-1,L+K-1]
    elif model_type=="J1J2":
        return PBC.reshape(batchsize, 1, L + K - 1, L + K - 1)  # [batchsize,1,L+K-1,L+K-1]

@njit #enforcing sign rule for J1J2 model
def sign_rule(spin_lattice):
    s=0.5*(spin_lattice+1)
    s1=s[::2,::2]
    s2=s[1::2,1::2]
    M=int(np.sum(s1)+np.sum(s2))
    return (-1)**M

#enforcing C4 symmetry for J1J2 model
def rot90(s):
    s=s.reshape((len(s),1,L+K-1,L+K-1))
    s1=np.array(s)
    s2=np.rot90(s1,1,axes=(2,3)).copy()
    s3=np.rot90(s1,2,axes=(2,3)).copy()
    s4=np.rot90(s1,3,axes=(2,3)).copy()
    return s1,s2,s3,s4

#sequencing of Fermions for tJ model
label_sequence=np.arange(0,site_number).reshape(L,L)
for i in range(L):
    if i%2==1:
        label_sequence[i]=label_sequence[i][::-1]

label_flattern=label_sequence.reshape(-1)
def calculate_sign(spin_lattice,site_1,site_2):
    s_flattern=np.abs(spin_lattice).reshape(-1)
    site_1_label=label_sequence[site_1[0],site_1[1]]
    site_2_label=label_sequence[site_2[0],site_2[1]]
    prod_list=[]
    for label in range(min([site_1_label,site_2_label])+1,max([site_1_label,site_2_label])):
        prod_list.append(s_flattern[np.where(label_flattern==label)])
    return (-1)**np.sum(prod_list)

def WS_CALCULATE(spin_lattice, net, model_type:str):
    with torch.no_grad():
        if model_type=="J1J2":
            s1 = make_PBC_spin_lattice(spin_lattice.reshape(1, L, L),model_type)
            s1,s2,s3,s4=rot90(s1)
            ws = net(torch.from_numpy(s1),
                     torch.from_numpy(s2),
                     torch.from_numpy(s3),
                     torch.from_numpy(s4)).squeeze()*sign_rule(spin_lattice)
            return ws.data.numpy()
        elif model_type=="tJ":
            s1 = make_PBC_spin_lattice(spin_lattice.reshape(1, L, L),model_type)
            spin_input_1 = torch.LongTensor(s1.reshape(1, L + K - 1, L + K - 1))
            ws = net(spin_input_1).squeeze()
            return ws.data.numpy()

def calculate_sprime_tJ(spin_lattice,t,J):
    E_s = 0
    propose_batch=[]
    for i in range(L):
        for j in range(L):
            site_1 = [i, j]
            if i + 1 < L:
                site_2 = [i + 1, j]
                spin_1 = spin_lattice[site_1[0], site_1[1]]
                spin_2 = spin_lattice[site_2[0], site_2[1]]
                if spin_1*spin_2==0 and spin_1+spin_2!=0:
                    sign = calculate_sign(spin_lattice, site_1, site_2)
                    propose_batch.append([site_1,spin_2,site_2,spin_1,sign*t])
                elif spin_1*spin_2==-1:
                    E_s-=J/2
                    propose_batch.append([site_1,spin_2,site_2,spin_1,J/2])
                else:
                    pass
            if j + 1 < L:
                site_2 = [i, j + 1]
                spin_1 = spin_lattice[site_1[0], site_1[1]]
                spin_2 = spin_lattice[site_2[0], site_2[1]]
                if spin_1*spin_2==0 and spin_1+spin_2!=0:
                    sign = 1#calculate_sign(spin_lattice, site_1, site_2)
                    propose_batch.append([site_1,spin_2,site_2,spin_1,sign*t])
                elif spin_1*spin_2==-1:
                    E_s-=J/2
                    propose_batch.append([site_1,spin_2,site_2,spin_1,J/2])
                else:
                    pass
    #PBC boundary
    for i in range(L):
        site_1=[i,0]
        site_2=[i,L-1]
        spin_1 = spin_lattice[site_1[0], site_1[1]]
        spin_2 = spin_lattice[site_2[0], site_2[1]]
        if spin_1*spin_2==0 and spin_1+spin_2!=0:
            sign = calculate_sign(spin_lattice, site_1, site_2)
            propose_batch.append([site_1,spin_2,site_2,spin_1,sign*t])
        elif spin_1*spin_2==-1:
            E_s-=J/2
            propose_batch.append([site_1,spin_2,site_2,spin_1,J/2])
        else:
            pass
    for j in range(L):
        site_1=[0,j]
        site_2=[L-1,j]
        spin_1 = spin_lattice[site_1[0], site_1[1]]
        spin_2 = spin_lattice[site_2[0], site_2[1]]
        if spin_1*spin_2==0 and spin_1+spin_2!=0:
            sign = calculate_sign(spin_lattice, site_1, site_2)
            propose_batch.append([site_1,spin_2,site_2,spin_1,sign*t])
        elif spin_1*spin_2==-1:
            E_s-=J/2
            propose_batch.append([site_1,spin_2,site_2,spin_1,J/2])
        else:
            pass
    return E_s,propose_batch

def calculate_sprime_J1J2(spin_lattice,J2):
    E_s,s1_length = 0,0
    propose_batch=[]
    # J1 interactions
    for i in range(L):
        for j in range(L):
            site_1 = [i, j]
            if i + 1 < L:
                site_2 = [i + 1, j]
            else:
                site_2 = [0, j]
            spin_1 = spin_lattice[site_1[0], site_1[1]]
            spin_2 = spin_lattice[site_2[0], site_2[1]]
            if spin_1 * spin_2 == -1:
                propose_batch.append([site_1,spin_2,site_2,spin_1])
                s1_length+=1
                E_s -= 1
            else:
                E_s += 1
            if j + 1 < L:
                site_2 = [i, j + 1]
            else:
                site_2 = [i, 0]
            spin_1 = spin_lattice[site_1[0], site_1[1]]
            spin_2 = spin_lattice[site_2[0], site_2[1]]
            if spin_1 * spin_2 == -1:
                propose_batch.append([site_1,spin_2,site_2,spin_1])
                s1_length += 1
                E_s -= 1
            else:
                E_s += 1
    # J2 interactions
    if J2!=0:
        for i in range(L):
            for j in range(L):
                site_1 = [i, j]
                if i + 1 < L and j + 1 < L:
                    site_2 = [i + 1, j + 1]
                elif i + 1 < L and j + 1 >= L:
                    site_2 = [i + 1, 0]
                elif i + 1 >= L and j + 1 < L:
                    site_2 = [0, j + 1]
                elif i + 1 >= L and j + 1 >= L:
                    site_2 = [0, 0]
                spin_1 = spin_lattice[site_1[0], site_1[1]]
                spin_2 = spin_lattice[site_2[0], site_2[1]]
                if spin_1 * spin_2 == -1:
                    propose_batch.append([site_1,spin_2,site_2,spin_1])
                    E_s -= J2
                else:
                    E_s += J2
                if i + 1 < L and j - 1 >= 0:
                    site_2 = [i + 1, j - 1]
                elif i + 1 < L and j - 1 < 0:
                    site_2 = [i + 1, -1]
                elif i + 1 >= L and j - 1 >= 0:
                    site_2 = [0, j - 1]
                elif i + 1 >= L and j - 1 < 0:
                    site_2 = [0, -1]
                spin_1 = spin_lattice[site_1[0], site_1[1]]
                spin_2 = spin_lattice[site_2[0], site_2[1]]
                if spin_1 * spin_2 == -1:
                    propose_batch.append([site_1,spin_2,site_2,spin_1])
                    E_s -= J2
                else:
                    E_s += J2
    return E_s,propose_batch,s1_length

def make_s_prime(spin_lattice,propose_batch):
    batchsize=len(propose_batch)
    s_prime_batch=np.zeros(shape=(batchsize,L,L)) #[batchsize,L,L]
    s_prime_batch[:,:,:]=spin_lattice[:,:]
    for i,propose in enumerate(propose_batch):
        site_1, spin_2, site_2, spin_1=propose[0],propose[1],propose[2],propose[3]
        s_prime_batch[i,site_1[0],site_1[1]]=spin_2
        s_prime_batch[i,site_2[0], site_2[1]]=spin_1
    return s_prime_batch #[batchsize,L,L]

def Energy_on_spin(ws, spin_lattice, net, model_type:str):
    if model_type == "J1J2":
        J1,J2 = 1,0.5
        E_s,propose_batch,s1_length = calculate_sprime_J1J2(spin_lattice, J2)
        # computing the total non-diagonal Es elements
        if len(propose_batch) != 0:
            with torch.no_grad():
                batchsize=len(propose_batch)
                s_prime_batch=make_s_prime(spin_lattice,propose_batch) #[batchsize,L,L]
                sign_batch = np.array([sign_rule(e) for e in s_prime_batch])
                s_prime_batch_PBC=make_PBC_spin_lattice(s_prime_batch,model_type) #[batchsize,1,L+K-1,L+K-1]
                s1,s2,s3,s4=rot90(s_prime_batch_PBC)
                ws_1_batch = net(torch.from_numpy(s1),
                                 torch.from_numpy(s2),
                                 torch.from_numpy(s3),
                                 torch.from_numpy(s4)
                                 ).squeeze()
                # calculate non-diagonal elements
                ws_1_batch = (ws_1_batch * torch.DoubleTensor(sign_batch)).data.numpy()/ws
                E_s += 2 * (np.sum(ws_1_batch[:s1_length]) + J2 * np.sum(ws_1_batch[s1_length:]))
        return E_s
    elif model_type=="tJ":
        t,J=-1,0.4
        E_s, propose_batch = calculate_sprime_tJ(spin_lattice, t, J)
        # computing the total non-diagonal Es elements
        if len(propose_batch) != 0:
            with torch.no_grad():
                batchsize = len(propose_batch)
                s_prime = make_s_prime(spin_lattice, propose_batch)  # [batchsize,L,L]
                s_prime_PBC = make_PBC_spin_lattice(s_prime,model_type)  # [batchsize,L+K-1,L+K-1]
                s1 = s_prime_PBC.reshape((batchsize, L + K - 1, L + K - 1))
                with torch.no_grad():
                    ws_1_batch = net(torch.LongTensor(s1)).squeeze()
                    sign_batch = [e[-1] for e in propose_batch]
                    ws_1_batch = ws_1_batch.data.numpy() / ws
                    ws_1_batch = ws_1_batch * np.array(sign_batch)
                    E_s += np.sum(ws_1_batch)
        return E_s

def MC_sequence(Nsweep,net,spin_lattice,model_type:str,rank):
    Es_list=[]
    sweep_count=0
    collected_samples=0
    fly_away_count=0
    initial_spin_lattice=np.copy(spin_lattice)
    ws=WS_CALCULATE(spin_lattice,net,model_type)
    while collected_samples<Nsweep:
        for i in range(L):
            for j in range(L):
                site_1=[i,j]
                if j+1<L:
                    site_2=[i,j+1]
                else:
                    site_2=[i,0]
                spin_1,spin_2=spin_lattice[site_1[0],site_1[1]],spin_lattice[site_2[0],site_2[1]]
                if spin_1!=spin_2:
                    #flip it
                    spin_lattice[site_1[0], site_1[1]], spin_lattice[site_2[0], site_2[1]]=spin_2,spin_1
                    ws_1=WS_CALCULATE(spin_lattice,net,model_type)
                    P=(ws_1/ws)**2
                    r=np.random.uniform(0,1)
                    if P>r:
                        ws=ws_1
                        pass
                    else:
                        #flip back
                        spin_lattice[site_1[0], site_1[1]], spin_lattice[site_2[0], site_2[1]] = spin_1, spin_2
        for j in range(L):
            for i in range(L):
                site_1=[i,j]
                if i+1<L:
                    site_2=[i+1,j]
                else:
                    site_2=[0,j]
                spin_1,spin_2=spin_lattice[site_1[0],site_1[1]],spin_lattice[site_2[0],site_2[1]]
                if spin_1!=spin_2:
                    #flip it
                    spin_lattice[site_1[0], site_1[1]], spin_lattice[site_2[0], site_2[1]]=spin_2,spin_1
                    ws_1=WS_CALCULATE(spin_lattice,net,model_type)
                    P=(ws_1/ws)**2
                    r=np.random.uniform(0,1)
                    if P>r:
                        ws=ws_1
                        pass
                    else:
                        #flip back
                        spin_lattice[site_1[0], site_1[1]], spin_lattice[site_2[0], site_2[1]] = spin_1, spin_2
        if model_type=="J1J2":
            ele1 = Energy_on_spin(ws, spin_lattice, net, model_type)
            Es = ele1 / (4 * site_number)
            if -0.6 < Es < -0.45:
                collected_samples += 1
                Es_list.append(Es)
                if rank == 0 and collected_samples % 10 == 0:
                    with open('MC_progress', 'a') as f:
                        f.write('{} collected on rank 0, Es is {}\n'.format(collected_samples,Es))
            else:
                fly_away_count += 1
                if fly_away_count > 3:
                    spin_lattice = initial_spin_lattice
                    ws = WS_CALCULATE(spin_lattice, net, model_type)
                    fly_away_count = 0
        elif model_type=="tJ":
            ele1 = Energy_on_spin(ws, spin_lattice, net, model_type)
            Es = ele1 / (site_number)
            if -15 < Es < 0:
                collected_samples += 1
                Es_list.append(Es)
                if rank == 0 and collected_samples % 10 == 0:
                    with open('MC_progress', 'a') as f:
                        f.write('{} collected on rank 0, Es is {}\n'.format(collected_samples,Es))
            else:
                fly_away_count += 1
                if fly_away_count > 10:
                    spin_lattice = initial_spin_lattice
                    ws = WS_CALCULATE(spin_lattice, net, model_type)
                    fly_away_count = 0
    return Es_list, spin_lattice
