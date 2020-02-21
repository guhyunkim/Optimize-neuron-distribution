import numpy as np
import os
import itertools
import torch
import pandas as pd

# calculate the number of neuron to core connections
def N_NC(T_input,X_input):
    P=((T_input.matmul(X_input))>0).float()
    return torch.sum(P,dim=(1,2))
# constraint 1: no duplication of a nueron over multiplc cores
def c1(X_input):
    return torch.pow(torch.sum(X_input,dim=2)-1,2)
# constraint 2: limit number of neurons in a core
def c2(X_input,N_input):
    values=torch.max(torch.sum(X_input,dim=1)-N_input,torch.zeros((X_input.shape[0],X_input.shape[2])).cuda())
    return torch.pow(values,2)
# constraint 3: limit number of synapses in a core
def c3(X_input,T_post_input,S_input):
    synapse=torch.sum(T_post_input*X_input.transpose(1,2),dim=(2))
    values=torch.max(synapse-S_input,torch.zeros((X_input.shape[0],X_input.shape[2])).cuda())
    return torch.pow(values,2)

# get directory
directory=os.getcwd()
save=np.zeros((1,3))# save N_NC and lagrangian

Q=1360 # number of neuron
M=16 # number of cores
N=128 # maximum number of neurons in a core
S=32768 # maximum number of synapses in a core

T=torch.zeros((Q,Q)).cuda();# topolgy of SNN
for i in range(1024):
    T[i,1024:1280]=1
for i in range(1024,1280):
    T[i,1280:1344]=1
for i in range(1280,1344):
    T[i,1344:1360]=1

T_post=torch.sum(T,dim=0)# number of fan-in synapses for each neuron

X=torch.zeros((1,Q,M)).cuda() #neuron distribution matrix

n_nb=Q*M # number of neighbors
one_location=np.array(list(itertools.combinations(range(n_nb),1)))
dim1=one_location[:,0]//M
dim2=one_location[:,0]%M
dim3=np.array(range(n_nb-1,-1,-1))
indices=np.vstack((dim3,dim1,dim2)).astype(int) # indices for neighbors
alphaX=torch.zeros((n_nb+1,Q,M)).cuda() # set of neighbors and X itself

lambda1, lambda2, lambda3 =torch.ones((Q)).cuda(), torch.ones((M)).cuda(), torch.ones((M)).cuda() # lambdas for constraints
eta=0.1 # update rate of lambda

i=0
while True:
    alphaX.data=X.expand(n_nb,-1,-1).clone()
    alphaX[tuple(indices)]=1-alphaX[tuple(indices)]
    alphaX.data=torch.cat((X,alphaX),dim=0).cuda() # get alphaX
    N_NC_X, cs1, cs2, cs3= N_NC(T,alphaX), c1(alphaX), c2(alphaX,N), c3(alphaX,T_post,S) # calculate N_NC and constraints
    L_d=N_NC_X+torch.sum(torch.mul(lambda1,cs1),dim=1)+torch.sum(torch.mul(lambda2,cs2),dim=1)+torch.sum(torch.mul(lambda3,cs3),dim=1)# get lagrangian

    save=np.append(save,np.array([[i,N_NC_X.cpu().numpy()[0],L_d.cpu().numpy()[0]]]),axis=0)

    if i%1000==0:
        print(i)
        save_data=pd.DataFrame(save)
        save_data.to_csv(directory+"/save_large.txt",index=False, header=False, sep=' ')

    if L_d[0]==N_NC_X[0]:
         break
    # find the element having minimum lagrangian
    min_index=torch.argmin(L_d).item()
    X.data=alphaX[min_index,:,:].unsqueeze(0)

    # update lambda
    lambda1+=eta*cs1[0,:]
    lambda2+=eta*cs2[0,:]
    lambda3+=eta*cs3[0,:]

    i+=1
save_data=pd.DataFrame(save)
save_data.to_csv(directory+"/save_large.txt",index=False, header=False,sep=' ')
X_data=pd.DataFrame(X.cpu().numpy().reshape(Q,M))
X_data.to_csv(directory+"/X_large.txt",index=False, header=False,sep=' ')    

print(np.sum(X.cpu().numpy()[0,0:1024,:],axis=0))
print(np.sum(X.cpu().numpy()[0,1024:1280,:],axis=0))
print(np.sum(X.cpu().numpy()[0,1280:1344,:],axis=0))
print(np.sum(X.cpu().numpy()[0,1344:1360,:],axis=0))
                                                
