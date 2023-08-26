# surfloss.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from matplotlib import cm

cross_df0=pd.DataFrame()
cross_df1=pd.DataFrame()
compact_df0=pd.DataFrame()
compact_df1=pd.DataFrame()

cross_df0r=pd.DataFrame()
compact_df0r=pd.DataFrame()
verbose=0
count=0
for n_in in [3,5]:
  for n_out in [3,5]:
    for nmp in [1,2]:
      for unet in [0,1]:
        for rot in [0,1]: # 
          for Loss in [1,2]: #
            # Path will be updated when interacts with the rotation channels
            # e.g. cafeMINI_iso_convlstm-batch2_lr1e-4_nin3_nout3_noise0_lossL2_nmp6_nhid64_ker3_RNN1_dx1_unet1_wd1e-3_rot1
            path='/usr/WS2/jaman1/temp/NPS/experiment/cafeMINI_iso_RNNpredrnn_v1-batch2_lr1e-4_nin'+str(n_in)+'_nout'+str(n_out)+'_noise0_lossL'+str(Loss)+'_nmp'+str(nmp)+'_nhid64_ker3_RNN1_dx1_unet'+str(unet)+'_wd1e-3_rot'+str(rot)+'/log'
            #path='/content/drive/MyDrive/LLNL/experiment/cafeMINI_iso_convlstm-batch2_lr1e-4_nin3_nout3_noise0_lossL2_nmp6_nhid64_ker3_RNN1_dx1_unet1_wd1e-3_rot1/log'
            #print('Reading : ',path)
            if os.path.exists(path):
              count=count+1
              if verbose==1:
                print('Reading : ',path)
              temp=pd.read_csv(path, sep='\r\n', header=None, engine='python')
              temp['unet']=unet
              temp['rot']=rot
              temp['nmp']=nmp
              temp['Loss']=Loss
              temp['n_in']=n_in
              temp['n_out']=n_out

              if rot==1:
                temp0_r=temp[temp[0].str.contains('valid per time/channel/seq: ')]
                temp0_r=pd.concat([temp0_r[0].str.split('\s+', expand=True),temp0_r.loc[:,temp0_r.columns!=0]], axis=1)
                compact_df0r=pd.concat([compact_df0r, temp0_r.iloc[[-1]]])
                cross_df0r=pd.concat([cross_df0r, temp0_r])
              else:
                temp0_nr=temp[temp[0].str.contains('valid per time/channel/seq: ')]
                temp0_nr=pd.concat([temp0_nr[0].str.split('\s+', expand=True),temp0_nr.loc[:,temp0_nr.columns!=0]], axis=1)
                compact_df0=pd.concat([compact_df0, temp0_nr.iloc[[-1]]])
                cross_df0=pd.concat([cross_df0, temp0_nr])
              
              temp1=temp[temp[0].str.contains(' Train_loss: ')]
              temp1=pd.concat([temp1[0].str.split('\s+', expand=True),temp1.loc[:,temp1.columns!=0]], axis=1)
              compact_df1=pd.concat([compact_df1, temp1.iloc[[-1]]])
              cross_df1=pd.concat([cross_df1, temp1])
if verbose==1:
  print('File processed: ', count,'\n') 

X=np.array([1,2]) # X axis /rot
Y=np.array([0,1]) # Y axis /unet
X,Y=np.meshgrid(X,Y)
Z=np.zeros(np.shape(X))
for i in np.arange(0,np.shape(Y)[0]):
  for j in np.arange(0,np.shape(X)[1]):

    # Lowest MSE config (partial)
    unet=Y[i,j]
    nmp=X[i,j]
    #nmp=6
    rot=1
    Loss=2
    n_in=3
    n_out=3
    surf_df0=compact_df1
    surf_df0=surf_df0[(surf_df0['unet']==unet)]
    surf_df0=surf_df0[(surf_df0['rot']==rot)] 
    surf_df0=surf_df0[(surf_df0['nmp']==nmp)] 
    surf_df0=surf_df0[(surf_df0['Loss']==Loss)] 
    surf_df0=surf_df0[(surf_df0['n_in']==n_in)] 
    surf_df0=surf_df0[(surf_df0['n_out']==n_out)]
    item=6 # MSE

    Z[i,j]=pd.to_numeric(surf_df0[item].values[0])

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,5))
surf = ax.plot_surface(X, Y, Z, cmap=cm.hot,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(np.min(Z), np.max(Z))
ax.set_zlabel('Loss')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.xlabel('nmp')
plt.ylabel('unet')

plt.show()
