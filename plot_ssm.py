import numpy as np
import os, sys, librosa
from matplotlib import pyplot as plt
import IPython.display as ipd
from numpy import genfromtxt


# Parameters
file_path = 'D:\dev\Mestrado\Onlydata-features-4.csv'
savepathssm = 'D:\\TCC\\ImagemTeste\\Normalizadas\\ROIs_normalizadas\\grave\\Método proposto\\grave\\SSM\\'


#Make save dir if not exists
if not os.path.exists(savepathssm):
    os.makedirs(savepathssm)


# Get data from csv
csv_feat = genfromtxt(file_path, delimiter=',')
n_img = len(csv_feat)
#Define data range
csv_feat = csv_feat[0:n_img, 0:300]

for i in range (n_img):
  b = csv_feat[i, 0:300]
  min_val = np.min(b)
  max_val = np.min(b)
  scaled_data = (b - min_val) / (max_val - min_val)*255
  X = scaled_data
  X = X.reshape(3, 100)
  S = np.dot(np.transpose(X), X)
  # save_img
  plt.imsave(savepathssm+'SSM'+str(i+1)+'.png', S, origin='lower')




