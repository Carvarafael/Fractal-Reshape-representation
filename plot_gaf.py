""""
# Gramian Angular Field


A Gramian Angular Field is an image obtained from a time series, representing
some temporal correlation between each time point. Two methods are
available: Gramian Angular Summation Field and Gramian Angular Difference
Field. This example illustrates the transformation on the first sample of
the *GunPoint* dataset. Both images are plotted side by side to illustrate
the differences.
It is implemented as :class:`pyts.image.GramianAngularField`.
"""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from pyts.datasets import load_gunpoint
from numpy import genfromtxt
from pathlib import Path, PureWindowsPath
import os
import numpy as np


# Parameters
file_path = 'D:\dev\Mestrado\Onlydata-features-4.csv'
savepathgasf = 'D:\\TCC\\ImagemTeste\\Imagens_n_Segmentadas\\grave\\Fractal\\GASF\\'
savepathgadf = 'D:\\TCC\\ImagemTeste\\Imagens_n_Segmentadas\\grave\\Fractal\\GADF\\'

#Make save dir if not exists
if not os.path.exists(savepathgasf):
    os.makedirs(savepathgasf)
if not os.path.exists(savepathgadf):
    os.makedirs(savepathgadf)

n_img = 74

# Get data from csv
X = genfromtxt(file_path, delimiter=',')
#Define data range
a = X[0:n_img, 0:300]

for i in range (n_img):
  b = a[i, 0:300]
  X = b
  X = X.reshape(3, 100)
  # Transform the time series into Gramian Angular Fields
  gasf = GramianAngularField(method='summation')
  X_gasf = gasf.fit_transform(X)
  gadf = GramianAngularField(method='difference')
  X_gadf = gadf.fit_transform(X)

 
  plt.imsave(savepathgasf+'GRAYGASFMink'+str(i+1)+'.png', X_gasf[0], cmap='gray', origin='lower')
  plt.imsave(savepathgasf+'GRAYGASFEucl'+str(i+1)+'.png', X_gasf[1], cmap='gray', origin='lower')
  plt.imsave(savepathgasf+'GRAYGASFManh'+str(i+1)+'.png', X_gasf[2], cmap='gray', origin='lower')
  plt.imsave(savepathgadf+'GRAYGADFMink'+str(i+1)+'.png', X_gadf[0], cmap='gray', origin='lower')
  plt.imsave(savepathgadf+'GRAYGADFEucl'+str(i+1)+'.png', X_gadf[1], cmap='gray', origin='lower')
  plt.imsave(savepathgadf+'GRAYGADFManh'+str(i+1)+'.png', X_gadf[2], cmap='gray', origin='lower')
    
  imagem = np.reshape(X_gasf, (100, 100,3))
  imagem = (255*(imagem - np.min(imagem))/np.ptp(imagem)).astype(int)
  imagem = np.uint8(imagem)
  plt.imsave(savepathgasf+'GASF'+str(i+1)+'.png', imagem, origin='lower')

  imagem = np.reshape(X_gadf, (100, 100,3))
  imagem = (255*(imagem - np.min(imagem))/np.ptp(imagem)).astype(int)
  imagem = np.uint8(imagem)
  plt.imsave(savepathgadf+'GASF'+str(i+1)+'.png', imagem, origin='lower')