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
from pyts.datasets import load_gunpoint
from numpy import genfromtxt
from pathlib import Path, PureWindowsPath
import os
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import MarkovTransitionField

# Parameters
file_path = 'D:\dev\Mestrado\Onlydata-features-4.csv'
savepathMTF = 'D:\\TCC\\ImagemTeste\\Imagens_n_Segmentadas\\grave\\Fractal\\MTF\\'

#Make save dir if not exists
if not os.path.exists(savepathMTF):
    os.makedirs(savepathMTF)
if not os.path.exists(savepathMTF):
    os.makedirs(savepathMTF)

n_img = 74

# Get data from csv
X = genfromtxt(file_path, delimiter=',')
#Define data range
a = X[0:n_img, 0:300]

for i in range (n_img):
  b = a[i, 0:300]
  X = b
  X = X.reshape(3, 100)
  # Transform the time series into Marcov Transition Fields
  mtf = MarkovTransitionField(n_bins=10)
  X_mtf = mtf.fit_transform(X)


  plt.imsave(savepathMTF+'GRAYMTFMink'+str(i+1)+'.png', X_mtf[0], cmap='gray', origin='lower')
  plt.imsave(savepathMTF+'GRAYMTFEucl'+str(i+1)+'.png', X_mtf[1], cmap='gray', origin='lower')
  plt.imsave(savepathMTF+'GRAYMTFManh'+str(i+1)+'.png', X_mtf[2], cmap='gray', origin='lower')
    
  imagem = np.reshape(X_mtf, (100, 100,3))
  imagem = (255*(imagem - np.min(imagem))/np.ptp(imagem)).astype(int)
  imagem = np.uint8(imagem)
  plt.imsave(savepathMTF+'MTF'+str(i+1)+'.png', imagem, origin='lower')