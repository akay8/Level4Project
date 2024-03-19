#! /usr/bin/env python
"""
Gauss_laser_mode.py
    
    Calculates the intensity- and phase distributions of Laguerre-Gauss
    (LG=True) or Hermite-Gauss (LG=False) laser modes.
    
    cc Fred van Goor, May 2020.
"""

# https://opticspy.github.io/lightpipes/HermiteGaussModes.html

from LightPipes import *
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


if LPversion < "2.0.0":
    print(r'You need to upgrade LightPipes to run this script.' + '\n'+r'Type at a terminal prompt: $ pip install --upgrade LightPipes')
    exit(1)

wavelength = 500*nm
size = 15*mm
N = 200
w0=3*mm
i=0
LG=False

n_max=3;m_max=4
fig, axs = plt.subplots(nrows=int(m_max/2), ncols=n_max,figsize=(11.0,7.0))
# if LG:
#     s=r'Laguerre-Gauss laser modes'
# else:
#     s=r'Hermite-Gauss laser modes'

# Setting TNR Font
# plt.rcParams.update({'font.size': 22})
# plt.rcParams.update({'font.family': 'Times New Roman'})

font = {'family' : 'Times New Roman',
        'size'   : 22}
plt.rc('font', **font)  # pass in the font dict as kwargs

# for ax in axs:
#     ax.rcParams.update({'font.size': 22})
#     ax.rcParams.update({'font.family': 'Times New Roman'})

print(plt.rcParams.keys())

fontdict_ = {'fontsize': 22,
 'fontweight': 'normal','fontfamily': 'Times New Roman'}

cols = list(np.zeros((2,3)))

F=Begin(size,wavelength,N)
for m in range(2):
    for n in range(n_max):
        F=GaussBeam(F, w0, LG=LG, n=n, m=m)
        I=Intensity(0,F)
        Phi=Phase(F)
        s=f'$TEM_{n}$' + f'$_{m}$'
        c = axs[m][n].imshow(I,cmap='jet')
        axs[m][n].axis('off')
        axs[m][n].set_title(s, **fontdict_)
        fig.colorbar(c,ax = axs[m][n])
        #axs[m+1+i][n].imshow(Phi,cmap='rainbow'); axs[m+1+i][n].axis('off');

plt.show()
