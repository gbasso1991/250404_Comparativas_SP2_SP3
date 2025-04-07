#%% VSM old ways - 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
from sklearn.metrics import r2_score 
from mlognormfit import fit3
from mvshtools import mvshtools as mt
from datetime import datetime
import re
def lineal(x,m,n):
    return m*x+n
        
#%% Leo Archivos 
# patron Superparamagnetico 2 - NF en Alumina
SP2 = np.loadtxt('Patron_SP2.txt', skiprows=12)
H_SP2 = SP2[:, 0]  # Gauss
m_SP2 = SP2[:, 1]  # emu

factor_Flavio= 6.92/6.902 # debido a calibracion posterior
m_SP2*=factor_Flavio
# mass_capsula = 1.19897  # g
# mass_capsula_c_SP1 = 1.47020  # g
mass_muestra_SP2 = 0.17659 # g
C_SP2 = 27.66  #concentracion estimada en g/L = kg/m³
vol_capsula=105*1e-6 #en L 
mass_NP_SP2=C_SP2*vol_capsula
# Normalizo momento por masa de NP

m_SP2 /= mass_NP_SP2  # emu/g


#% Generar señales anhisteréticas
H_anhist_SP2, m_anhist_SP2 = mt.anhysteretic(H_SP2, m_SP2)

# Graficar señales anhisteréticas
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
ax.plot(H_SP2, m_SP2, '.-', label='SP2')
ax.plot(H_anhist_SP2, m_anhist_SP2, '.-', label='SP2 anhisteretica')

for a in [ax]:
    a.legend(ncol=2)
    a.grid()
    a.set_ylabel('m (emu/g)')
plt.xlabel('H (G)')
plt.show()

#%Realizo fits en ciclos CON contribucion diamag
fit_SP2 = fit3.session(H_anhist_SP2, m_anhist_SP2, fname='SP2', divbymass=False)
fit_SP2.fix('sig0')
fit_SP2.fix('mu0')
fit_SP2.free('dc')
fit_SP2.fit()
fit_SP2.update()
fit_SP2.free('sig0')
fit_SP2.free('mu0')
fit_SP2.set_yE_as('sep')
fit_SP2.fit()
fit_SP2.update()
fit_SP2.save()
fit_SP2.print_pars()
H_SP2_fit = fit_SP2.X
m_SP2_fit = fit_SP2.Y
m_SP2_sin_diamag = m_anhist_SP2 - lineal(H_anhist_SP2, fit_SP2.params['C'].value, fit_SP2.params['dc'].value)

#% Graficar resultados eliminando comportamiento diamagnético
fig, ax = plt.subplots(nrows=1, figsize=(8, 6),constrained_layout=True)

ax.plot(H_SP2, m_SP2, '.-', label='SP2')
ax.plot(H_anhist_SP2, m_anhist_SP2, '.-', label='SP2 anhisteretica')
ax.plot(H_anhist_SP2, m_SP2_sin_diamag, '-', label='SP2 s/ diamag')
ax.plot(H_SP2_fit, m_SP2_fit, '-', label='SP2 fit')

for a in [ax]:
    a.legend(ncol=1)
    a.grid()
    a.set_ylabel('m (emu/g)')
plt.xlabel('H (G)')
plt.show()

#% Salvo ciclo VSM fiteado 
data = np.column_stack((H_SP2_fit, m_SP2_fit))
# Guarda la matriz en un archivo de texto
np.savetxt('SP2_fitting.txt', data, fmt=('%e','%e'),
           header='H_G  |  m_emu/g',delimiter='\t')
#%% patron Superparamagnetico 3 - NF@citrato en Alumina
SP3 = np.loadtxt('Patron_SP3.txt', skiprows=12)
H_SP3 = SP3[:, 0]  # Gauss
m_SP3 = SP3[:, 1]  # emu

factor_Flavio= 6.92/6.902 # debido a calibracion posterior
m_SP3*=factor_Flavio
# mass_capsula = 1.19897  # g
# mass_capsula_c_SP1 = 1.47020  # g
mass_muestra_SP3 = 0.17659 # g
C_SP3 = 44.31  #concentracion estimada en g/L = kg/m³
vol_capsula=114*1e-6 #en L 
mass_NP_SP3=C_SP3*vol_capsula
# Normalizo momento por masa de NP

m_SP3 /= mass_NP_SP3  # emu/g


#% Generar señales anhisteréticas
H_anhist_SP3, m_anhist_SP3 = mt.anhysteretic(H_SP3, m_SP3)

# Graficar señales anhisteréticas
fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
ax.plot(H_SP3, m_SP3, '.-', label='SP3')
ax.plot(H_anhist_SP3, m_anhist_SP3, '.-', label='SP3 anhisteretica')

for a in [ax]:
    a.legend(ncol=2)
    a.grid()
    a.set_ylabel('m (emu/g)')
plt.xlabel('H (G)')
plt.show()

#%Realizo fits en ciclos CON contribucion diamag
fit_SP3 = fit3.session(H_anhist_SP3, m_anhist_SP3, fname='SP3', divbymass=False)
fit_SP3.fix('sig0')
fit_SP3.fix('mu0')
fit_SP3.free('dc')
fit_SP3.fit()
fit_SP3.update()
fit_SP3.free('sig0')
fit_SP3.free('mu0')
fit_SP3.set_yE_as('sep')
fit_SP3.fit()
fit_SP3.update()
fit_SP3.save()
fit_SP3.print_pars()
H_SP3_fit = fit_SP3.X
m_SP3_fit = fit_SP3.Y
m_SP3_sin_diamag = m_anhist_SP3 - lineal(H_anhist_SP3, fit_SP3.params['C'].value, fit_SP3.params['dc'].value)

#% Graficar resultados eliminando comportamiento diamagnético
fig, ax = plt.subplots(nrows=1, figsize=(8, 6),constrained_layout=True)

ax.plot(H_SP3, m_SP3, '.-', label='SP3')
ax.plot(H_anhist_SP3, m_anhist_SP3, '.-', label='SP3 anhisteretica')
ax.plot(H_anhist_SP3, m_SP3_sin_diamag, '-', label='SP3 s/ diamag')
ax.plot(H_SP3_fit, m_SP3_fit, '-', label='SP3 fit')

for a in [ax]:
    a.legend(ncol=1)
    a.grid()
    a.set_ylabel('m (emu/g)')
plt.xlabel('H (G)')
plt.show()

#% Salvo ciclo VSM fiteado 
data = np.column_stack((H_SP3_fit, m_SP3_fit))
# Guarda la matriz en un archivo de texto
np.savetxt('SP3_fitting.txt', data, fmt=('%e','%e'),
           header='H_G  |  m_emu/g',delimiter='\t')

# %%
