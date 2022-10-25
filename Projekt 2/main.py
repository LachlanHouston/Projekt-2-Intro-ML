#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv (r'/Users/frederikravnborg/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU-Frederik’s MacBook Pro/ML/Project 1/penguins.csv')
#data = pd.read_csv(r'C:/Users/ssjsi/Documents/GitHub/Projekt-2-Intro-ML/Projekt 2/penguins_size.csv')


cullen = data["culmen_length_mm"]
culdep = data["culmen_depth_mm"]
flilen = data["flipper_length_mm"]
bodmas = data["body_mass_g"]

# Standardize
cullens = (cullen-np.mean(cullen))/np.std(cullen)
culdeps = (culdep-np.mean(culdep))/np.std(culdep)
flilens = (flilen-np.mean(flilen))/np.std(flilen)
bodmass = (bodmas-np.mean(bodmas))/np.std(bodmas)



#%%



#%%