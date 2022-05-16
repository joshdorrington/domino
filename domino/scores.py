import numpy as np

def BS(fc,obs):
    return np.mean((fc-obs)**2)    

def BSS(fc,obs):
    clim_fc=np.ones_like(obs)*np.mean(obs) #Constantly predicts mean prob of the obs
    return 1-(BS(fc,obs)/BS(clim_fc,obs))
