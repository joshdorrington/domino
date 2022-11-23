import numpy as np
import xarray as xr

def get_occurrence(states,state_combinations=None):
    
    if np.ndim(states[0])==0:
        states=np.atleast_2d(states)

    states=np.array([a for b in states for a in b])
    
    if state_combinations is None:
        state_combinations=np.unique(states)
    K=len(state_combinations)
    occ=np.zeros(K)

    occ=np.array([sum(states==k) for k in state_combinations])
    occ=occ/len(states)
    
    assert np.abs(np.sum(occ)-1)<1e-5
    return occ

def get_xr_occurrence(da,dim='time',s=None,coord_name='regime'):

    da=da.dropna(dim)
    if s is None:
        states=np.unique(da)
    elif type(s)==xr.Dataset:
        states=s[da.name].values
        states=states[~np.isnan(states)]
    else:
        states=s
        
    if len(da)==0:
        occ=np.zeros_like(states)
    else:
        occ=get_occurrence(da,state_combinations=states)
    
    return xr.DataArray(data=occ,coords={coord_name:np.arange(len(occ))},attrs=da.attrs)


#Takes a state sequence or list of state sequences and calculates a transition matrix.
#states can be grouped with state_combinations, and the diag removed with exclude_diag.

def get_transmat(states,state_combinations=None,exclude_diag=False,t=1):
    
    #Rest of function assumes a list of state sequences
    if np.ndim(states[0])==0:
        states=np.atleast_2d(states)
        
    #Data setup
    if state_combinations is None:
        state_combinations=np.unique([x for y in states for x in y])
    K=len(state_combinations)
    trans=np.zeros([K,K])

    #We loop over transition matrix elements, using list comprehensions
    #to unravel our list of sequences
    for i,state1 in enumerate(state_combinations):
        for j,state2 in enumerate(state_combinations):
            trans[i,j]=np.sum([sum((np.isin(s[t:],state2))&(np.isin(s[:-t],state1)))\
                               for s in states])/np.sum([sum(np.isin(s[:-t],state1)) for s in states])

    if exclude_diag:
        trans -= np.diag(trans)*np.eye(K)
        trans /= trans.sum(axis=1)[:,None]
    return trans    

def synthetic_states_from_transmat(T,L,init_its=50,state_labs=None):
    
    if state_labs is None:
        K=T.shape[0]
        state_labs=np.arange(K)
        
    #get the occurrence distribution
    init_T=T
    for n in range(init_its):
        init_T=np.matmul(init_T,T)
    s0=np.digitize(np.random.rand(1).item(),np.cumsum(np.diagonal(init_T)))
    
    probs=np.random.rand(L)
    
    states=[s0]
    Tcumsum=np.cumsum(T,axis=1)
    for l in range(L):
        states.append(np.digitize(probs[l],Tcumsum[states[-1]]).item())
        
    states=[state_labs[s] for s in states]
    return np.array(states)
