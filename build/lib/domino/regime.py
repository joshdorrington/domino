import numpy as np
import xarray as xr
def xr_reg_occurrence(da,dim='time',s=None,coord_name='regime'):

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

def reg_lens(state_arr):
    sa=np.asarray(state_arr)
    n=len(sa)
    
    if n==0:
        return (None,None)
    
    else:
        #we create a truth array for if a state is about to change:
        x=np.array(sa[1:]!=sa[:-1])
        #we create an array containing those entries of sa where x is true
        #and append the final entry:
        y=np.append(np.where(x),n-1)
        #we create an array of persistent state lengths:
        L=np.diff(np.append(-1,y))
        #and an array of those state values:
        S=sa[y]
        return (L,S)
    
#This is an extension of reg lens that takes multiple discontiguous samples
#Data should be [Nsamples,Npoints] or a list of time series (allowed to be of different lengths)
def reg_lens_multi(state_arrs):
    sa=np.array(state_arrs)
    if np.ndim(sa[0])==0:
        sa=np.atleast_2d(sa)
        
    #we create a truth array for if a state is about to change:
    x=[np.array(s[1:])!=np.array(s[:-1]) for s in sa]
    #we create an array containing those entries of sa where x is true
    #and append the final entry:
    y=np.array([np.append(np.where(s),len(s)) for s in x])
    #we create an array of persistent state lengths:
    L=np.array([np.diff(np.append(-1,s)) for s in y])
    #and an array of those state values:
    S=np.array([np.array(s)[w] for s,w in zip(sa,y)])
    return(L,S)

def persistent_state(ix,state,min_pers):
    L,S=reg_lens(ix)

    keepS=S==state
    keepL=L>=min_pers
    keep_points=(np.repeat(keepL,L)*np.repeat(keepS,L))

    return keep_points

#Takes in a list of state sequences and returns a list of booleans,
#where only regime==state events lasting min_pers+ days are True.
def persistent_state_multi(ixs,state,min_pers,out="bool"):
    L,S=reg_lens_multi(ixs)

    keepS=[s==state for s in S]
    keepL=[l>=min_pers for l in L]
    keep_points=[np.repeat(kl,l)*np.repeat(ks,l) for kl,l,ks in zip(keepL,L,keepS)]
    if out=="bool":
        return keep_points
    if out=="states":
        keep_states=[np.array(s)[kp] for s,kp in zip(state_arr,keep_points)]
    
        return keep_states
    else: raise(ValueError(f"Unknown argument '{out}' to keyword 'out'"))

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

#Takes in a list of state sequences, some percentile bounds and a number of bootstraps
#to perform. It samples (by default with replacement) the list of sequences and calculates
#bootstrap estimates for transition matrix elements.
def get_transmat_bootstraps(states,percentiles,bootnum=400, replacement=True):
    
    if np.ndim(states[0])==0:
        states=np.atleast_2d(states)
        
    T=len(states)
    M=len(percentiles)
    K=len(np.unique([x for y in states for x in y]))
    
    transmat=-1*np.ones([M,K,K])
    
    boot_transmats=[]
    for n in range(bootnum):
        boot_states=np.array(states)[np.random.choice(T,T,replace=replacement)]
        boot_trans=get_transmat(boot_states)
        boot_transmats.append(boot_trans)
    return np.percentile(np.array(boot_transmats),percentiles,axis=0)
    
#Take 2 state sequences and return a matrix where element i,j says how many
#days in state i of s1 are also in state j of s2
#Can be given as numbers, or normalised by s1
def get_similarity_matrix(s1,s2,norm=False):
    
    assert len(s1)==len(s2)
    
    X=np.unique(s1)
    Y=np.unique(s2)
    sim_mat=np.zeros([len(X),len(Y)])
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            
            sim_mat[i,j]=sum((s1==x)&(s2==y))
    if norm is True:
        
        sim_mat=(sim_mat.T/np.array([np.sum(s1==x) for x in X])).T
    return sim_mat

def get_occurrence(states,state_combinations=None):
    
    if np.ndim(states[0])==0:
        states=np.atleast_2d(states)

    states=np.array([a for b in states for a in b])
    
    if state_combinations is None:
        state_combinations=np.unique(states)
    K=len(state_combinations)
    occ=np.zeros(K)

    #We loop over transition matrix elements, using list comprehensions
    #to unravel our list of sequences
    for i,state in enumerate(state_combinations):
        
        occ[i]=sum(states==state)/len(states)
    
    assert np.abs(np.sum(occ)-1)<1e-5
    return occ

#These are useful for batch computations:

def get_transmat_numerator(states,state_combinations=None,exclude_diag=False):
    
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
            trans[i,j]=np.sum([sum((np.isin(s[1:],state2))&(np.isin(s[:-1],state1))) for s in states])

    if exclude_diag:
        trans -= np.diag(trans)*np.eye(K)
        trans /= trans.sum(axis=1)[:,None]
    return trans    

def get_transmat_denominator(states,state_combinations=None,exclude_diag=False):
    
    
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
            trans[i,j]=np.sum([sum(np.isin(s[:-1],state1)) for s in states])

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
    for l in range(L):
        t=T[states[-1]]
        states.append(np.digitize(probs[l],np.cumsum(t)).item())
        
    states=[state_labs[s] for s in states]
    return np.array(states)
        