import numpy as np

def policy_evaluation(S,A,P,R,pi,gamma=1.0,theta=1e-10):
    nS,nA,_=P.shape; V=np.zeros(nS)
    while True:
        delta=0; V_new=np.zeros_like(V)
        for s in range(nS):
            val=0
            for a in range(nA):
                if pi[s,a]==0: continue
                val+=pi[s,a]*np.sum(P[s,a,:]*(R[s,a,:]+gamma*V))
            V_new[s]=val; delta=max(delta,abs(V_new[s]-V[s]))
        V=V_new
        if delta<theta: break
    return V

def q_from_v(S,A,P,R,V,gamma=1.0):
    nS,nA,_=P.shape; Q=np.zeros((nS,nA))
    for s in range(nS):
        for a in range(nA):
            Q[s,a]=np.sum(P[s,a,:]*(R[s,a,:]+gamma*V))
    return Q
