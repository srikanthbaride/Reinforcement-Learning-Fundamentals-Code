import numpy as np
def compute():
    mu = np.array([0.5,0.6,0.7]); T=100; chosen=0
    mu_star = mu.max()
    regret = T*mu_star - T*mu[chosen]
    return {"optimal":np.argmax(mu)+1,"mu*":mu_star,"chosen_mu":mu[chosen],"regret":regret}
if __name__=="__main__": print(compute())
