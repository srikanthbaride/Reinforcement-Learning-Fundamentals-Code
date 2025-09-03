import numpy as np, math
def compute():
    Q=[0.6,0.7,0.4]; N=[5,10,2]; t=20;c=1.0
    ucb=[Q[a]+c*math.sqrt(math.log(t)/N[a]) for a in range(3)]
    return {"ucb":ucb,"selected":np.argmax(ucb)+1}
if __name__=="__main__": print(compute())

