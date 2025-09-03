def compute():
    N,Q,R=4,0.5,1; N_new=N+1
    Q_new=Q+(R-Q)/N_new
    return {"N":N,"N_new":N_new,"Q_new":Q_new}
if __name__=="__main__": print(compute())
