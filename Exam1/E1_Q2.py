import numpy as np
np.random.seed(0)

mus=np.array([0.02,0.07,0.15,0.2])
stds=np.array([0.05,0.12,0.17,0.25])
R=np.array([[1.0, 0.3 ,0.3 ,0.3],
            [0.3 ,1.0, 0.6, 0.6],
            [0.3, 0.6 ,1.0 ,0.6],
            [0.3, 0.6, 0.6, 1.0]])


def tangency_portfolio(mus,stds,R,r):
    '''
    Parameters
    ----------
    mus : np.array
        return vector.
    stds : np.array
        standard deviation vector.
    R : np.array
        DESCRIPTION.
    r : float
        risk-free asset rate
    Returns
    -------
    w_tangency : np.array
        tangency allocation weight.
    mu_tangency : float
        tangency allocation return.
    std_tangency : float
        tangency allocation risk.
    '''
    dim=R.shape[0]
    unit=np.ones(dim)
    S=np.diag(stds)
    sigma=S.T.dot(R).dot(S)

    A=unit.T.dot(np.linalg.inv(sigma)).dot(unit)
    B=mus.T.dot(np.linalg.inv(sigma)).dot(unit)
    C=mus.T.dot(np.linalg.inv(sigma)).dot(mus)
    
    w_tangency=np.linalg.inv(sigma).dot(mus-r*unit)
    mu_tangency=(C-B*r)/(B-A*r)
    std_tangency=np.sqrt((C-2*r*B+r**2*A)/((B-A*r)**2))
        
    return w_tangency, mu_tangency, std_tangency

# w_tangency, mu_tangency, std_tangency=tangency_portfolio(mus, stds, R, r=0.005)
# print(w_tangency)
# print(mu_tangency)
# print(std_tangency)

import pandas as pd
df=pd.DataFrame()
df.index=['r50','r100','r150','r175']
ws_tan=[]
mus_tan=[]
stds_tan=[]
for r in [0.005,0.01,0.015,0.0175]:
    w_tangency, mu_tangency, std_tangency=tangency_portfolio(mus, stds, R, r=r)
    ws_tan.append(str(w_tangency))
    mus_tan.append(str(mu_tangency))
    stds_tan.append(str(std_tangency))
df['weight_tan']=ws_tan
df['return_tan']=mus_tan
df['stds_tan']=stds_tan

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = plt.axes()
# ax.set_title('Cloud of Simulated Allocation')
# ax.set_xlabel('Risk')
# ax.set_ylabel('Return')

# fig.colorbar(ax.scatter(stds_port, mus_port, c=np.array(mus_port) / np.array(stds_port), 
#                         marker='o', cmap='RdYlGn', edgecolors='black'), label='Sharpe Ratio') 


