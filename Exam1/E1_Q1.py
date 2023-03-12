import numpy as np
np.random.seed(0)
mus=np.array([0.05,0.07,0.15,0.27])
stds=np.array([0.07,0.12,0.30,0.60])
R=np.array([[1,0.8,0.5,0.4],
            [0.8,1.0,0.7,0.5],
            [0.5,0.7,1.0,0.8],
            [0.4,0.5,0.8,1.0]])

# R=np.array([[1, 0, 0, 0],
#             [0, 1.0, 0, 0],
#             [0, 0, 1.0, 0],
#             [0, 0, 0, 1.0]])


mus=np.array([0.02,0.07,0.15,0.2])
stds=np.array([0.05,0.12,0.17,0.25])
R=np.array([[1.0, 0.3 ,0.3 ,0.3],
            [0.3 ,1.0, 0.6, 0.6],
            [0.3, 0.6 ,1.0 ,0.6],
            [0.3, 0.6, 0.6, 1.0]])


def optimize_portfolio(mus,stds,R,m='global_min'):
    '''
    Parameters
    ----------
    mus : np.array
        return vector.
    stds : np.array
        standard deviation vector.
    R : np.array
        DESCRIPTION.
    m : 'global_min' or float
        global minimum variance portfolio if return is not specified.
    Returns
    -------
    w_star : np.array
        optimal allocations.
    '''
    dim=R.shape[0]
    unit=np.ones(dim)
    S=np.diag(stds)
    sigma=S.T.dot(R).dot(S)

    A=unit.T.dot(np.linalg.inv(sigma)).dot(unit)
    B=mus.T.dot(np.linalg.inv(sigma)).dot(unit)
    C=mus.T.dot(np.linalg.inv(sigma)).dot(mus)
    if m=='global_min':
        m=B/A
    w_star=1/(A*C-B**2)*np.linalg.inv(sigma).dot((A*mus-B*unit)*m+(C*unit-B*mus))
        
    return w_star,sigma

w_star,sigma=optimize_portfolio(mus, stds, R, m='global_min')
# w_star,sigma=optimize_portfolio(mus, stds, R, m=0.1)
print(w_star)
print(sigma)
print(w_star.dot(mus))
print(np.sqrt(w_star.T.dot(sigma).dot(w_star)))

ws=[np.random.random(R.shape[0]) for _ in range(700)]  ##700 random allocation sets
ws_norm=list(map(lambda w:w/sum(w),ws))       ##700 random allocation sets which  satisfy w1 = 1
mus_port=list(map(lambda w:w.T.dot(mus),ws_norm)) 
stds_port=list(map(lambda w:np.sqrt(w.T.dot(sigma).dot(w)),ws_norm)) 
# import pandas as pd
# df=pd.DataFrame()
# df['mus_port']=mus_port
# df['stds_port']=stds_port
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes()
ax.set_title('Cloud of Simulated Allocation')
ax.set_xlabel('Risk')
ax.set_ylabel('Return')

fig.colorbar(ax.scatter(stds_port, mus_port, c=np.array(mus_port) / np.array(stds_port), 
                        marker='o', cmap='RdYlGn', edgecolors='black'), label='Sharpe Ratio') 


