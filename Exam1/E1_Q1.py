import numpy as np

# mus=np.array([0.05,0.07,0.15,0.27])
# stds=np.array([0.07,0.12,0.30,0.60])
# R=np.array([[1,0.8,0.5,0.4],[0.8,1.0,0.7,0.5],[0.5,0.7,1.0,0.8],[0.4,0.5,0.8,1.0]])

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
        
    return w_star

print(optimize_portfolio(mus, stds, R, m='global_min'))
# print(optimize_portfolio(mus, stds, R, m=0.1))