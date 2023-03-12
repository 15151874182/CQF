import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0) ##set random seed

class Portfolio():
    
    def __init__(self,u,std,R):
        self.u=u     ##given risky asset return
        self.std=std ##given risky asset std
        self.R=R     ##given risky asset correlation matrix
        
        self.dim=R.shape[0]  ##risky assets num
        self.unit=np.ones(self.dim) ## unit vect
        self.S=np.diag(std)  ##diagnoal matrix
        self.sigma=self.S.T.dot(self.R).dot(self.S)  ##sigma matrix
        self.inv_sigma=np.linalg.inv(self.sigma) ##inverse sigma matrix
        
        self.A=self.unit.T.dot(np.linalg.inv(self.sigma)).dot(self.unit)
        self.B=self.u.T.dot(np.linalg.inv(self.sigma)).dot(self.unit)
        self.C=self.u.T.dot(np.linalg.inv(self.sigma)).dot(self.u)        
    
    def generate_samples(self,N=700): ##N->number of samples
        ws=[np.random.random(self.dim) for _ in range(N)]  ##N random allocations
        ws_norm=list(map(lambda w:w/sum(w),ws))  ##Standardise weights which satisfy sum(w) = 1
        u_port_list=list(map(lambda w:w.T.dot(self.u),ws_norm)) 
        std_port_list=list(map(lambda w:np.sqrt(w.T.dot(self.sigma).dot(w)),ws_norm)) 
        return std_port_list, u_port_list
    
    def get_EF_points(self, m, r=None): ##m->given portfolio return  r->risk free rate
        if not r: ##risky assets only
            w_star=1/(self.A*self.C-self.B**2)*self.inv_sigma.dot((self.A*self.u-self.B*self.unit)*m+(self.C*self.unit-self.B*self.u))
        else:     ##with risk-free asset
            numerator=(m-r)*self.inv_sigma.dot(self.u-r*self.unit) 
            denominator=(self.u-r*self.unit).dot(self.inv_sigma).dot(self.u-r*self.unit) 
            w_star=numerator/denominator
        std_port=np.sqrt(w_star.T.dot(self.sigma).dot(w_star))
        u_port=w_star.T.dot(self.u)
        return w_star, std_port, u_port

    def get_CML(self, r, wt_std_port, wt_u_port):
        slope=(wt_u_port-r)/wt_std_port
        intercept=r
        return slope, intercept

    def get_min_var_portfolio(self): 
        w_g=self.inv_sigma.dot(self.unit)/self.A
        wg_std_port=np.sqrt(w_g.T.dot(self.sigma).dot(w_g))
        wg_u_port=w_g.T.dot(self.u)
        return w_g, wg_std_port, wg_u_port    

    def get_tangency_portfolio(self,r=None): ##r->risk free rate 
        w_t=self.inv_sigma.dot(self.u-r*self.unit)/(self.B-self.A*r)
        wt_std_port=np.sqrt(w_t.T.dot(self.sigma).dot(w_t))
        wt_u_port=w_t.T.dot(self.u)
        return w_t, wt_std_port, wt_u_port      
    
    def plot(self,r, std_port_list, u_port_list, 
                  wg_std_port, wg_u_port,
                  wt_std_port, wt_u_port):
        u_list1 = list(np.linspace(wg_u_port, wt_u_port, 100))
        std_list1=[self.get_EF_points(u)[1] for u in u_list1]
        
        slope, intercept=self.get_CML(r, wt_std_port, wt_u_port)
        std_list2 = list(np.linspace(0, 0.25, 100))
        u_list2=[std*slope+intercept for std in std_list2]        
        
        plt.colorbar(plt.scatter(std_port_list, u_port_list, c=np.array(u_port_list) / np.array(std_port_list), 
                                marker='o', cmap='RdYlGn', edgecolors='black'), label='Sharpe Ratio') 
        plt.plot(wg_std_port, wg_u_port, 'r*', markersize=18)
        plt.plot(wt_std_port, wt_u_port, 'g*', markersize=18)
        plt.plot(std_list1, u_list1, 'r-')
        plt.plot(std_list2, u_list2, 'b--')
        plt.xlim(0, 0.25)
        plt.ylim(0, 0.25)
        plt.title('Cloud of Simulated Allocation')
        plt.xlabel('Risk')
        plt.ylabel('Return')
        plt.grid('True')
    
  

if __name__=='__main__':
    r=0.005
    u=np.array([0.02,0.07,0.15,0.2])
    std=np.array([0.05,0.12,0.17,0.25])
    R=np.array([[1.0, 0.3 ,0.3 ,0.3],
                [0.3 ,1.0, 0.6, 0.6],
                [0.3, 0.6 ,1.0 ,0.6],
                [0.3, 0.6, 0.6, 1.0]])

    portfolio=Portfolio(u=u,std=std,R=R)
    w_g, wg_std_port, wg_u_port=portfolio.get_min_var_portfolio()
    # w_star, std_port, u_port=portfolio.get_EF_points(m=0.1,r=0.025)
    w_t, wt_std_port, wt_u_port=portfolio.get_tangency_portfolio(r=r)
    std_port_list, u_port_list=portfolio.generate_samples(N=700)
    portfolio.plot(r=r, std_port_list=std_port_list,u_port_list=u_port_list,
                   wg_std_port=wg_std_port,wg_u_port=wg_u_port,
                   wt_std_port=wt_std_port, wt_u_port=wt_u_port)


