import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker


class EdgeME:
    """
    Calculates maximum entropy edge joint degree distribution, ejk, for a given k range
    with specified mean across range of variances and assortativities.
    Calculates maximum entropy node degree distribution, pk, simultaneously.
    Ref: XXX
    """


    def __init__(self,k_limits=(3,20)):
        """
        Initialise with limits on edge degrees. 
        Larger edge degrees only contribute significantly at low p6.
        Poisson Voronoi distribution expected to have 3<=k<=16.
        
        :param k_limits: lower and upper limit of edge degrees.
        :type k_limits: tuple of int
        """

        # Set up calculation and results vectors
        self.k = np.arange(k_limits[0],k_limits[-1]+1,dtype=int)
        self.kj = np.array([np.array([ki for i in range(self.k.size)]) for ki in self.k])
        self.kk = np.array([np.array([kj for kj in self.k]) for i in range(self.k.size)])
        self.calculate_coefficients()
        self.ejk = [] # me edge joint distribution
        self.pk = [] # me node distribution
        self.r = [] # assortative mixing


    def calculate_coefficients(self):
        """
        Calculate coefficients for Lagrange multipliers in objective function to speed up calculation.
        Chi denotes coefficients for x, gamma coefficients for y, zeta coefficients for z.
        """

        self.chi = 0.5*(self.kj+self.kk)
        self.gamma = 0.5*(1/self.kj+1/self.kk)
        self.zeta = self.kj*self.kk


    def run(self,k_mean=6,pnts=1000,y_range=(3000,0),z=0.0):
        """
        Calculate maximum entropy distribution for specified <k> at given number of points.
        
        :param k_mean: mean node degree
        :type k_mean: float
        :param pnts: number of points to calculate maximum entropy
        :type pnts: int
        :param y_range: optional parameter specifying range for second Lagrange multiplier, default selected for <k>=6
        :type y_range: tuple of floats
        :param z: value of Lagrange multiplier determining the assortativity
        :type z: float
        """

        # Solve objective function for x given y and z values
        self.k_mean = k_mean
        constraint = 1/self.k_mean
        for y in np.linspace(y_range[0],y_range[1],pnts):
            opt = root(obj,x0=1,args=(y,z,self.chi,self.gamma,self.zeta,constraint,self.kj,self.kk))
            if opt.success:
                ejk = calculate_ejk(opt.x,y,z,self.chi,self.gamma,self.zeta)
                pk,r = self.analyse_ejk(ejk)
                self.ejk.append(ejk)
                self.pk.append(pk)
                self.r.append(r)


    def analyse_ejk(self,ejk):
        """
        Calculate network metrics from edge joint degree distribution.
         
        :param ejk: edge joint degree distribution
        :type ejk: numpy (n,n) array of floats
        :return: node degree distribution, pk; assortativity, r. 
        """

        qk = np.sum(ejk,axis=1) # edge degree distribution
        pk = self.k_mean*qk/self.k # node degree distribution

        # Assortativity
        r = 0.0
        for j,kj in enumerate(self.k):
            for k,kk in enumerate(self.k):
                r += kj*kk*(ejk[j,k]-qk[j]*qk[k])
        r /= (self.k*self.k*qk).sum() - ((self.k*qk).sum())**2

        return pk,r


    def get_ejk(self):
        """
        Get all edge joint distributions.
        
        :return: numpy array of (pnts,ejk)
        """

        return np.array(self.ejk)


    def get_pk(self,k=None):
        """
        Get all node distributions, for single k value or range of k values.
        
        :param k: k values to include in pk, if None then includes all k
        :type k: None, int or list of ints
        :return: numpy array of (pnts,pk)
        """

        if k is None:
            return np.array(self.pk)
        else:
            return np.array(self.pk)[:,self.k==k]


    def get_r(self):
        """
        Get all assortativities.
        
        :return: numpy array of (pnts,r)
        """

        return np.array(self.r)


    def plot_pk_r(self,k=6):
        """
        Plot results of maximum entropy calculation, pk vs assortativity.
        
        :param k: pk to plot against variance
        :type k: int
        """

        # Set up plot
        params = {'font.family': 'serif','font.serif': 'DejaVu Serif','mathtext.fontset': 'dejavuserif',
                  'axes.labelsize': 12,'axes.titlesize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12}
        pylab.rcParams.update(params)
        fig, ax = plt.subplots()

        # Scatter graph of p_k vs variance
        ax.scatter(self.get_pk(k=k),self.get_r(),color='mediumblue',marker='o',s=5)
        ax.set_xlabel(r'$p_{}$'.format(k))
        ax.set_ylabel(r'$r$')
        ax.set_xlim((0,1))
        ax.set_ylim((-1,1))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        plt.show()


def obj(x,y,z,chi,gamma,zeta,constraint,kj,kk):
    """
    Objective function. 
    ejk = e^(-x*chi) * e^(-y*gamma) * e^(-z*zeta) / Z
    Solve sum_jk ((1/j+1/k)/2-1/k_mean)ejk == 0
    
    :param x: Lagrange multiplier 
    :param y: Lagrange multiplier
    :param z: Lagrange multiplier
    :param chi: coefficients for Lagrange multiplier
    :param gamma: coefficients for Lagrange multiplier
    :param zeta: coefficients for Lagrange multiplier
    :param constraint: reciprocal of mean of k
    :param kj: edge degrees included in calculation of j index
    :param kk: edge degrees included in calculation of k index
    :return: sum_jk ((1/j+1/k)/2-1/k_mean)ejk
    """

    # Get significant ejk values
    ejk = calculate_ejk(x,y,z,chi,gamma,zeta)

    # Calculate objective function
    f = np.sum((0.5*(1/kj+1/kk)-constraint) * ejk)

    return f


def calculate_ejk(x,y,z,chi,gamma,zeta,tol=-20):
    """
    Calculate ejk in logarithmic space to avoid underflow errors,
    as can get very small contributions from some ring sizes.
    ejk = e^(-x*chi) * e^(-y*gamma) * e^(-z*zeta) / Z
    
    :param x: Lagrange multiplier
    :param y: Lagrange multiplier
    :param z: Lagrange multiplier
    :param chi: coefficients for Lagrange multiplier
    :param gamma: coefficients for Lagrange multiplier
    :param zeta: coefficients for Lagrange multiplier
    :param tol: cutoff for significant k values
    :return: normalised ejk with insignificant values set to zero
    """

    # Calculate unnormalised ejk, sum of ejk and therefore normalised ejk
    log_ejk = - x*chi - y*gamma - z*zeta
    log_ejk_max = np.max(log_ejk)
    log_norm = log_ejk_max + np.log(np.sum(np.exp(log_ejk - log_ejk_max)))
    log_ejk = log_ejk - log_norm

    # Mask k values making very small contributions
    ejk = np.zeros_like(log_ejk,dtype=float)
    jk_sig_mask = log_ejk>tol # significant j,k values to include
    ejk[jk_sig_mask] = np.exp(log_ejk[jk_sig_mask])

    return ejk


if __name__ == '__main__':

    me = EdgeME()
    me.run(z=0.1)
    me.plot_pk_r()
    # me.write()
