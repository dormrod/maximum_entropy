import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker


class NodeME:
    """
    Calculates maximum entropy node degree distribution, p_k, for a given k range across all p_6 values.
    Ref: A. ﻿Gervois, J.P. Troadec, J. Lemaitre, ﻿Journal of Physics A, 1992.
    """


    def __init__(self,k_limits=(3,20)):
        """
        Initialise with limits on node degrees. 
        Larger node degrees only contribute significantly at low p_6.
        Poisson Voronoi distribution expected to have 3<=k<=16.
        
        :param k_limits: lower and upper limit of node degrees.
        :type k_limits: tuple of int
        """

        # Set up calculation and results vectors
        self.k = np.arange(k_limits[0],k_limits[-1]+1,dtype=int)
        self.calculate_coefficients()
        self.pk = [] # me distribution
        self.k_var = [] # variance <k^2> - <k>^2


    def calculate_coefficients(self):
        """
        Calculate exponents for objective function to speed up calculation.
        Chi denotes coefficients for x, gamma coefficients for y.
        """

        self.chi = self.k
        self.gamma = 1/self.k


    def run(self,k_mean=6,pnts=1000,y_range=(2000,0)):
        """
        Calculate maximum entropy distribution for specified <k> at given number of points.
        
        :param k_mean: mean node degree
        :type k_mean: float
        :param pnts: number of points to calculate maximum entropy
        :type pnts: int
        :param y_range: optional parameter specifying range for second Lagrange multiplier, default selected for <k>=6
        :type y_range: tuple of floats
        """

        # Solve objective function for x given y values
        for y in np.linspace(y_range[0],y_range[1],pnts):
            opt = root(obj,x0=1,args=(y,self.chi,self.gamma,k_mean,self.k),)
            if opt.success:
                pk = calculate_pk(opt.x,y,self.chi,self.gamma)
                var_k = (self.k*self.k*pk).sum() - ((self.k*pk).sum())**2
                self.pk.append(pk)
                self.k_var.append(var_k)


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


    def get_variance(self):
        """
        Get variance of node distributions.
        
        :return: numpy array of (pnts)
        """

        return np.array(self.k_var)


    def plot_pk_variance(self,k=6):
        """
        Plot results of maximum entropy calculation, pk vs variance (Lemaitre's law).
        
        :param k: p_k to plot against variance
        :type k: int
        """

        # Set up plot
        params = {'font.family': 'serif','font.serif': 'DejaVu Serif','mathtext.fontset': 'dejavuserif',
            'axes.labelsize': 12,'axes.titlesize': 12,'xtick.labelsize': 12,'ytick.labelsize': 12}
        pylab.rcParams.update(params)
        fig, ax = plt.subplots()

        # Line graph of p_k vs variance
        ax.plot(self.get_pk(k=k),self.get_variance(),lw='1.5',color='mediumblue')
        ax.set_xlabel(r'$p_{}$'.format(k))
        ax.set_ylabel(r'$\langle k^2\rangle - \langle k \rangle^2$')
        ax.set_xlim((0,1))
        ax.set_ylim((0,None))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        plt.show()


    def write(self):
        """
        Write results of p_k and variance to file, 'node_dist.dat'
        """

        with open('./node_dist.dat','w') as f:
            f.write(('{:12.8f} '*(self.k.size)+'  {} \n').format(*self.k,'var'))
            fmt = '{:12.8f} '*(self.k.size+1)+'\n'
            for i,pk in enumerate(self.pk):
                f.write(fmt.format(*pk,self.k_var[i]))


def obj(x,y,chi,gamma,constraint,k):
    """
    Lemaitre objective function. 
    pk = e^(-x*chi) * e^(-y*gamma) / Z
    Solve sum_k (k-k_mean)pk == 0
    
    :param x: Lagrange multiplier 
    :param y: Lagrange multiplier
    :param chi: coefficients for Lagrange multiplier
    :param gamma: coefficients for Lagrange multiplier
    :param constraint: mean of k
    :param k: node degrees included in calculation
    :return: sum_k (k-k_mean)p_k
    """

    # Get significant p_k values
    p_k = calculate_pk(x,y,chi,gamma)

    # Calculate objective function
    f = np.sum((k-constraint) * p_k)

    return f


def calculate_pk(x,y,chi,gamma,tol=-20):
    """
    Calculate pk in logarithmic space to avoid underflow errors,
    as can get very small contributions from some ring sizes.
    pk = e^(-x*chi) * e^(-y*gamma) / Z
    
    :param x: Lagrange multiplier
    :param y: Lagrange multiplier
    :param chi: coefficients for Lagrange multiplier
    :param gamma: coefficients for Lagrange multiplier
    :param tol: cutoff for significant k values
    :return: normalised p_k with insignificant values set to zero
    """

    # Calculate unnormalised pk, sum of pk and therefore normalised pk
    log_pk = -x*chi + -y*gamma
    log_pk_max = np.max(log_pk)
    log_z = log_pk_max + np.log(np.sum(np.exp(log_pk - log_pk_max)))
    log_pk = log_pk - log_z

    # Mask k values making very small contributions
    pk = np.zeros_like(log_pk,dtype=float)
    k_sig_mask = log_pk>tol # significant k values to include
    pk[k_sig_mask] = np.exp(log_pk[k_sig_mask])

    return pk


if __name__ == '__main__':

    me = NodeME()
    me.run()
    me.plot_pk_variance()
    me.write()





