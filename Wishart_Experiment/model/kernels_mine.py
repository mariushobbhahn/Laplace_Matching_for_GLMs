import numpy as np

def ou(t1, t2, params):
    """
    Ornstein-Uhlenbeck kernel. Commonly used for financial data because
    it's not quite as smooth as the squared-exponential kernel.
    """
    tau = params[0]
    sigma = params[1]
    
    return sigma**2 * np.exp(-abs(t2 - t1)/tau)
    #return np.exp(-abs(t2 - t1)/tau)

def squared_exponential(t1, t2, params):
    """
    Squared-exponential kernel.
    """
    tau = params[0]
    sigma = params[1]

    return sigma**2 * np.exp(-0.5*((t1 - t2)**2/tau**2))
    #return np.exp(-0.5*((t1 - t2)**2/tau**2))

def periodic(t1, t2, params):
    """
    A simple periodic kernel function.
    """
    sigma = params[0]    
    tau = params[1]
    p = params[2]
    
    return(sigma**2 * np.exp(-2*np.sin(np.abs(t1 - t2)/p)**2 / tau**2))

def linear(t1, t2, params):
    """
    A simple linear kernel function.
    """
    a = params[0]
    sigma = params[1]
    
    return(sigma**2 + a * t1 * t2)

def rational_quadratic(t1,t2, params, a=1):
    """
    The rational quadratic kernel
    """
    tau = params[0]
    sigma = params[1]
    return(sigma**2 * (1 + (t1- t2)**2/(2 * a * tau**2))**(-a))

def sum_kernel(t1, t2, k1, k2, params1, params2):
    """
    Just add two kernels of your choice with respective parameter sets.
    """
    
    return(0.5*k1(t1, t2, params1) + 0.5*k2(t1, t2, params2))


"""
def generate_sum_kernel(k1, k2, num_taus=(1, 1)):
    def k(t1, t2, params):
        p1, p2 = params[:num_taus[0]], params[num_taus[0]:]

        return 0.5 * k1(t1, t2, p1) + 0.5 * k2(t1, t2, p2)
    
    return k

def generate_prod_kernel(k1, k2, num_taus=(1, 1)):
    def k(t1, t2, params):
        p1, p2 = params[:num_taus[0]], params[num_taus[0]:]

        return k1(t1, t2, p1) * k2(t1, t2, p2)

    return k
"""
