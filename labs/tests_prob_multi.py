from scipy.stats import multivariate_normal, norm
import numpy as np
import math

if __name__ == "__main__":

    '''
    mean = (93.33, 156.7)
    cov = 
    var = multivariate_normal(mean, cov)
    print(var.pdf((100, 225)))


    mean = (80, 203.3)
    cov = [[100, 50], [50, 233.3]]
    var = multivariate_normal(mean, cov)
    print(var.pdf((100, 225)))
    '''

    # also can use transposed vectors like np.array().T

    var1 = [30,30,50,50]
    var2 = [30,40,30,40]
    value = np.array([10,10])

    a_ = np.array([var1,var2])

    a = np.array([30,30,30,30])

    #mean = np.mean(a)
    #var = np.var(a, ddof=1)
    cov = np.cov(a_, ddof=1)
    #std = math.sqrt(var)
    
    mean = np.array([np.mean(var1),np.mean(var2)])

    
    print("mean: " + str(mean))
   # print("variance: " + str(var)) # sample like
    print("cov: " + str(cov)) # sample like
   # print("std dev: " + str(std)) # sample like

    #s = norm(mean, std).pdf(10)
    #print("probability " + str(s))
    

    try:
        K = cov
        inverse = np.linalg.inv(K)

        print("inverse")
        print(inverse)
        
        subt = value - mean

        x = np.dot(subt.T, inverse)

        x = np.dot(x, subt)

        det = np.linalg.det(K)
        x = (math.exp(-1/2*x))/(2*math.pi*math.sqrt(det))

        print("calculated")
        print(x)

        print("with librarys")
        var = multivariate_normal(mean, cov)
        print(var.pdf(value))
    except np.linalg.LinAlgError:
        # Not invertible. Skip this one.
        pass

    
    

