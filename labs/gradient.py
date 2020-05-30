import sympy as sym
from sympy import init_printing, pprint, lambdify
import numpy as np
def sigmoid(x):
    return 1/(1 + sym.exp(-x))
    
if __name__ == "__main__":

    init_printing() 

    x = sym.Symbol("x")
    #print(sym.diff(x**5))

    #print(sym.diff((x**2 + 1) *  sym.cos(x) ))


    # partial div
    x, y = sym.symbols("x y")
    f = x**2 * y
    #print(sym.diff(f, x))
    #print(sym.diff(f, y))

    t,w,x = sym.symbols("t w x")

    f = (((t - sigmoid(2*w*x))**2)/2)

    result = sym.diff(f,w)


    sigm = sigmoid(w)
    result = 2* ((t - sigm) *  ( sigm * (1 - sigm ) ) * x )

    pprint(result)

    H = lambdify( [t,w,x], result, "numpy" )

    #a = result.evalf(subs={t: 1, w: sym.Matrix([1, 1,1]), x: sym.Matrix([1, 2,1])})


   # print(H(1,np.array([[1],[1],[1]]),np.array([[1],[2],[1]])))
    
    # doesnt seem to work with the vectors, I have to first do the dot product...
    w = np.array([1,1,1])
    x = np.array([1,1,1])
    _2wx = 2*np.dot(w,x)

    res = H(1,_2wx,x)

    print(res)

    # then do this for all points and sum the first weight vector
    res = w - res
    print(res)
    

    # this way we're just doing stochastic because its only choosing a random x (1,1) with the first element being the bias