import numpy as np 
import math

def tanh(x):
    return 2/(1+math.exp(-2*x)) -1

if __name__ == "__main__":
    print(tanh(0.164))