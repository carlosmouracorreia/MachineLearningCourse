
import numpy as np 
import math


def log(v):
    return -math.log2(v)*v


def entropy(vector):
    vector = [log(v) for v in vector]
    result = np.sum(vector)
    return result

if __name__ == "__main__":
    x = 2/5*entropy([1/2,1/2]) + 3/5*entropy([2/3,1/3])
    final = 1.92 - x
    print(final)
    