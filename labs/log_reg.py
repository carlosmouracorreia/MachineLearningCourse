import numpy as np
import math

target = [[-2],[-1],[0],[2]] 
y = [[2],[3],[1],[-1]]


x_input = [ np.insert(y, 0, 1, axis=0) for y in target ]


x_input = np.array(x_input)

print(x_input)


y_real_output = np.array(y)

def output(model, x):
    new_x = np.insert(x, 0, 1, axis=0)
    new_calc = np.dot(new_x, model)
    return new_calc


def modelParams():
    # first row filled with 1's, bias
   
    try:
        r = np.dot(x_input.T, x_input)

        r = np.linalg.inv(r)

        r = np.dot(r, x_input.T)
        
        r = np.dot(r,y_real_output)


        print("model params")
        print(r)
        return r

    except np.linalg.LinAlgError:
        # Not invertible. Skip this one.
        pass

if __name__ == "__main__":
    # main notation for array is one vector per row

    model = modelParams()
    out = output(model, [1])
    print("output is " + str(out))

    err = 0
    for i, x in enumerate(target):
        err = err + (y[i] - output(model,x))**2
    err = err / len(target)
    print ("error is " + str(err))

    # error by formula
    '''
    temp = y - np.dot(x_input, model)
    final = np.dot(temp.T, temp)

    print(final)
    '''

