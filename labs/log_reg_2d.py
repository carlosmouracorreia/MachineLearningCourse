import numpy as np
import math

target = [[-0.95,0.62],[0.63,0.31],[-0.12,-0.21],[-0.24,-0.5], [0.07, -0.42], [0.03,0.91], [0.05,0.09], [-0.83,0.22]] 
y = [[0],[0],[1],[0], [1], [0], [1], [0]]

def square(vector):
    return [x**2 for x in vector]


x_input = [ np.insert(square(y), 0, 1, axis=0) for y in target ]


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
    out = output(model, [2,3])
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


# still - plot linear regression (for more than 2d)