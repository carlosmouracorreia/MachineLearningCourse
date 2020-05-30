        # save initial weights - bias and Wo ?? but no inital examples yet, so what does the model really builts? 50  nets.py
    # here instead of stochastic gradient descent, ADAM is used. What is this? 69 multi_nets.py



    # Convultion notes

    searching for occurrences of a feature represented by a kernel in the image. It corresponds to sliding a window with the same size as the kernel through the image.


    CNNs take a different approach towards regularization: they take advantage of the hierarchical pattern in data and assemble more complex patterns using smaller and simpler patterns. Therefore, on the scale of connectedness and complexity, CNNs are on the lower extreme.

    Convolution is a specialized kind of linear operation. Convolutional networks are simply neural networks that use convolution in place of general matrix multiplication in at least one of their layers.[11]


    .activation function for last layers is the positive part of its argument max(0,x)!  before it was softmax

    .hidden layers have convolutions in it. mathematically it is a sliding dot product / cross-relation. It affects how weight is determined at a specific index 


    INPUT for initial layer is a tensor with shape (nr_images) x (image_Width) * (image_height) * (image_depth/nr_of_bits)

    then abstracted to a feature map, with shape (number of images) x (feature map width) x (feature map height) x (feature map channels). 


    CONVOLUTION OP IS A dot product between input matrix (image) and kernel matrix starting at the current i,j of the image width, height. so if we're futher on the width, height convolution matrix we'll get less terms in the dot product.

    we can also define how much width, height we want to take into consideration and thus have a a model with a more "shrinked" image "idea". This is called variable stride (for width or height)

    padding can also be used (k=x, p=y for x extra imaginary lines or columns) to look around all possible input positions

    CHANNELS - each color has as matrix of pixels so we need 3 matrixes for colors (3 channels usually)

    apply convolution between each color channel and its corresponding kernel matrix and then  sum the results of the 3 matrixes.


    CONVOLUTION LAYER - we say that the output of a layer with k kernels is an new representation of an image with k channels. Use different kernels for the different colors in different groups. Resulting in several image ideas with e.g "9 channels", "9 kernels" and 3 image ideas (if an image uses the RGB system - 3 kernels for each image)



    POOLING OP - No kernel matrixes but instead max or avg are done between the sliding window we've discussed until here.
    usually cool because it allows noise to fade, we have a smaller image and less parameters, so more generalization. don't overpool otherwise we lose not only noise but important features.


    POOLING LAYER - pooling op done independently for each channel.



    SEE REST OF THE CLASS RECORDING 9 codlab

    SEE MORE PRATICAL CLASS SOLVED EXERCICES