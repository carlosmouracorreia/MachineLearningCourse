import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np

import sklearn.decomposition
import matplotlib.pyplot as plt


if __name__ == "__main__":
    iris_data, iris_info = tfds.load('iris', with_info=True)
    print(iris_info)

    # x is feature vectors (matrix)
    iris_x = np.asarray([instance['features'] for instance in tfds.as_numpy(iris_data['train'])])
    # y is class label (vector)
    iris_y = np.asarray([instance['label'] for instance in tfds.as_numpy(iris_data['train'])])

    # print training instances
    for f, c in zip(iris_x[:5], iris_y[:5]):
        print('Features: {}\tClass: {}'.format(f,c))

    pca = sklearn.decomposition.PCA(n_components=2)
    iris_2d = pca.fit_transform(iris_x)

    iris_classes = iris_info.features['label'].names
    colors = ['.r', '.g', '.b']

    # iris dataset with 2 components and the class labels in colors
    plt.figure()
    plt.title('Iris Dataset')
    for i in range(iris_info.features['label'].num_classes):
        plt.plot(*iris_2d[np.where(iris_y==i)].T, colors[i], label=iris_classes[i])
    plt.legend(loc='best')
    plt.show()

    
    single_layer_model = tf.keras.Sequential(name='single_layer')

    print("IRIS FEATURES")
    print(iris_info.features['features'].shape)

    single_layer_model.add(tf.keras.layers.Input(iris_info.features['features'].shape))
    single_layer_model.add(tf.keras.layers.Dense(iris_info.features['label'].num_classes, activation='softmax', name='output'))

    single_layer_model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # show model summary (output shape, total params, etc)
    single_layer_model.summary()

    # save initial weights - bias and Wo ?? but no inital examples yet, so what does the model really builts?
    single_layer_model.save_weights('single_layer_init.h5')
    # train neural network (backprop and all)
    single_layer_train = single_layer_model.fit(iris_x, iris_y, epochs=100, batch_size=32)

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(20,7))

    # loss or error through cross entropy
    loss_ax.set_title('Loss')
    loss_ax.plot(single_layer_train.history['loss'], '-r', label='Train')

    acc_ax.set_title('Accuracy')
    # accuracy is done through comparing the train model output classes label and the original ones
    acc_ax.plot(single_layer_train.history['accuracy'], '-r', label='Train')

    plt.legend(loc=4)
    plt.show()

    # use 5 of the first examples and do forward propagation to get predictions
    predictions = single_layer_model.predict(iris_x[:5])
    print(predictions)

    np.argmax(predictions, axis=1)

    # best accuracy over the epochs
    loss, acc = single_layer_model.evaluate(iris_x, iris_y)
    print('Accuracy: {}'.format(acc))

    '''
    # SPLIT DATASET INTO TRAIN AND VALIDATION PART
    '''

    # get back to initial weights without having the training data
    single_layer_model.load_weights('single_layer_init.h5')

    # take care for the data not to be ordered by class - validation split takes the last % elements. Use validation_data parameter if you want to provide your validation set instead
    single_layer_train = single_layer_model.fit(iris_x, iris_y, validation_split=0.2, epochs=100, batch_size=32)


    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(20,7))

    loss_ax.set_title('Loss')
    loss_ax.plot(single_layer_train.history['loss'], '-r', label='Train')
    loss_ax.plot(single_layer_train.history['val_loss'], '-g', label='Validation')

    acc_ax.set_title('Accuracy')
    acc_ax.plot(single_layer_train.history['accuracy'], '-r', label='Train')
    acc_ax.plot(single_layer_train.history['val_accuracy'], '-g', label='Validation')

    plt.legend(loc=4)
    plt.show()

    # load initial weights again to use a strategy that stops the fitting by epochs when a certain accuracy isn't improved after a number of tries - called patience
    single_layer_model.load_weights('single_layer_init.h5')

    '''
    Configure callbacks for fitting the model again
    '''

    # use validation set accuracy on this callback.
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, verbose=1)

    # save best model parameters/weights whenever validation set accuracy improves - using callback also as in previous instruction
    checkpoint = tf.keras.callbacks.ModelCheckpoint('single_layer_best.h5', monitor='val_accuracy', verbose=1, save_best_only=True)

    # use a huge nr of epochs
    single_layer_train = single_layer_model.fit(iris_x, iris_y, validation_split=0.2, callbacks=[earlystop, checkpoint], epochs=10000, batch_size=32)



    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(20,7))

    loss_ax.set_title('Loss w patience strategy')
    loss_ax.plot(single_layer_train.history['loss'], '-r', label='Train')
    loss_ax.plot(single_layer_train.history['val_loss'], '-g', label='Validation')

    acc_ax.set_title('Accuracy  w patience strategy')
    acc_ax.plot(single_layer_train.history['accuracy'], '-r', label='Train')
    acc_ax.plot(single_layer_train.history['val_accuracy'], '-g', label='Validation')

    plt.legend(loc=4)
    plt.show()