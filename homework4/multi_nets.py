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


    multi_layer_model = tf.keras.Sequential(name='multi_layer')
    multi_layer_model.add(tf.keras.layers.Input(iris_info.features['features'].shape))
    multi_layer_model.add(tf.keras.layers.Dense(256, activation='tanh', name='hidden'))
    multi_layer_model.add(tf.keras.layers.Dense(iris_info.features['label'].num_classes, activation='softmax', name='output'))

    multi_layer_model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    multi_layer_model.summary()

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('multi_layer_best.h5', monitor='val_accuracy', verbose=1, save_best_only=True)

    multi_layer_train = multi_layer_model.fit(iris_x, iris_y, validation_split=0.2, callbacks=[earlystop,checkpoint], epochs=10000, batch_size=32)

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(20,7))

    loss_ax.set_title('Loss')
    loss_ax.plot(multi_layer_train.history['loss'], '-r', label='Train')
    loss_ax.plot(multi_layer_train.history['val_loss'], '-g', label='Validation')

    acc_ax.set_title('Accuracy')
    acc_ax.plot(multi_layer_train.history['accuracy'], '-r', label='Train')
    acc_ax.plot(multi_layer_train.history['val_accuracy'], '-g', label='Validation')

    plt.legend(loc=4)
    plt.show()

    '''
    USING REGULARIZATION NOW - L2 on the weights of the hidden layer

    When adding layers to the network, we can also include regularization in those layers using three different parameters: kernel_regularizer, bias_regularizer, and activity_regularizer. The first applies regularization to the weights of the layer, the second to its bias, and the last to its output.

    '''

    multi_layer_reg_model = tf.keras.Sequential(name='multi_layer_regularization')
    multi_layer_reg_model.add(tf.keras.layers.Input(iris_info.features['features'].shape))
    multi_layer_reg_model.add(tf.keras.layers.Dense(256, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.01), name='hidden'))
    # softmax activation function in the last layer so that crossentropy can be used (instead of normal tanh? review this)
    multi_layer_reg_model.add(tf.keras.layers.Dense(iris_info.features['label'].num_classes, activation='softmax', name='output'))

    # here instead of stochastic gradient descent, ADAM is used. What is this?
    multi_layer_reg_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    multi_layer_reg_model.summary()

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('multi_layer_reg_best.h5', monitor='val_accuracy', verbose=1, save_best_only=True)

    multi_layer_reg_train = multi_layer_reg_model.fit(iris_x, iris_y, validation_split=0.2, callbacks=[earlystop,checkpoint], epochs=10000, batch_size=32)

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(20,7))

    loss_ax.set_title('Loss')
    loss_ax.plot(multi_layer_reg_train.history['loss'], '-r', label='Train')
    loss_ax.plot(multi_layer_reg_train.history['val_loss'], '-g', label='Validation')

    acc_ax.set_title('Accuracy')
    acc_ax.plot(multi_layer_reg_train.history['accuracy'], '-r', label='Train')
    acc_ax.plot(multi_layer_reg_train.history['val_accuracy'], '-g', label='Validation')

    plt.legend(loc=4)
    plt.show()