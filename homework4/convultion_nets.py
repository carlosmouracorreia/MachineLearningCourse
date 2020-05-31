import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np

import sklearn.decomposition
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # 28 per 28 images with one color channel
    # 10 classes (1 to 9 numbers)
    mnist_data, mnist_info = tfds.load('mnist', with_info=True)
    print(mnist_info)

    # why divide by 255? RGB. (we get a number from 0 to 1) each RBG pixel is encoded with a color that goes from 0 to 255
    mnist_train_x = np.asarray([instance['image']/255 for instance in tfds.as_numpy(mnist_data['train'])])
    mnist_train_y = np.asarray([instance['label'] for instance in tfds.as_numpy(mnist_data['train'])])

    mnist_test_x = np.asarray([instance['image']/255 for instance in tfds.as_numpy(mnist_data['test'])])
    mnist_test_y = np.asarray([instance['label'] for instance in tfds.as_numpy(mnist_data['test'])])

   # tfds.show_examples(mnist_info, mnist_data['test'])
    '''

    mnist_baseline_model = tf.keras.Sequential(name='mnist_baseline')
    mnist_baseline_model.add(tf.keras.layers.Input(mnist_info.features['image'].shape))
    # convert from 2d with a color (3 input vectors to only one number)
    mnist_baseline_model.add(tf.keras.layers.Flatten(name='flatten'))
    mnist_baseline_model.add(tf.keras.layers.Dense(mnist_info.features['label'].num_classes, activation='softmax', name='output'))
    mnist_baseline_model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    mnist_baseline_model.summary()

    # were making patience pretty small here... opposed to the other exercices
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('mnist_baseline_best.h5', monitor='val_accuracy', verbose=1, save_best_only=True)
    mnist_baseline_model_train = mnist_baseline_model.fit(mnist_train_x, mnist_train_y, validation_split=0.2, callbacks=[earlystop,checkpoint], epochs=10000, batch_size=256)


    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(20,7))

    loss_ax.set_title('Loss')
    loss_ax.plot(mnist_baseline_model_train.history['loss'], '-r', label='Train')
    loss_ax.plot(mnist_baseline_model_train.history['val_loss'], '-g', label='Validation')

    acc_ax.set_title('Accuracy')
    acc_ax.plot(mnist_baseline_model_train.history['accuracy'], '-r', label='Train')
    acc_ax.plot(mnist_baseline_model_train.history['val_accuracy'], '-g', label='Validation')

    plt.legend(loc=4)
    plt.show()

    # load best model for validation data and evaluate it on the test set
    mnist_baseline_model.load_weights('mnist_baseline_best.h5')
    loss, acc = mnist_baseline_model.evaluate(mnist_test_x, mnist_test_y)
    print('Accuracy: {}'.format(acc))
    '''
    '''
        Convolutional layers, resulting in better accuracy now

        We're using pooling technique, using max.

        Conv hidden (with kernels?) -> Max Pooling -> Flatten image so last layer can digest it

        ConvOperation: Same padding used as the number of strides (which is one by default; one row/column extra of padding in the conv operation - see practical class 8 slides resolution)

        Pooling operation: window is square because (kernel_size is not defined?) only one number is defined.
        Strides parameter defaults (none is provided) to the size of the pooling window (no overlap happens ... !?)
    '''

    '''
    mnist_conv_model = tf.keras.Sequential(name='mnist_cnn')
    mnist_conv_model.add(tf.keras.layers.Input(mnist_info.features['image'].shape))
    mnist_conv_model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=4, activation='relu', padding='same', name='convolution'))
    mnist_conv_model.add(tf.keras.layers.MaxPool2D(pool_size=2, name='pooling'))
    mnist_conv_model.add(tf.keras.layers.Flatten(name='flatten'))
    mnist_conv_model.add(tf.keras.layers.Dense(mnist_info.features['label'].num_classes, activation='softmax', name='output'))
    mnist_conv_model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    mnist_conv_model.summary()

    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('mnist_conv_best.h5', monitor='val_accuracy', verbose=1, save_best_only=True)

    mnist_conv_model_train = mnist_conv_model.fit(mnist_train_x, mnist_train_y, validation_split=0.2, callbacks=[earlystop,checkpoint], epochs=10000, batch_size=256)

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(20,7))

    loss_ax.set_title('Loss')
    loss_ax.plot(mnist_conv_model_train.history['loss'], '-r', label='Train')
    loss_ax.plot(mnist_conv_model_train.history['val_loss'], '-g', label='Validation')

    acc_ax.set_title('Accuracy')
    acc_ax.plot(mnist_conv_model_train.history['accuracy'], '-r', label='Train')
    acc_ax.plot(mnist_conv_model_train.history['val_accuracy'], '-g', label='Validation')

    plt.legend(loc=4)
    plt.show()

    mnist_conv_model.load_weights('mnist_conv_best.h5')
    loss, acc = mnist_conv_model.evaluate(mnist_test_x, mnist_test_y)
    print('Accuracy: {}'.format(acc))
    '''


    '''


    DROPOUT - randomly remove some layer neurons in each step of the training pahse


    consequently, training accuracy decreases.

                validation naccuaracy might improve, usually by chance.
                
                starting from the 0th epoch:
                    accuracy start already with value bigger than 0 for validation set
                    loss start already with a value smaller than half of the training loss in the beginning

'''


mnist_conv_drop_model = tf.keras.Sequential(name='mnist_cnn_dropout')
mnist_conv_drop_model.add(tf.keras.layers.Input(mnist_info.features['image'].shape))
mnist_conv_drop_model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=4, activation='relu', padding='same', name='convolution'))
mnist_conv_drop_model.add(tf.keras.layers.MaxPool2D(pool_size=2, name='pooling'))
mnist_conv_drop_model.add(tf.keras.layers.Dropout(0.5, name='dropout'))
mnist_conv_drop_model.add(tf.keras.layers.Flatten(name='flatten'))
mnist_conv_drop_model.add(tf.keras.layers.Dense(mnist_info.features['label'].num_classes, activation='softmax', name='output'))
mnist_conv_drop_model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
mnist_conv_drop_model.summary()


earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
checkpoint = tf.keras.callbacks.ModelCheckpoint('mnist_conv_drop_best.h5', monitor='val_accuracy', verbose=1, save_best_only=True)

mnist_conv_drop_model_train = mnist_conv_drop_model.fit(mnist_train_x, mnist_train_y, validation_split=0.2, callbacks=[earlystop,checkpoint], epochs=10000, batch_size=256)

fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(20,7))

loss_ax.set_title('Loss')
loss_ax.plot(mnist_conv_model_train.history['loss'], '-r', label='Train')
loss_ax.plot(mnist_conv_model_train.history['val_loss'], '-g', label='Validation')

acc_ax.set_title('Accuracy')
acc_ax.plot(mnist_conv_model_train.history['accuracy'], '-r', label='Train')
acc_ax.plot(mnist_conv_model_train.history['val_accuracy'], '-g', label='Validation')

plt.legend(loc=4)
plt.show()

mnist_conv_drop_model.load_weights('mnist_conv_drop_best.h5')
loss, acc = mnist_conv_drop_model.evaluate(mnist_test_x, mnist_test_y)
print('Accuracy: {}'.format(acc))