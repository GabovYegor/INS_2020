from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
num_epochs = 30
kernel_size = 5
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
num_train, depth, height, width = X_train.shape
num_test = X_test.shape[0]
num_classes = np.unique(y_train).shape[0]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= np.max(X_train)
X_test /= np.max(X_train)
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

def buildModel(dropout):
    inp = Input(shape=(depth, height, width))
    conv_1 = Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu')(inp)
    conv_2 = Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu')(conv_1)
    if dropout:
        drop_1 = Dropout(drop_prob_1)(MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2))
    else:
        drop_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    conv_3 = Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu')(drop_1)
    conv_4 = Convolution2D(conv_depth_2, kernel_size, kernel_size, border_mode='same', activation='relu')(conv_3)
    if dropout:
        drop_2 = Dropout(drop_prob_1)(MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4))
    else:
        drop_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
    flat = Flatten()(drop_2)
    hidden = Dense(hidden_size, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(Dropout(drop_prob_2)(hidden))
    model = Model(input=inp, output=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def printRes(type):
    plt.clf()
    if(type == "acc") :
        plt.title("training and test accuracy")
        plt.plot(history.history['accuracy'], 'g', label='Training acc')
        plt.plot(history.history['val_accuracy'], 'b', label='Validation acc')
    if(type == "loss") :
        plt.title("training and test loss")
        plt.plot(history.history['loss'], 'g', label='Training loss')
        plt.plot(history.history['val_loss'], 'b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

model = buildModel(False)
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=num_epochs,verbose=1, validation_split=0.1)
model.evaluate(X_test, Y_test, verbose=1)
printRes("acc")
printRes("loss")