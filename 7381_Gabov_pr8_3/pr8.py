import tensorflow.keras.callbacks
import gens
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

values_test = []

def getData(size=500, img_size=50):
    X, y = gen_data(size, img_size)
    X, y = shuffle(X, y)#Выборки были не перемешаны
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=11)
    return (X_trn, y_trn), (X_tst, y_tst)

class CallBack(tensorflow.keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        print(self.values)
        plt.plot(self.values)
        plt.title('Number of observations with accuracy <90 on train')
        plt.ylabel('Count')
        plt.xlabel('epoch')
        plt.savefig('on_train_plot.png')
        plt.show()

        plt.plot(values_test)
        plt.title('Number of observations with accuracy <90 on test')
        plt.ylabel('Count')
        plt.xlabel('epoch')
        plt.savefig('on_tests_plot.png')
        plt.show()

    def on_epoch_end(self, epoch, logs=None):
        count = 0
        predicted = self.model.predict(self.x_train)
        for i in range(len(predicted)):
            index = np.argmax(predicted[i])
            if self.y_train[i][index] == 1 and predicted[i][index] <= 0.90:
                count += 1
        self.values.append(count)
        count = 0
        predicted = self.model.predict(self.x_test)
        for i in range(len(predicted)):
            index = np.argmax(predicted[i])
            if self.y_test[i][index] == 1 and predicted[i][index] < 0.90:
                count += 1
        values_test.append(count)

    def __init__(self, x_test, y_test, x_train, y_train):
        super(CallBack, self).__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train
        self.values = []

batch_size = 20
epochs = 12

def singleModelPlots( history ):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    return

def gen_data(size=500, img_size=50):
    c1 = size // 2
    c2 = size - c1
    label_c1 = np.full([c1, 1], 'Horizontal')
    data_c1 = np.array([gens.gen_h_line(img_size) for i in range(c1)])
    label_c2 = np.full([c2, 1], 'Vertical')
    data_c2 = np.array([gens.gen_v_line(img_size) for i in range(c2)])
    data = np.vstack((data_c1, data_c2))
    label = np.vstack((label_c1, label_c2))
    return data, label

(X_train, y_train), (X_test, y_test) = getData()
"""
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, 50, 50)
    X_test = X_test.reshape(X_test.shape[0], 1, 50, 50)
    input_shape = (1, 50, 50)
else:
"""
#
X_train = X_train.reshape(X_train.shape[0], 50, 50, 1)
X_test = X_test.reshape(X_test.shape[0], 50, 50, 1)
input_shape = (50, 50, 1)

encoder = LabelEncoder()
Y_train = encoder.fit_transform(y_train)
Y_test = encoder.fit_transform(y_test)

Y_train = keras.utils.to_categorical(Y_train, 2)
Y_test = keras.utils.to_categorical(Y_test, 2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))#either V or H
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
history = model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, Y_test), callbacks = [CallBack(X_test, Y_test, X_train, Y_train)])
score = model.evaluate(X_train, Y_train, verbose=0)

print(history.history)

singleModelPlots(history)

print('Train loss:', score[0])
print('Train accuracy:', score[1])

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])