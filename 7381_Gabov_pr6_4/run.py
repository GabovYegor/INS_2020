from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from var4 import gen_data
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def printRes(type) :
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

def getData():
    data, labels = gen_data(size=1000)
    data, labels = shuffle(data, labels)
    dataTrain, dataTest, labelTrain, labelTest = train_test_split(data, labels, test_size=0.2, random_state=11)
    dataTrain = dataTrain.reshape(dataTrain.shape[0], 50, 50, 1)
    dataTest = dataTest.reshape(dataTest.shape[0], 50, 50, 1)

    encoder = LabelEncoder()
    encoder.fit(labelTrain)
    labelTrain = encoder.transform(labelTrain)
    labelTrain = to_categorical(labelTrain, 2)

    encoder.fit(labelTest)
    labelTest = encoder.transform(labelTest)
    labelTest = to_categorical(labelTest, 2)
    return dataTrain, labelTrain, dataTest, labelTest

def createModel():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='Adagrad',
                  metrics=['accuracy'])
    return model

dataTrain, labelTrain, dataTest, labelTest = getData()
model = createModel()
history = model.fit(dataTrain, labelTrain, batch_size=20, epochs=9, verbose=1, validation_data=(dataTest, labelTest))
score = model.evaluate(dataTest, labelTest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

printRes("acc")
printRes("loss")
