import pandas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    dataframe = pandas.read_csv('sonar.csv', header=None)
    dataset = dataframe.values
    np.random.shuffle(dataset)
    X = dataset[:, 0:60].astype(float)
    Y = dataset[:, 60]
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded = encoder.transform(Y)
    model = Sequential()
    model.add(Dense(60, kernel_initializer='normal', activation='relu', input_dim=60))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X, encoded, epochs=100, batch_size=10, validation_split=0.1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()