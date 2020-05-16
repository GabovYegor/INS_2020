import sys
import numpy
import tensorflow.keras.callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

def build_model():
    model = Sequential()
    model.add(LSTM(250, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.25))
    model.add(Dense(y.shape[1], activation='softmax'))
    return model

def generateSequence(model):
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    print("Result:")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    for i in range(1000):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = numpy.argmax(prediction)
        result = int_to_char[index]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

class Callback(tensorflow.keras.callbacks.Callback):
    def __init__(self, epochs): 
        super(Callback, self).__init__()
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs={}):
        if epoch in self.epochs:
            generateSequence(model)

filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Patterns: ", n_patterns)

model = build_model()
model.compile(loss='categorical_crossentropy', optimizer='adam')

path= "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks = [checkpoint, Callback([0, 5, 10, 15])]

model.fit(numpy.reshape(dataX, (n_patterns, seq_length, 1)) / float(n_vocab), np_utils.to_categorical(dataY), epochs=15, batch_size=100, callbacks=callbacks)
