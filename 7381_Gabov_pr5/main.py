from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def getData(n):
    data = np.zeros((n, 6))
    targets = np.zeros(n)
    for i in range(n):
        x = np.random.normal(0, 10)
        e = np.random.normal(0, 0.3)
        data[i, :] = (x ** 2 + x + e, np.abs(x) + e, np.sin(x - (np.pi / 4)) + e, np.log(np.abs(x)) + e, -x ** 3 + e, -x + e)
        targets[i] = -x / 4 + e
    return data, targets


def print_loss(name, H):
    plt.plot(H.history[name], 'y', label='train')
    plt.plot(H.history['val_' + name], 'g', label='validation')
    plt.title(name)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
    plt.clf()


def print_diff(name, one, two):
    plt.plot(one, 'r|')
    plt.plot(two, 'b*')
    plt.title(name)
    plt.ylabel('val')
    plt.legend()
    plt.show()
    plt.clf()


train_data, train_targets = getData(200)
test_data, test_targets = getData(20)

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std
test_data -= mean
test_data /= std

input = Input(shape=(6,), name='input')

# Encoder
enc = Dense(64, activation='relu')(input)
enc = Dense(32, activation='relu')(enc)
enc = Dense(8, activation='relu', name="encode")(enc)

# Decoder
dec = Dense(32, activation='relu')(enc)
dec = Dense(64, activation='relu')(dec)
dec = Dense(6, name='decode')(dec)

# Predictor
pred = Dense(64, activation='relu')(enc)
pred = Dense(32, activation='relu')(pred)
pred = Dense(16, activation='relu')(pred)
pred = Dense(1, name="predict")(pred)

# define model with 1 input && 2 output
model = Model(input=input, outputs=[pred, dec])
model.compile(optimizer='adam', loss='mse')
H = model.fit(x=train_data, y=[train_targets, train_data],
              epochs=150,
              batch_size=5,
              validation_data=(test_data, [test_targets, test_data]))

print_loss('loss', H)
print_loss('predict_loss', H)
print_loss('decode_loss', H)

pd.DataFrame(train_data).to_csv("input_data.csv")
pd.DataFrame(train_targets).to_csv("data_targets.csv")

encoder = Model(input, enc)
encoded_data = encoder.predict(test_data)
pd.DataFrame(encoded_data).to_csv("encoded_data.csv")
encoder.save('encoder.h5')

decoder = Model(input, dec)
decoded_data = decoder.predict(test_data)
print_diff('dec-test', decoded_data, test_data)
pd.DataFrame(decoded_data).to_csv("decoded_data.csv")
decoder.save('decoder.h5')

predictor = Model(input, pred)
predicted_data = predictor.predict(test_data)
print_diff('pred-targ', predicted_data, test_targets)
pd.DataFrame(predicted_data).to_csv("predicted_data.csv")
predictor.save('predictor.h5')