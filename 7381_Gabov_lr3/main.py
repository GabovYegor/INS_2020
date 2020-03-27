import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100

all_val_mae_history=[]
all_mae_history=[]
all_val_loss_history=[]
all_loss_history=[]

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    history=model.fit(partial_train_data, partial_train_targets, validation_data=(val_data,val_targets),epochs=num_epochs, batch_size=1, verbose=0).history

    val_mae_history=history['val_mae']
    val_loss_history=history['val_loss']

    all_val_loss_history.append(val_loss_history)
    all_val_mae_history.append(val_mae_history)

    plt.plot(range(num_epochs), val_mae_history,'b',label='Val mae')
    plt.title('k=' + str(i))
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.show()

val_mae=[np.mean([x[i] for x in all_val_mae_history]) for i in range(num_epochs)]
smooth_val_mae_history = smooth_curve(val_mae[10:])

plt.plot(range(1, len(smooth_val_mae_history)+1),smooth_val_mae_history,'b',label='Val mae')
plt.title('mean acc')
plt.xlabel('epochs')
plt.ylabel('mae')
plt.show()

val_loss=[np.mean([x[i] for x in all_val_loss_history]) for i in range(num_epochs)]

plt.plot(range(len(val_loss)),val_loss,'b',label='Val mae')
plt.title('loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

model=build_model()
model.fit(train_data,train_targets,epochs=30, batch_size=1,verbose=0)
val_mse,val_mae=model.evaluate(test_data,test_targets,verbose=0)
print(val_mae)