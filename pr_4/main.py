import numpy as np
from keras.layers import Dense
from keras.models import Sequential

def relu(x):
    return np.maximum(x, 0.)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logic_operation(a, b, c) -> int:
    return (a and b) or c

def matrix_logic_operation(train_data):
    return np.array([logic_operation(*i) for i in train_data])

def tensor_result(dataset, weights):
    result = dataset.copy()
    layers = [relu for i in range(len(weights) - 1)]
    layers.append(sigmoid)
    for i in range(len(weights)):
        result = layers[i](np.dot(result, weights[i][0]) + weights[i][1])
    return result

def each_element_of_tensor_result(dataset, weights):
    result = dataset.copy()
    layers = [relu for i in range(len(weights) - 1)]
    layers.append(sigmoid)
    for weight in range(len(weights)):
        len_current_weight = len(weights[weight][1])
        step_result = np.zeros((len(result), len_current_weight))
        for i in range(len(result)):
            for j in range(len_current_weight):
                sum = 0
                for k in range(len(result[i])):
                    sum += result[i][k] * weights[weight][0][k][j]
                step_result[i][j] = layers[weight](sum + weights[weight][1][j])
        result = step_result
    return result


def smart_print(model, dataset):
    weights = [layer.get_weights() for layer in model.layers]
    print("Result of tensor calculation:")
    print(tensor_result(dataset, weights))
    print("The result of the elementwise calculation:")
    print(each_element_of_tensor_result(dataset, weights))
    print("The result of the run through the trained model:")
    print(model.predict(dataset))

def custom_print(model):
    weights = [layer.get_weights() for layer in model.layers]
    def a(x):
        print(x[0][0])
        return 1 if x[0][0] > .5 else 0
    datas = [
        np.array([[1, 1, 1]]),
        np.array([[1, 0, 1]]),
        np.array([[0, 0, 1]])
    ]
    for data in datas:
        print('------START ITERATION {}-------'.format(data))
        print('Keras model:', a(model.predict(data)))
        print('Tensor model:', a(tensor_result(data, weights)))
        print('each element:', a(each_element_of_tensor_result(data, weights)))
        print('----END ITERATION------')

train_data = np.array([ [0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 0],
                        [1, 1, 1]])
validation_data = matrix_logic_operation(train_data)

model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(3,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print('NOT fitting')
smart_print(model, train_data)
custom_print(model)
print('fitting')
model.fit(train_data, validation_data, epochs=100, batch_size=1)
smart_print(model, train_data)
custom_print(model)
print(matrix_logic_operation(train_data))
