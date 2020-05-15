import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models
from keras.datasets import imdb
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence

def textPreparing(func):
    def FWrapper(*args, **kwargs):
        with open(args[0]) as f:
            text = f.read().lower()
            text = text_to_word_sequence(text)
        indexes = imdb.get_word_index()
        text_indexes = []
        for el in text:
            if el in indexes and indexes[el] < 10000:
                text_indexes.append(indexes[el])
        text_indexes = sequence.pad_sequences([text_indexes], maxlen=max_review_length)
        return func(text_indexes, **kwargs)
    return FWrapper

def predict_for_text(text):
    results = []
    for model in models:
        model.fit(train_x, train_y, epochs=2, batch_size=64, validation_data=(test_x, test_y),)
        results.append(model.predict(text))
    print(sum(results) / 2)

def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def build_model_1():
    model = models.Sequential()
    model.add(layers.Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(100))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def build_model_2():
    model = models.Sequential()
    model.add(layers.Embedding(top_words, embedding_vector_length, input_length=max_review_length))
    model.add(layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_length=2))
    model.add(layers.LSTM(100))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def printModel(history, type):
    if (type == "accuracy"):
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Training and validation accuracy")
        plt.ylabel("Accuracy")
    if (type == "loss"):
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Training and validation loos")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    plt.show()

(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=10000)
targets = np.concatenate((train_y, test_y), axis=0)
data = np.concatenate((train_x, test_x), axis=0)

max_review_length = 500
test_x = sequence.pad_sequences(test_x, maxlen=max_review_length)
train_x = sequence.pad_sequences(train_x, maxlen=max_review_length)

top_words = 10000
embedding_vector_length = 32

models = [build_model_1(), build_model_2()]
ensamble_acc = 0
modelsSize = 0
for model in models:
    modelsSize += 1
    ensamble_acc += model.evaluate(test_x, test_y, verbose=0)[1]
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=3, batch_size=50)
    print(model.evaluate(test_x, test_y, verbose=0))
    print("Test-Accuracy:", np.mean(history.history["val_accuracy"]))
    printModel(history, "accuracy")
    printModel(history, "loss")

print("Ensemble accuracy:")
print(ensamble_acc / modelsSize)
predict_for_text("test.txt")
