
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)

# the data, split between train and test sets
X = np.load("mydata.npy")
y = np.load("mydatay.npy")
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# n=5000
#x_train = x_train[1:n]; y_train=y_train[1:n]
#x_test=x_test[1:500]; y_test=y_test[1:500]

# Scale images to the [0, 1] range
min_max_scaler = preprocessing.MinMaxScaler()
#x_train = min_max_scaler.fit_transform(x_train)
#x_test = min_max_scaler.fit_transform(x_test)
y_train = min_max_scaler.fit_transform(y_train)
y_test = min_max_scaler.fit_transform(y_test)
x_train = (x_train.astype("float32") + 6) / 13
x_test = (x_test.astype("float32") + 6) / 13
print("orig x_train shape:", x_train.shape)

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

use_saved_model = False
if use_saved_model:
    model = keras.models.load_model("chess.model")
else:
    model = keras.Sequential()
    model.add(Conv2D(16, (3, 3), padding='same',
                     input_shape=x_train.shape[1:], activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(4, (3, 3), padding='same', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(4, (1, 1), padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(15, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(3, activation='relu',
                    kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='sgd',
                  loss='mse',
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    model.summary()

    batch_size = 142
    epochs = 20
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs, validation_split=0.2)
    model.save("chess.model")
    plt.subplot(211)
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model mean_squared_error')
    plt.ylabel('mean_squared_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

from sklearn.metrics import r2_score

print("Train data")
ypreds = model.predict(x_train)
print(mean_squared_error(ypreds, y_train))
print(r2_score(y_train,ypreds))


print("Test data")
ypreds = model.predict(x_test)
print(y_test)
print(ypreds)
print(mean_squared_error(ypreds, y_test))
print(r2_score( y_test,ypreds))


score, mse = model.evaluate(x_test, y_test,
                            batch_size=128)
print('Test score:', score)
print('Test mse:', mse)



