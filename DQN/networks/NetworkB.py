from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
import tensorflow as tf


class NetworkB:    
    def buildModel(self, input_shape, learning_rate, action_space_len):
        model = Sequential()
        model.add(Conv2D(filters=8, kernel_size=(7, 7), strides=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(action_space_len, activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate, epsilon=1e-7))
        return model
