from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
import tensorflow as tf


class NetworkFast:    
    def buildModel(self, input_shape, learning_rate, action_space_len):
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=1, activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=2, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(action_space_len, activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
        return model
