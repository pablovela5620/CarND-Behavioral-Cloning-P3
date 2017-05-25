from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Flatten, MaxPooling2D, Dropout, Dense, Convolution2D
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from helper_functions import generator, valid_generator

CSV_PATH = "data/driving_log.csv"


def load_data(csv_path):
    '''
    Loads data from csv file then splits into training and validation set
    :param csv_path: Path to csv file
    :return: The train validation split for X and y
    '''
    data_df = pd.read_csv(csv_path)
    X = data_df.loc[:, ('center', 'left', 'right')].values
    y = data_df.loc[:, 'steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=0)
    return X_train, X_valid, y_train, y_valid, data_df


X_train, X_valid, y_train, y_valid, data_df = load_data(CSV_PATH)

train_generator = generator(X_train, y_train)
valid_generator = valid_generator(X_valid, y_valid)

'''
Model architecture based on NVIDIA end to end self driving training, modified to include a Lamda layer
for normalization and dropout layers to avoid overfitting
'''
model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(66, 200, 3)))
model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(y_train), nb_epoch=3, validation_data=valid_generator,
                    nb_val_samples=len(y_valid))
model.save('model.h5')
