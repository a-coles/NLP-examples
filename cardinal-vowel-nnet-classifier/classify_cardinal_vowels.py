%matplotlib inline
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np.random.seed(123)

from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten


# Import vowels and make plot from 3
vowels = pd.read_csv('vowels.csv')

ah = vowels[vowels.Vowel == 'ah']
iy = vowels[vowels.Vowel == 'iy']
uw = vowels[vowels.Vowel == 'uw']

plt.scatter(iy.F1Steady, iy.F2Steady, color='red')
plt.scatter(ah.F1Steady, ah.F2Steady, color='blue')
plt.scatter(uw.F1Steady, uw.F2Steady, color='green')

plt.xlabel('F1')
plt.xlabel('F2')
plt.show()

# Configure model
model = Sequential()
model.add(Dense(12, input_dim=2, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics['accuracy'])

# Get training data
ah = pd.concat([ah.F1Steady, ah.F2Steady], axis=1)
iy = pd.concat([iy.F1Steady, iy.F2Steady], axis=1)
uw = pd.concat([uw.F1Steady, uw.F2Steady], axis=1)
x_train = pd.concat([ah, iy, uw], axis=0)
x_train = x_train.values

# Get training labels
y_ah = np.full(ah.shape[0], 0)
y_iy = np.full(iy.shape[0], 1)
y_uw = np.full(uw.shape[0], 2)
y_train = np.concatenate((y_ah, y_iy, y_uw), axis=0)
one_hot_labels = to_categorical(y_train, num_classes=4)

# Fit model
model.fit(x_train, one_hot_labels, epochs=75, batch_size=32) 