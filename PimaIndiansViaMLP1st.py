import numpy as np # linear algebra
import pandas as pd # data processing, reading CSV file
import theano
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from theano import tensor
from sklearn.model_selection import train_test_split
import os
seed = 7
# fix random seed for reproducibility
np.random.seed(seed)
# load pima indians dataset
dataset = pd.read_csv("pima-indians-diabetes-database\diabetes.csv")
# split the original dataset into input (X) and output (Y) variables
# pick the line data in the dataset
X = dataset.iloc[:,0:8].values
Y = dataset.iloc[:,8].values

# split into 50% for train and 50% for test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50, random_state=seed)

# create model
model = Sequential()
# 1st approach
# add each layer to the model 77.99% accuracy
model.add(Dense(32, input_dim=8, activation='relu')) # output matrix size (*, 32)
# model.add(Dense(28, activation='relu'))
model.add(Dense(24, activation='relu'))
# model.add(Dense(16, activation='relu')) #plus
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation = 'relu'))
# model.add(Dense(6,  activation='relu')) #plus
model.add(Dense(1, activation='sigmoid'))

#2nd approach 81.12% accuracy
# model.add(Dense(64, input_dim=8, activation='sigmoid')) # output matrix size (*, 32)
# model.add(Dense(16, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))

# 3rd approch
# model.add(Dense(32, input_dim=8, activation='relu')) # output matrix size (*, 32)
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=400, batch_size=40)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()