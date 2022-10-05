"""
@author: vinay
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from keras.layers import MaxPool2D, Conv2D, Flatten, BatchNormalization, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras import regularizers
#from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt

# Reading whole data saved in npz format
EEG_data_and_labels = np.load("C:/Users/vinay/Downloads/FEIS_v1_1/experiments/combined_data.npz")

# creating instance of one-hot-encoder
encoder = OneHotEncoder(handle_unknown='ignore')

# perform one-hot encoding on 'team' column 
one_hot_encoded_labels = encoder.fit_transform(EEG_data_and_labels["labels"].reshape(-1, 1)).toarray()

X_train, X_test, y_train, y_test = train_test_split(EEG_data_and_labels["data"], one_hot_encoded_labels, test_size=0.2, random_state=42)

# Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
adam = Adam(learning_rate=0.00001,beta_1=0.9,beta_2=0.999,epsilon=1e-07)

# The Sequential model is a linear stack of layers.
tf.keras.backend.clear_session()
model = Sequential()

# the authors of Batch Normalization say that It should be applied immediately before the non-linearity of the current layer
# Use the keyword argument input_shape when using this layer as the first layer in a model.
model.add(BatchNormalization(input_shape=(14,256,1)))

# detailed explanation https://keras.io/api/layers/convolution_layers/convolution2d/
model.add(Conv2D(32, kernel_size = (1,4), activation='relu'))
model.add(Conv2D(25, kernel_size = (14,1), activation='relu'))

# detailed explanation https://keras.io/api/layers/pooling_layers/max_pooling2d/
model.add(MaxPool2D(pool_size=(1, 4) , strides= 3))

model.add(Conv2D(50, kernel_size = (1,4), activation='relu'))
#model.add(Reshape((50,6,1), input_shape=(1,6,50)))
#model.add(Reshape((50,78,1), input_shape=(1,78,50)))
#model.add(MaxPool2D(pool_size=(1,3) , strides= None))
#model.add(Conv2D(100, kernel_size = (50,2), activation='relu'))

# Flatten the inputs
model.add(Flatten())

# detailed explanation https://keras.io/api/layers/core_layers/dense/
model.add(Dense(500, activation = 'relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))

model.add(Dropout(0.5)) # Fraction of the input units to drop
model.add(BatchNormalization())
model.add(Dense(128, activation = 'relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))

model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(32, activation = 'relu'))

model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(16, activation = 'softmax'))

# Configures the model for training.
model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics=['accuracy'])

# Creating instance of CSVlogger
csv_logger = CSVLogger("./model_history/model_acc_loss_1.csv")

# Saving model layers in text file
with open('./model_history/model_summary_1.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# Fit the model
history = model.fit(X_train, y_train, validation_data =(X_test, y_test) , callbacks=[csv_logger], verbose=1, epochs = 10, batch_size = 50)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""
filepath="/model_history/net_dc.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history=model.fit(X_train, y_train, validation_data =(X_test, y_test) , callbacks=callbacks_list, verbose=1, epochs = 5, batch_size = 50)
#Load and evaluate the best model version
model = load_model(filepath)
yhat = model.predict(X_test)
print('Model MSE on test data = ', mse(y_test, yhat).numpy())

from tensorflow.keras import callbacks
callbacks.ModelCheckpoint(
     filepath='model.{epoch:02d}-{val_loss:.4f}.h5', 
     save_freq='epoch', verbose=1, monitor='val_loss', 
     save_weights_only=True, 
)

checkpoint_filepath = './checkpoints2/checkpoint_default'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, mode="auto", save_freq=1, save_weights_only=True)

"""