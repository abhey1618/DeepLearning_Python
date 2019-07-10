from keras.datasets import imdb
import keras
import numpy as np

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
np.load = np_load_old

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
	results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#In the previous notebook, we saw we were overfitting on our training dataset.

#Let us divide our training data to training and validation data.
x_val = x_train[:5000]
y_val = y_train[:5000]
x_train_1 = x_train[5000:]
y_train_1 = y_train[5000:]

#Defining a keras callback
class my_callback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get('acc') > 0.95):
            print("Stopping to prevent overfitting")
            self.model.stop_training = True

callback = my_callback()

from keras.layers import Input, Dense
from keras.models import Model

#Reduced the amount of neurons in first layer drastically and
#increased in the second layer a little.
X = Input(shape = (x_train.shape[1],))
Y = Dense(16, activation = 'relu')(X)
Y = Dense(16, activation = 'relu')(Y)
Y = Dense(1, activation = 'sigmoid')(Y)

model = Model(inputs = [X], outputs = [Y])
model.compile(optimizer = 'Adam', loss = 'binary_crossentropy',
              metrics = ['accuracy'])
history = model.fit(x_train_1, y_train_1, epochs = 15, callbacks = [callback],
          batch_size = 1024, validation_data = (x_val, y_val))
#Using batch_size trains our model with each gradient descent step trained on
#a random batch with 1024 samples. It is good to keep batch_size as a power
#of 2 because that helps us in utilising the power of GPU efficiently.

#model.fit returns a History object

hist_dict = history.history

import matplotlib.pyplot as plt

plt.figure(figsize=(20,8))

plt.subplot(1, 2, 1)
plt.plot(np.arange(len(hist_dict['acc'])), hist_dict['acc'], '.-', color = 'blue')
plt.plot(np.arange(len(hist_dict['val_acc'])), hist_dict['val_acc'], 'o-', color = 'green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy in Training v/s Validation')
plt.legend(('Training', 'Validation'))

plt.subplot(1, 2, 2)
plt.plot(np.arange(len(hist_dict['loss'])), hist_dict['loss'], '.-', color = 'blue')
plt.plot(np.arange(len(hist_dict['val_loss'])), hist_dict['val_loss'], 'o-', color = 'green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss in Training v/s Validation')
plt.legend(('Training', 'Validation'))

plt.show()

#To tackle overfitting, we can use the following ways:-

#Increase training data
#Reduce the network size
#Add weight regularization
#Adding Dropout

from keras.regularizers import l1, l2, l1_l2

X = Input(shape = (x_train.shape[1],))
Y = Dense(16, activation = 'relu', kernel_regularizer = l1(0.001))(X)
Y = Dense(16, activation = 'relu', kernel_regularizer = l1(0.002))(Y)
Y = Dense(1, activation = 'sigmoid')(Y)

model2 = Model(inputs = [X], outputs = [Y])
model2.compile(optimizer = 'Adam', loss = 'binary_crossentropy',
              metrics = ['accuracy'])
history2 = model2.fit(x_train_1, y_train_1, epochs = 15, callbacks = [callback],
          batch_size = 1024, validation_data = (x_val, y_val))

hist_dict = history2.history

plt.figure(figsize=(20,8))

plt.subplot(1, 2, 1)
plt.plot(np.arange(len(hist_dict['acc'])), hist_dict['acc'], '.-', color = 'blue')
plt.plot(np.arange(len(hist_dict['val_acc'])), hist_dict['val_acc'], 'o-', color = 'green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy in Training v/s Validation')
plt.legend(('Training', 'Validation'))

plt.subplot(1, 2, 2)
plt.plot(np.arange(len(hist_dict['loss'])), hist_dict['loss'], '.-', color = 'blue')
plt.plot(np.arange(len(hist_dict['val_loss'])), hist_dict['val_loss'], 'o-', color = 'green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss in Training v/s Validation')
plt.legend(('Training', 'Validation'))

plt.show()

from keras.layers import Dropout

X = Input(shape = (x_train.shape[1],))
Y = Dense(16, activation = 'relu')(X)
Y = Dropout(0.7)(Y)
Y = Dense(16, activation = 'relu')(Y)
Y = Dropout(0.5)(Y)
Y = Dense(1, activation = 'sigmoid')(Y)

model3 = Model(inputs = [X], outputs = [Y])
model3.compile(optimizer = 'Adam', loss = 'binary_crossentropy',
              metrics = ['accuracy'])
history3 = model3.fit(x_train_1, y_train_1, epochs = 15, callbacks = [callback],
          batch_size = 1024, validation_data = (x_val, y_val))

hist_dict = history3.history

plt.figure(figsize=(20,8))

plt.subplot(1, 2, 1)
plt.plot(np.arange(len(hist_dict['acc'])), hist_dict['acc'], '.-', color = 'blue')
plt.plot(np.arange(len(hist_dict['val_acc'])), hist_dict['val_acc'], 'o-', color = 'green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy in Training v/s Validation')
plt.legend(('Training', 'Validation'))

plt.subplot(1, 2, 2)
plt.plot(np.arange(len(hist_dict['loss'])), hist_dict['loss'], '.-', color = 'blue')
plt.plot(np.arange(len(hist_dict['val_loss'])), hist_dict['val_loss'], 'o-', color = 'green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss in Training v/s Validation')
plt.legend(('Training', 'Validation'))

plt.show()