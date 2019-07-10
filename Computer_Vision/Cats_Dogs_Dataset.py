from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop

#Defining structure of the model
model = Sequential()
model.add(Conv2D(16, (3,3), activation = 'relu', input_shape = (150,150,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32, (5,5), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(AveragePooling2D((2,2)))
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = RMSprop(lr = 0.005),
	loss = 'binary_crossentropy', metrics = ['accuracy'])

#ImageDataGenerator to read images and create dataset for training

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

dir(train_datagen)

#We will be using the flow_from_directory method to create a 
#generator object which will continuously give us batches of size
#batch_size specified below.

train_dir = "path"
validation_dir = "path"
train_generator = train_datagen.flow_from_directory(
	train_dir, target_size = (150, 150), batch_size = 20, 
	class_mode = 'binary')

valid_generator = train_datagen.flow_from_directory(
	validation_dir, target_size = (150, 150), batch_size = 20, 
	class_mode = 'binary')
#This will resize all the images to 28 X 28

history = model.fit_generator(train_generator,
	steps_per_epoch = 100, epochs = 15, 
	validation_data = valid_generator,
	validation_steps = 25)

#NOTE - Generator objects will keep on generating batches from the
#dataset they have been assigned. So, the steps_per_epoch argument
#tells how many times the generator will be called before using break
#to stop the generation per epoch.

#So, 100 batches of size 20 will be generated at each 
#epoch for training

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