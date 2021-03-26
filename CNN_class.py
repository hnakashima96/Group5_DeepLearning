#Libraries
import numpy as np
import pandas as pd
from keras import layers 
from keras import models 
from keras import optimizers 

train_dir = r'C:\Users\hirom\Documents\GitHub\Group5_DeepLearning\CNN_example\data\train'
test_dir = r'C:\Users\hirom\Documents\GitHub\Group5_DeepLearning\CNN_example\data\test'

#model
model = models.Sequential()
model.add(layers.Conv2D(32,(2,2), activation = 'relu', input_shape = (150,150,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(32, (2,2), activation = 'relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (2,2), activation = 'relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(10, activation='softmax'))

#compile 
model.compile(loss='categorical_crossentropy', 
              optimizer = 'rmsprop',
             metrics = ['acc'])

from keras.preprocessing.image import ImageDataGenerator

#split the data in train and test set 
train_datagen = ImageDataGenerator(rescale =1./255)
test_datagen = ImageDataGenerator(rescale =1./255)
#255 is to rescale the pixel in the range of 0 and 1

train_generator = train_datagen.flow_from_directory(train_dir,
                                                   target_size = (150,150),
                                                   batch_size=70,
                                                   class_mode='categorical') 

validation_generator = test_datagen.flow_from_directory(test_dir,
                                                       target_size=(150,150),
                                                       batch_size=70,
                                                       class_mode='categorical')


STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

#fit the model 
history_cnn = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=150
)

#visualization
import matplotlib.pyplot as plt 
#print(history_track.history.keys())
                    
acc = history_cnn.history['acc']
val_acc = history_cnn.history['val_acc']
loss = history_cnn.history['loss']
val_loss = history_cnn.history['val_loss']

epochs = range(1, len(acc)+1)

fig, (ax0, ax1) = plt.subplots(1,2, figsize=(15,5))

ax0.plot(epochs, loss, 'bo', label = 'Loss')
ax0.plot(epochs, val_loss, 'b', label = 'Val_loss')
ax0.set_title('Training Loss')
ax0.set_xlabel('Epochs')
ax0.set_ylabel('Loss')
ax0.legend()

ax1.plot(epochs, acc, 'bo', label = 'Acc')
ax1.plot(epochs, val_acc, 'b', label = 'Val_Acc')
ax1.set_title('Training Acc')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Acc')
ax1.legend()

plt.show()