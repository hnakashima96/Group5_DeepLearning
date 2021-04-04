#basic libraries 
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

#path of the json file preprocessed
DATASET_PATH = r'...\GitHub\Group5_DeepLearning\data_oficial.json'

#function to load the data to process 
def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)


    #convert a list into numpy array
    inputs = np.array(data['mfcc'])
    targets = np.array(data['labels'])

    return inputs,targets

if __name__ == '__main__':
    #load data
    inputs, targets = load_data(DATASET_PATH)
    
    #split the data 
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size = 0.3)

    # build the NN 
    model3 = keras.Sequential([
        #flatten the dimension of a input (2D)
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        #avoid vanishing gradient (relu function)
        keras.layers.Dense(512, activation = 'relu'), #effective for training the data (faster)
        keras.layers.Dense(256, activation = 'relu'),
        keras.layers.Dense(64, activation = 'relu'),
        keras.layers.Dense(32, activation = 'relu'),
        #output layer (softmax as classifier)
        keras.layers.Dense(10, activation='softmax') 
    ])
#compile the network
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.summary()

#train 
history_track = model.fit(inputs_train,targets_train,
            validation_data=(inputs_test, targets_test),
            epochs = 500,
            batch_size=32,
            verbose=1)   #way the NN is trained 

#Evaluation of the model as values
accr = model.evaluate(inputs_test,targets_test)

#visualization
import matplotlib.pyplot as plt 
#print(history_track.history.keys())

loss = history_track.history['loss']
val_loss = history_track.history['val_loss']
acc = history_track.history['accuracy']
val_acc = history_track.history['val_accuracy']

epochs = range(1, len(history_track.history['accuracy'])+1)

fig, (ax0, ax1) = plt.subplots(1,2)

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
