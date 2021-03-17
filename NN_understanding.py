import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATASET_PATH = r'C:\Users\hirom\Documents\Master Degree - Data Science\Group5_DeepLearning'

def load_data(data_path):
    with open(data_path) as fp:
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
    model = keras.Sequential([
        #flatten the dimension of a input (2D)
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        #avoid vanishing gradient (relu function)
        keras.layers.Dense(512, activation = 'relu'), #effective for training the data (faster)
        keras.layers.Dense(256, activation = 'relu'),
        keras.layers.Dense(64, activation = 'relu'),
        #output layer (softmax as classifier)
        keras.layers.Dense(10, activation='softmax') 
    ])
#compile the network
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
            loss='sparse_categorical_crossententropy',
            metric=['accuracy'])

model.summary()

#train 
model.fit(inputs_train,targets_train,
            validation_data=(inputs_test, targets_test),
            epochs = 50,
            batch_size=32)   #way the NN is trained 

