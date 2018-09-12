# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

# Define consts
DATADIR = '/Users/7w0r4ng3s/Desktop/sentdex_deep_learning/part_2/kagglecatsanddogs_3367a/PetImages'
CATEGORIES = ['Dog', 'Cat'] # labels
IMG_SIZE = 50

# print(img_array.shape)

# Create trianing dataset
training_data = []

def create_training_data():
    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)
        # path to cats or dogs directory
        path = os.path.join(DATADIR, category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                print("Error: " + str(e))

create_training_data()

print('Training data length: ', len(training_data))

# Shuffle the data
random.shuffle(training_data)
# Check the data is shuffled
for sample in training_data[:10]:
    print(sample[1]) # make sure the data is correctly shuffled

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open('X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

# # Check if X is saved correctly
# pickle_in = open('X.pickle', 'rb')
# X = pickle.load(pickle_in)
# print(X) # check if X is loaded correctly



X = pickle.load(open('X.pickle', 'rb'))
y = pickle.load(open('y.pickle', 'rb'))
X = X/255.0

# Building model
model = Sequential()

# Layer 1
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:])) # Conv layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
model.add(Conv2D(64, (3, 3))) # Conv layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

# Output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, validation_split=0.1)

# Save the model
model.save('cats_dogs_classifier.model')