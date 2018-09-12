# import dependencies
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import utils, Sequential, models
from keras.layers import Flatten, Dense
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# -------------------- Loading the data --------------------

# 28x28 images of hand written digist 0-9
mnist = tf.keras.datasets.mnist

# -------------------- Prepare the data --------------------
# unpack dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# original image
plt.imshow(x_train[0]) # add cmap=plt.cm.binary to make it black and white

# normalize the data so that it's easier for the network to learn
x_train = utils.normalize(x_train, axis=1)
x_test = utils.normalize(x_test, axis=1)

# image after scaling
plt.imshow(x_train[0])

# -------------------- Build the model --------------------

# building the model
model = Sequential()
model.add(Flatten())
# 128 nerons
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(10, activation=tf.nn.softmax))

# adam - default optimizer 
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# calculate validation loss
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# save model
model.save('num_classify.model')

# -------------------- Use the model --------------------

# load new models
new_model = models.load_model('num_classify.model')
predictions = new_model.predict([x_test])
# predict
print(np.argmax(predictions[1]))
# examine the image
plt.imshow(x_test[1])



