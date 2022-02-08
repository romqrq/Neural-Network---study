import os
import cv2 # computer vision - used for image processing
import numpy as np # numpy arrays
import matplotlib.pyplot as plt # digits visualization
import tensorflow as tf # machine learning part

# Get labeled data - Usually needs to separate into training and testing data. 
# Loading this way it already comes separated.
# mnist = tf.keras.datasets.mnist
# # x => Pixel data(hand written digit itself), Y =Classification(the equivalent number)
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Normalization - Getting all values to fit between 0 and 1
# # Here we want only to normalize the pixel data
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# # Create neural network model - commented once the model was created and saved
# model = tf.keras.models.Sequential()

# # Adding layers to the model
# # Flatten layer => converts a 28x28 pixels input into 784 x 1 pixel
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# # Dense layer => most basic layer. Needs to select an activation function. It can be activation=tf.nn."chosen activation function" or activation='string'
# # relu = rectifying linear unit
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# # softmax => makes sure that all outputs add up to 1. Gives the probability for each digit to be the answer
# model.add(tf.keras.layers.Dense(10, activation='softmax')) # This one is going to be an output layer, the 10 units represents the 10 digits

# # Compile the model, choose an optimizer, choose loss function and the metrics we are interested in.
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Fit/Train the model. Once this step is done, we should have a fully working model.
# model.fit(x_train, y_train, epochs = 3)

# # Saving this model here Allows us to just load it when we need instead of going over the training all over again.
# model.save('handwrite.model')


model = tf.keras.models.load_model('handwrite.model')

# # Evaluate the model - commented after evaluation
# loss, accuracy = model.evaluate(x_test, y_test)

# print(loss)
# print(accuracy)

# For this application, we can create our own images writing them in a piece of paper, an image processing program...as long as we resize it to a 28x28 pixels.
# Images saved in /digits

image_number = 1
while os.path.isfile(f'digits/digit{image_number}.png'):
    try:
        # here we aren't caring about colors, only the shape, we only get the first channel [:,:,0]
        img = cv2.imread(f'digits/digit{image_number}.png')[:,:,0]
        # we invert the image from black in white to white in black
        # we need to pass the image itself in a list as a numpy array to the neural network 
        img = np.invert(np.array([img]))
        # prediction: will give us the activation for digits neurons. not the result
        prediction = model.predict(img)
        print(f'This digit is probably a {np.argmax(prediction)}')
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        # look into kinds of exception expected
        print('Error!')
    finally:
        image_number += 1
    
