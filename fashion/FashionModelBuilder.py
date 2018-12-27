# THIS CLASS:
# - DOWNLOADS PICTURE DATA
# - CREATES MODEL and TRAIN MODEL
# - SAVES MODEL TO FILE

print("# 1")
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras

print("# 2")
# Download training data (images, labels):
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("# 3")
# CHECK DATASET:
print("train images shape: " + str(train_images.shape))
print("train labels number: " + str(len(train_labels)))
print("train labels: " + str(train_labels))

print("test images shape: " + str(test_images.shape))
print("test labels number: " + str(len(test_labels)))
print("test labels: " + str(test_labels))

# PRE-PROCESS DATA
# The data must be preprocessed before training the network.
# If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:
print("# 4")
print("PRE-PROCESSING DATA")
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.savefig('model\pictures\pic01_inspect_first_image.png')

# It's important that the training set and the testing set are preprocessed in the same way:
train_images = train_images / 255.0
test_images = test_images / 255.0

print("# 5")
print("BUILD THE MODEL")
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

print("# 6")
print("START MODEL TRAINING...")
model.fit(train_images, train_labels, epochs=5)

print("# 7")
print("START MODEL EVALUATION...")
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
predictions = model.predict(test_images)


# save the model to disk
filename = 'model/fashion_trained_model.h5'
model.save(filename)
print("saved model to file: " + str(filename))