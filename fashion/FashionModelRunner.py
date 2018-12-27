import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras

# download dataset for model test experiments
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# It's important that the training set and the testing set are preprocessed in the same way:
train_images = train_images / 255.0
test_images = test_images / 255.0

# load prepared model from disk
filename = 'model/fashion_trained_model.h5'
loaded_model = keras.models.load_model(filename)
print("LOADED model from file: " + str(filename))

loaded_model.summary()
predictions = loaded_model.predict(test_images)

print("# 8")
print("TESTING MODEL - make predictions:")
print("label with highest confidence value: " + str(predictions[0]) + " -refer number to article table")
print("let's check test label if prediction was correct: " + str(test_labels[0]))

# *********************
# PICTURES INSPECTION (for visual reference purpose only)
# *********************
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.savefig("model/pictures/pic02_inspect_result_images.png")

# Let's look at 0th image
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.savefig("model/pictures/pic03_inspect_0th_image.png")

# Let's look at 12th image
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.savefig("model/pictures/pic04_inspect_12th_image.png")



# EXPERIMENTS WITH TRAINED MODEL:
# make a prediction to single image.
# image is grabbed from dataset
print("# 9")
print("EXPERIMENTS WITH SINGLE ITEM ON TRAINED MODEL")
print("List of first 10 TEST_LABELs (label is current fashion)")
print("take index 0-9 from label list to indicate fashion number:")
iteration = 0
for item in test_labels:
    print(str(item))
    iteration = iteration + 1
    if (iteration == 10):
        break

# set image to test
tested_image_index_from_label_list = 1
print("tested_image index from test_label_list: " + str(tested_image_index_from_label_list))
print("tested image number ------------------>: " + str(test_labels[tested_image_index_from_label_list]))

img = test_images[tested_image_index_from_label_list]
print("image shape: " + str(img.shape))

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))
print("image shape as the only one: " + str(img.shape))

# now predict single image:
predictions_single = loaded_model.predict(img)
print("predictions for single item: " + str(predictions_single))

print("class array index for max probable item: " + str(np.argmax(predictions_single[0])))

print("**********************************")
if(np.argmax(predictions_single[0])== test_labels[tested_image_index_from_label_list]):
    print("* RESULT: CORRECT PREDICTION :-) *")
else:
    print("* RESULT: WRONG PREDICTION :-( *")
print("**********************************")
print("")
print("# CLASS INDEX LEGEND:")
print(" 0	T-shirt/top")
print(" 1	Trouser")
print(" 2	Pullover")
print(" 3	Dress")
print(" 4	Coat")
print(" 5	Sandal")
print(" 6	Shirt")
print(" 7	Sneaker")
print(" 8	Bag")
print(" 9	Ankle boot")
print("END OF EXPERIMENT")
