import tensorflow as tf

print("TENSORFLOW demo with Keras (high level ML library)")
print("Training model based on dataset from http://yann.lecun.com/exdb/mnist/")

# 1) Download a dataset
print("DAWNLOAD dataset for model build...")
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# 1a) Reformat the images / labels (TODO)

# 2) Build model
print("BUILD model...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

print("COMPILE model...")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3) Evaluate model

print("TRAINING model...")
model.fit(x_train, y_train, epochs=5)

print("EVALUATE model...")
model.evaluate(x_test, y_test)
