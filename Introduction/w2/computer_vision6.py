import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import Callback

(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# Early stop training callback for accuracy
class MyCallbacks(Callback):
    def __init__(self, threshold=0.6):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if(logs.get("accuracy") >= self.threshold):
            print(f"\nReached {self.threshold*100:.2f}% accuracy, cancelling training!")
            self.model.stop_training=True

callbacks = MyCallbacks(0.9)

model.fit(training_images,
          training_labels,
          epochs=40,
          callbacks=[callbacks])

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])