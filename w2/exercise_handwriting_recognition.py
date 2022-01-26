from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

plt.imshow(training_images[0])
plt.show()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

class MyCallback(Callback):
    def __init__(self, threshold=0.6):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if logs.get("loss") < self.threshold:
            print(f"\nReach {self.threshold*100:.2f}% accuracy, canceling training!")
            self.model.stop_training=True

callbacks = MyCallback(0.1)

model.fit(training_images,
          training_labels,
          epochs=20,
          callbacks=[callbacks])

model.evaluate(test_images, test_labels)