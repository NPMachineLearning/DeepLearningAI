from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.datasets import fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = Sequential([
    layers.Conv2D(16, kernel_size=(3, 3), input_shape=(28, 28, 1)),
    layers.MaxPool2D(2, 2),
    layers.Conv2D(16, kernel_size=(3, 3)),
    layers.MaxPool2D(2, 2),
    layers.Conv2D(16, kernel_size=(3, 3)),
    layers.MaxPool2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

class MyCallback(Callback):
    def __init__(self, threshold=0.1):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if logs.get("loss") < self.threshold:
            print(f"\nReach {self.threshold*100:.2f}% loss and cancel training!")
            self.model.stop_training=True

callbacks = MyCallback(0.1)

model.fit(training_images,
          training_labels,
          epochs=20,
          callbacks=[callbacks])

model.evaluate(test_images, test_labels)

model.save("./cnn_l3.h5")