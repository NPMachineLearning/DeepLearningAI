import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

# load model
model1 = tf.keras.models.load_model("./cnn_l1.h5")
print(f"model 1 eval: {model1.evaluate(test_images, test_labels)}")
model2 = tf.keras.models.load_model("./cnn_l2.h5")
print(f"model 2 eval: {model2.evaluate(test_images, test_labels)}")
model3 = tf.keras.models.load_model("./cnn_l3.h5")
print(f"model 3 eval: {model3.evaluate(test_images, test_labels)}")

model = model2

print(model.summary())

# visualize convolutional and pooling
print(test_labels[:100])

FIRST_IMAGE = 0
SECOND_IMAGE = 1
THIRD_IMAGE = 2

f, ax = plt.subplots(1, 3)
ax[0].imshow(test_images[FIRST_IMAGE])
ax[0].set_title(test_labels[FIRST_IMAGE])
ax[1].imshow(test_images[SECOND_IMAGE])
ax[1].set_title(test_labels[SECOND_IMAGE])
ax[2].imshow(test_images[THIRD_IMAGE])
ax[2].set_title(test_labels[THIRD_IMAGE])
plt.show()

f, ax = plt.subplots(3, 4)
CONV_NUMBER = 14

layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input,
                                         outputs=layer_outputs)

for x in range(0, 4):
    f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
    ax[0, x].imshow(f1[0, :, :, CONV_NUMBER], cmap="inferno")
    ax[0, x].grid(False)

    f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    ax[1, x].imshow(f2[0, :, :, CONV_NUMBER], cmap="inferno")
    ax[1, x].grid(False)

    f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
    ax[2, x].imshow(f3[0, :, :, CONV_NUMBER], cmap="inferno")
    ax[2, x].grid(False)

plt.show()