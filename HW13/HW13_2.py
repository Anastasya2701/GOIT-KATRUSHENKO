import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


x_train = tf.image.grayscale_to_rgb(tf.expand_dims(x_train, -1))
x_test = tf.image.grayscale_to_rgb(tf.expand_dims(x_test, -1))


x_train = tf.image.resize(x_train, (32, 32))
x_test = tf.image.resize(x_test, (32, 32))


x_train = x_train / 255.0
x_test = x_test / 255.0


vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
vgg_base.trainable = False  


model = Sequential([
    vgg_base,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))


test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Точність на тестових даних: {test_accuracy * 100:.2f}%')


model.save("vgg16_fashion_mnist.h5")