import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

(X_train, y_train),(X_test, y_test) = keras.datasets.cifar10.load_data()
print(X_train.shape, y_train.shape)

class_names =['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

X_train = X_train / 255.0
X_test = X_test / 255.0

# plt.imshow(X_train[0])
# plt.show()
plt.imshow(X_test[10])
# plt.show()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[32, 32, 3]))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
print(model.summary())
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer='Adam',
              metrics=['sparse_categorical_accuracy'])
model.fit(X_train, y_train, epochs=5)
test_loss, test_accuracy = model. evaluate(X_test, y_test)
print('테스트 정확도: {}'.format(test_accuracy))




