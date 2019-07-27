import os
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

train_dir = os.path.join('./data/train','train')
validation_dir = os.path.join('./data/validation','validation')

IMG_SHAPE = (128,128,3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
prediction_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(global_average_layer)
model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
data_gen_train = ImageDataGenerator(rescale=1/255.)
data_gen_valid = ImageDataGenerator(rescale=1/255.)

train_generator = data_gen_train.flow_from_directory('./data/train',
                                                     target_size=(128,128),
                                                     batch_size=128,
                                                     class_mode='binary')
valid_generator = data_gen_valid.flow_from_directory('./data/validation',
                                                     target_size=(128,128),
                                                     batch_size=128,
                                                     class_mode='binary')
model.fit_generator(train_generator,
                    epochs=5,
                    validation_data = valid_generator)
valid_loss, valid_accuracy = model.evaluate_generator(valid_generator)
print('튜닝 전 정확도 : {}'.format(valid_accuracy))
base_model.trainable = True
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
"""
Model Tuning
"""
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])
model.fit_generator(train_generator,
                    epochs=5,
                    validation_data=valid_generator)
valid_loss, valid_accuracy = model.evaluate_generator(valid_accuracy)
print('튜닝 후 정확도 : {}'.format(valid_accuracy))