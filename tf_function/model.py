import tensorflow as tf

class TfFunction:
    def __init__(self):
        pass

    def create_model(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # 채널 차원 추가. 차원은 컬럼(=feature, variable) 의 의미
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
        train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)
        ).shuffle(10000).batch(32)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)