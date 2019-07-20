import tensorflow as tf


class Add:
    def __init__(self):
        a = tf.constant(1)
        b = tf.constant(2)
        c = a + b
        print('a + b = %d ' % (c))