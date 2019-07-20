import tensorflow as tf
import numpy as np

class Mean:
    def __init__(self):
        x_array = np.arange(18).reshape(3, 2, 3)
        x2 = tf.reshape(x_array, shape=(-1, 6))
        ## 각 열의 합을 계산
        xsum = tf.reduce_sum(x2, axis=0)
        ## 각 열의 평균을 계산
        xmean = tf.reduce_mean(x2, axis=0)

        print('입력 크기: ', x_array.shape)
        print('크기가 변경된 입력: \n', x2.numpy())
        print('열의 합: \n', xsum.numpy())
        print('열의 평균: \n', xmean.numpy())