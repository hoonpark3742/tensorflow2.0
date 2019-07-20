# 세션, 플레이스홀더, saver를 사용하지 않는 tensorflow2

import tensorflow as tf

class CalculatorModel:
    def __init__(self):
        self.a = 0
        self.b = 0

    def input_number(self):
        self.a = int(input('1st number\n'))
        self.b = int(input('2nd number\n'))

    def hook(self, flag):
        self.input_number()
        if flag == 1: result = self.plus()
        elif flag ==2: result = self.subtract()
        elif flag == 3: result = self.multiply()
        elif flag == 4: result = self.divide()
        return tf.keras.backend.eval(result)    # 계속 연산하지 않고 단편적인 결과값을 출력하고 싶을 때 사용

    @tf.function    # 계속 연산(다층 레이어)하기 위해 사용 (= 딥러닝)
    def plus(self):
        result = tf.math.add(self.a, self.b)
        return result

    @tf.function
    def subtract(self):
        result = tf.math.subtract(self.a, self.b)
        return result

    @tf.function
    def multiply(self):
        result = tf.math.multiply(self.a, self.b)
        return result

    @tf.function
    def divide(self):
        result = tf.math.divide(self.a, self.b)
        return result