import tensorflow as tf
import tensorflow_datasets as tfds
from imdb_test.model import Imdb

"""
케라스와 텐서플로 허브를 사용한 영화 리뷰 텍스트 분류하기
"""

if __name__ == '__main__':
    imdb = Imdb()
    imdb.create_model()