import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
class Imdb:
    def __init__(self):
        pass

    def create_model(self):
        imdb = keras.datasets.imdb
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
        # print("훈련 샘플: {}, 레이블: {}".format(len(train_data), len(train_labels)))
        # print("검증 샘플: {}, 레이블: {}".format(len(test_data), len(test_data)))
        # 첫번째 리뷰
        # print(train_data[0])
        # print('단어의 수')
        # print(len(train_data[0]), len(train_data[1]))
        # 단어와 정수 인덱스를 매핑한 딕셔너리
        word_index = imdb.get_word_index()

        # 처음 몇 개 인덱스는 사전에 정의되어 있습니다
        word_index = {k: (v + 3) for k, v in word_index.items()}
        word_index["<PAD>"] = 0
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2  # unknown
        word_index["<UNUSED>"] = 3

        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        temp = self.decode_review(train_data[0],reverse_word_index)


        train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                                value=word_index["<PAD>"],
                                                                padding='post',
                                                                maxlen=256)

        test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                               value=word_index["<PAD>"],
                                                               padding='post',
                                                               maxlen=256)
        vocab_size = 10000

        model = keras.Sequential()
        model.add(keras.layers.Embedding(vocab_size, 16, input_shape=(None,)))
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dense(16, activation=tf.nn.relu))
        model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

        # print(model.summary())
        model.compile(optimizer=tf.optimizers.Adam(),
                      loss='binary_crossentropy',
                      metrics=['acc'])
        x_val = train_data[:10000]
        partial_x_train = train_data[10000:]

        y_val = train_labels[:10000]
        partial_y_train = train_labels[10000:]
        history = model.fit(partial_x_train,
                            partial_y_train,
                            epochs=40,
                            batch_size=512,
                            validation_data=(x_val, y_val),
                            verbose=1)
        results = model.evaluate(test_data, test_labels)

        print(results)

        history_dict = history.history
        history_dict.keys()
        acc = history_dict['acc']
        val_acc = history_dict['val_acc']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epochs = range(1, len(acc) + 1)
        print('==============')
        # "bo"는 "파란색 점"입니다
        plt.plot(epochs, loss, 'bo', label='Training loss')
        # b는 "파란 실선"입니다
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
        plt.clf()  # 그림을 초기화합니다

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

        model.save('./data/saved_model.h5')



    @staticmethod
    def decode_review(text,reverse_word_index ):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])