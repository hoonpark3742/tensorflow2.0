"""
원-핫 인코딩
컴퓨터는 근본적으로 숫자만을 인식하므로,
자연어 처리에서 문자를 숫자로 바꾸는 형식
단어 집합의 크기를 벡터의 크기로 하고,
표현하고 싶은 단어의 인덱스값에 1을 부여하고
다른 인덱스에는 0을 부여하는 단어의 벡터 방식
이렇게 표현된 벡터를 원-핫 벡터라고 합니다.
"""

from konlpy.tag import Okt
okt = Okt()
token = okt.morphs("나는 자연어 처리를 배운다")
print(token)
# []나는, 자연어, 처리를, 배운다]
word2index = {}
for voca in token:
    if voca not in word2index.keys():
        word2index[voca] = len(word2index)

print('각 단어에 고유한 인덱스 부여')
print(word2index)
# {나:0, 는:1, 자연어:2, 처리:3, 를:4, 배운다:5}

def one_hot_encoding(word, word2index):
    one_hot_vector = [0]*(len(word2index))
    index = word2index[word]
    one_hot_vector[index] = 1
    return  one_hot_vector

# 단어를 입력하면 원-핫 벡터를 만들어내는 함수
print(one_hot_encoding('자연어', word2index))
