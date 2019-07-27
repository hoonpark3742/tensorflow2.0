from konlpy.corpus import kolaw
import matplotlib.pyplot as plt
# print(kolaw.fileids())
font_path = 'C:\Windows\Fonts\malgun.ttf'
c = kolaw.open('constitution.txt').read()
print(c[:40])

from konlpy.corpus import kobill
kobill.fileids()
d = kobill.open('1809890.txt').read()
print(d[:40])

from konlpy.tag import *
hannanum = Hannanum()
kkma = Kkma()
komoran = Komoran()
# mecab = Mecab()
okt = Okt()

hannanum.nouns(c[:40])
# print(c[:40])

kkma.nouns(c[:40])
# print(c[:40])

hannanum.morphs(c[:40])
# print(c[:40])

kkma.morphs(c[:40])
# print(c[:40])

# komoran은 빈줄이 있으면 에러가 남
komoran.morphs("\n".join([s for s in c[:40].split("\n") if s]))
# print(c[:40])

okt.morphs(c[:40])

hannanum.pos(c[:40])

kkma.pos(c[:40])

# komoran은 빈줄이 있으면 에러가 남
komoran.pos("\n".join([s for s in c[:40].split("\n") if s]))

okt.pos(c[:40])

okt.tagset

from nltk import Text

kolaw = Text(okt.nouns(c), name="kolaw")
kolaw.plot(30)
plt.show()

from wordcloud import WordCloud

# 자신의 컴퓨터 환경에 맞는 한글 폰트 경로를 설정


wc = WordCloud(width = 1000, height = 600, background_color="white", font_path=font_path)
plt.imshow(wc.generate_from_frequencies(kolaw.vocab()))
plt.axis("off")
plt.show()
