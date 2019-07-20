from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
import glob
import os
import numpy as np
import nltk


class MailChecker:
    def __init__(self):
        ham_path = './data/ham.txt'
        spam_path = '.data/spam.txt'
        with open(ham_path, 'r') as infile:
            ham_sample = infile.read()
        cv = CountVectorizer(stop_words= 'english', max_features=500)
        """
        stop_words= 'english' 내장된 삭제할 영단어
        """
        emails, labels = [], []
        file_path = './data/'
        for filename in glob.glob(os.path.join(file_path, 'ham.txt')):
            with open(filename, 'r', encoding='ISO-8859-1') as infile:
                emails.append(infile.read())
                labels.append(0)

        for filename in glob.glob(os.path.join(file_path, 'spam.txt')):
            with open(filename, 'r', encoding='ISO-8859-1') as infile:
                emails.append(infile.read())
                labels.append(1)

    @staticmethod
    def letters_only(astr):
        return astr.isaalpha()  # 알파벳만 남기고 숫자나 기호 제거

    @staticmethod
    def down_eng_dictionary():
        nltk.download()

