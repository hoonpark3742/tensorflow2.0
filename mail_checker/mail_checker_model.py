from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
import glob
import os
import numpy as np
import nltk
from collections import defaultdict


class MailChecker:
    def __init__(self):
        ham_path = './data/ham.txt'
        spam_path = './data/spam.txt'
        with open(ham_path, 'r') as infile:
            ham_sample = infile.read()
        self.cv = CountVectorizer(stop_words='english', max_features=500)
        """
        stop_words='english' 내장된 삭제할 영단어
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
        return astr.isalpha() #alpabeth 만 남기고 숫자나 기호 제거

    @staticmethod
    def down_eng_dictionary():
        nltk.download()

    def clean_text(self,docs):
        all_names = set(names.words())
        lemmatizer = WordNetLemmatizer()
        cleaned_docs = []
        for doc in docs:
            cleaned_docs.append(' '.join([lemmatizer.lemmatize(word.lower())
                                          for word in doc.split()
                                          if self.letters_only(word)
                                          and word not in all_names]))
        return cleaned_docs

    def execute(self):
        cleaned_emails = self.clean_text(self.emails)
        term_docs = self.cv.fit_transform(cleaned_emails)
        feature_mapping = self.cv.vocalbulary
        feature_names = self.cv.get_feature_names()

        return [term_docs,feature_mapping,feature_names ]

    @staticmethod
    def get_label_index (labels):
        label_index = defaultdict(list)
        for index, label in enumerate(labels):
            label_index[label].append(index)
        return label_index

    @staticmethod
    def get_prior(label_index):
        prior = {label: len(index) for label, index in label_index.items()}
        total_count = sum(prior.values())
        for label in prior:
            prior[label] /= float(total_count)  # 결과를 누적해서 연속 나눗셈을 수행
        return prior

    @staticmethod
    def get_likelihood(term_document_matrix, label_index, smoothing=0):
        """
        아규먼트의 의미
        term_document_matrix: 행렬구조로 된 문자
        label_index: 그룹핑된 레이블의 인덱스
        smoothing:
        likelihood(가능성): 주어진 가설이 참인 결과가 나올 확률
        """
        likelihood = {}
        for label, index in label_index.items():
            likelihood[label] = term_document_matrix[index, :].sum(axis=0) + smoothing
            # axis = 0 이면 열, axis = 1 이면 행 연산
            likelihood[label] = np.asanyarray(likelihood[label])[0]
            total_count = likelihood[label].sum()
            likelihood[label] = likelihood[label] / float(total_count)
        return likelihood

    @staticmethod
    def get_posterior(term_document_matrix, prior, likelihood):
        num_docs = term_document_matrix.shape[0]
        posteriors = []
        for i in range(num_docs):
            posterior = {key: np.log(prior_label)
                          for key, prior_label in prior.items}
            for label, likelihood_label in likelihood.items():
                term_document_vector = term_document_matrix.getrow(i)
                counts = term_document_vector.data
                indices = term_document_vector.indices
                for count, index in zip(counts, indices):
                    posterior[label] += np.log(likelihood_label[index]) * count
            min_log_posterior = min(posterior.values())
            for label in posterior:
                try:
                    posterior[label] = np.exp(posterior[label] - min_log_posterior)
                except:
                    posterior[label] = float('inf')
            sum_posterior = sum(posterior.values())
            for label in posterior:
                if posterior[label] == float('inf'):
                    posterior[label] = 1.0
                else:
                    posterior[label] /= sum_posterior
            posteriors.append(posterior.copy())
        return posteriors

    def email_test(self, target):
        cleaned_test = self.clean_text(target)
        term_docs_test = self.cv.transform(cleaned_test)
        prior = self.get_prior()
        likelihood = self.get_likelihood(self.term_docs, self.get_label_index(),smoothing=1 )
        posterior = self.get_posterior(term_docs_test, prior, likelihood)
        return posterior