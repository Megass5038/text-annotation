import re
import gensim.corpora as corpora
import spacy
from nltk.corpus import stopwords
from gensim.test.utils import datapath
from gensim.models.ldamodel import LdaModel


class LDATrainer:
    lda_model = None
    data_lemmatized = None
    id2word = None
    corpus = None

    def __init__(self, data_list):
        self.data = self.preprocess_data(data_list)
        self.stopwords = stopwords.words()
        self.steamer = spacy.load('en', disable=['parser', 'ner'])

    def preprocess_data(self, data):
        data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
        data = [re.sub('\s+', ' ', sent) for sent in data]
        data = [re.sub("\'", "", sent) for sent in data]
        return data

    def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = self.steamer(sent)
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def prepare_data(self):
        self.data_lemmatized = self.lemmatization(self.data)
        self.id2word = corpora.Dictionary(self.data_lemmatized)
        self.corpus = [self.id2word.doc2bow(text) for text in self.data_lemmatized]

    def train_model(self):
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.id2word,
            num_topics=20,
            random_state=100,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        return self.lda_model

    def save_model(self, path):
        file = datapath(path)
        self.lda_model.save(file)
