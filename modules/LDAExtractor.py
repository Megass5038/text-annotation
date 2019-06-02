import re
import operator
import spacy
import statistics
from statistics import StatisticsError
from gensim.test.utils import datapath
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk import sent_tokenize


class LDAExtractor:

    def __init__(self, model_path):
        file = datapath(model_path)
        self.model = LdaModel.load(file)
        self.id2word = self.model.id2word
        self.stopwords = stopwords.words()
        self.steamer = spacy.load('en', disable=['parser', 'ner'])

    def preprocess_text(self, text):
        text = re.sub('\S*@\S*\s?', '', text)
        text = re.sub('\s+', ' ', text)
        text = re.sub("\'", "", text)
        return text

    def lemmatize_text(self, text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        doc = self.steamer(text)
        return [token.lemma_ for token in doc if token.pos_ in allowed_postags]

    def extract_keywords(self, text, keywords_num=10, with_coef=False):
        text = self.preprocess_text(text)
        tokens = self.lemmatize_text(text)
        doc = self.id2word.doc2bow(tokens)
        topics = self.model.get_document_topics(doc)
        return self.extract_keywords_from_topics(topics, keywords_num, with_coef)

    def extract_keywords_from_topics(self, topics, keywords_num, with_coef=False):
        scored_topics = {}

        for id, score in topics:
            scored_topics[id] = {
                'keywords': self.model.show_topic(id),
                'score': score
            }
        scored_keywords = {}
        for topic_id, topic in scored_topics.items():
            topic_score = 1 + float(topic['score'])
            for keyword in topic['keywords']:
                keyword_score = float(keyword[1]) * topic_score
                if keyword not in scored_keywords or (scored_keywords[keyword[0]] < keyword_score):
                    scored_keywords[keyword[0]] = keyword_score
        sorted_keywords = sorted(scored_keywords.items(), key=operator.itemgetter(1), reverse=True)[0:keywords_num]
        return sorted_keywords if with_coef else [keyword for keyword, score in sorted_keywords]

    def extract_key_sentences(self, text, sent_num=5):
        keywords = self.extract_keywords(text, 999, True)
        keywords_dict = dict(keywords)
        text = self.preprocess_text(text)
        sentences = sent_tokenize(text)
        sent_scores = [(sent, self.calculate_sent_score(sent, keywords_dict)) for sent in sentences][0:sent_num]
        sorted_sents = sorted(sent_scores, key=lambda tup: tup[1], reverse=True)
        return [sent for sent, score in sorted_sents]

    def calculate_sent_score(self, text, scores):
        tokens = self.lemmatize_text(text)
        sent_scores = []
        for token in tokens:
            sent_scores.append(scores.get(token, 0))
        try:
            return statistics.mean(sent_scores)
        except StatisticsError as e:
            return 0
