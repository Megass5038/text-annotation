import re
import operator
from pymystem3 import Mystem
from gensim.test.utils import datapath
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import pos_tag


class LDAExtractor:

    def __init__(self, model_path):
        file = datapath(model_path)
        self.model = LdaModel.load(file)
        self.id2word = self.model.id2word
        self.stopwords = stopwords.words('russian')
        self.steamer = Mystem()

    def preprocess_text(self, text):
        text = re.sub('\S*@\S*\s?', '', text)
        text = re.sub('\s+', ' ', text)
        text = re.sub("\'", "", text)
        return text

    def lemmatize_text(self, text, allowed_postags=['S', 'ADV', 'V', 'A']):
        doc = self.steamer.lemmatize(text)
        doc = "".join(doc)
        words = word_tokenize(doc, language='russian')
        tags = pos_tag(words, lang='rus')
        texts_out = [token[0] for token in tags if token[1] in allowed_postags and token not in self.stopwords]
        return texts_out

    def extract_keywords(self, text, keywords_num=10):
        text = self.preprocess_text(text)
        tokens = self.lemmatize_text(text)
        doc = self.id2word.doc2bow(tokens)
        topics = self.model.get_document_topics(doc)
        return self.extract_keywords_from_topics(topics, keywords_num)

    def extract_keywords_from_topics(self, topics, keywords_num):
        scored_topics = {}

        for id, score in topics:
            scored_topics[id] = {
                'keywords': self.model.show_topic(keywords_num),
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
        return [keyword for keyword, score in sorted_keywords]