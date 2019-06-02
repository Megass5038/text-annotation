from modules.KerasProcessor import KerasProcessor
from modules.LDAExtractor import LDAExtractor
from helpers.helpers import *


class TextAnnotator:
    def __init__(self, lda_model, lstm_model):
        self.lda = LDAExtractor(lda_model)
        self.lstm = KerasProcessor(model_path=lstm_model)
        self.lstm.build_model()

    def generate_annotation(self, text):
        sentences = self.lda.extract_key_sentences(text)
        text_word_ids = self.lstm.word2id(chunk_text(text), self.lstm.word_ids)
        results = []
        for sent in sentences:
            word_ids = self.lstm.word2id(chunk_text(sent), self.lstm.word_ids)
            first_part, second_part = split_list(word_ids)
            second_part = self.lstm.predict_sent(sent, data_ids=text_word_ids, num_predict=len(second_part))
            results.append((self.convert_word_ids_to_text(first_part+second_part)).capitalize())
        return results

    def convert_word_ids_to_text(self, word_ids):
        return (" ".join([self.lstm.reversed_dictionary[word_id] for word_id in word_ids])).replace("EOS", "\n")
