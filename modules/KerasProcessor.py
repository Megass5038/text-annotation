from __future__ import print_function

import collections
import os
import numpy as np
import pickle

os.environ['KERAS_BACKEND'] = 'tensorflow'

from helpers.helpers import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.models import model_from_json
from modules.KerasBatchGenerator import KerasBatchGenerator
from keras.callbacks import ModelCheckpoint


class KerasProcessor:
    checkpointer = None
    model = None

    def __init__(
            self, train_data=[], valid_data=[], test_data=[], model_path="",
            num_steps=30, batch_size=20,
            hidden_size=500, use_dropout=True
    ):
        self.train_words = chunk_text(train_data) if train_data else []
        self.word_ids = self.build_vocab(self.train_words) if self.train_words else []
        self.train_data = self.word2id(self.train_words, self.word_ids) if train_data else []
        self.valid_data = self.word2id(chunk_text(valid_data), self.word_ids) if valid_data else []
        self.test_data = self.word2id(chunk_text(test_data), self.word_ids) if test_data else []
        self.model_path = model_path
        self.model_structure_path = os.path.splitext(self.model_path)[0] + '.json'
        self.model_ids_path = os.path.splitext(self.model_path)[0] + '.pickle'
        self.vocabulary = len(self.word_ids)
        self.reversed_dictionary = dict(zip(self.word_ids.values(), self.word_ids.keys())) if self.word_ids else {}
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.use_dropout = use_dropout

    def build_vocab(self, words):
        counter = collections.Counter(words)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        words, _ = list(zip(*count_pairs))
        word2id = dict(zip(words, range(len(words))))

        return word2id

    def word2id(self, words, word2id):
        return [word2id[word] for word in words if word in word2id]

    def build_model(self):
        if not self.load_structure_if_exists():
            model = Sequential()
            model.add(Embedding(self.vocabulary, self.hidden_size, input_length=self.num_steps))
            model.add(LSTM(self.hidden_size, return_sequences=True))
            # model.add(LSTM(self.hidden_size, return_sequences=True))
            if self.use_dropout:
                model.add(Dropout(0.5))
            # model.add(TimeDistributed(Dense(self.vocabulary)))
            model.add(Dense(self.vocabulary, activation='softmax'))
            # optimizer = Adam()
            self.model = model
            self.save_model_structure()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.checkpointer = ModelCheckpoint(filepath=self.model_path, verbose=1)
        self.load_weights_if_exists()

    def load_weights_if_exists(self):
        exists = os.path.isfile(self.model_path)
        if exists:
            self.model.load_weights(self.model_path)

    def load_structure_if_exists(self):
        exists_structure = os.path.isfile(self.model_structure_path)
        exists_word_ids = os.path.isfile(self.model_ids_path)
        if exists_structure and exists_word_ids:
            with open(self.model_structure_path) as f:
                json_model = f.read()
            self.model = model_from_json(json_model)
            with open(self.model_ids_path, 'rb') as f:
                self.word_ids = pickle.load(f)
            self.vocabulary = len(self.word_ids)
            self.reversed_dictionary = dict(zip(self.word_ids.values(), self.word_ids.keys())) if self.word_ids else {}
        return exists_structure and exists_word_ids

    def save_model_structure(self):
        model_json = self.model.to_json()
        with open(self.model_structure_path, "w") as json_file:
            json_file.write(model_json)
        with open(self.model_ids_path, "wb") as word_ids_file:
            pickle.dump(self.word_ids, word_ids_file, protocol=pickle.HIGHEST_PROTOCOL)

    def train_model(self, num_epochs=40):
        train_data_generator = KerasBatchGenerator(self.train_data, self.num_steps, self.batch_size, self.vocabulary,
                                                   skip_step=self.num_steps)
        valid_data_generator = KerasBatchGenerator(self.valid_data, self.num_steps, self.batch_size, self.vocabulary,
                                                   skip_step=self.num_steps)
        validation_steps = len(self.valid_data) // (self.batch_size * self.num_steps)
        validation_steps = validation_steps if validation_steps else 1
        self.model.fit_generator(
            train_data_generator.generate(), len(self.train_data) // (self.batch_size * self.num_steps), num_epochs,
            validation_data=valid_data_generator.generate(),
            validation_steps=validation_steps, callbacks=[self.checkpointer]
        )
        self.model.save(self.model_path)

    def load_model(self):
        self.model = load_model(self.model_path)

    def predict_train_data(self):
        dummy_iters = 40
        example_training_generator = KerasBatchGenerator(self.train_data, self.num_steps, 1, self.vocabulary,
                                                         skip_step=1)
        print("Training data:")
        for i in range(dummy_iters):
            next(example_training_generator.generate())

        num_predict = 10
        true_print_out = "Actual words: "
        pred_print_out = "Predicted words: "

        for i in range(num_predict):
            data = next(example_training_generator.generate())
            prediction = self.model.predict(data[0])
            predict_word = np.argmax(prediction[:, self.num_steps - 1, :])
            true_print_out += self.reversed_dictionary[self.train_data[self.num_steps + dummy_iters + i]] + " "
            pred_print_out += self.reversed_dictionary[predict_word] + " "

        print(true_print_out)
        print(pred_print_out)

    def predict_test_data(self):
        dummy_iters = 75
        example_test_generator = KerasBatchGenerator(self.test_data, self.num_steps, 1, self.vocabulary,
                                                     skip_step=1)
        print("Test data:")
        for i in range(dummy_iters):
            next(example_test_generator.generate())

        num_predict = 10
        true_print_out = "Actual words: "
        pred_print_out = "Predicted words: "

        for i in range(num_predict):
            data = next(example_test_generator.generate())
            prediction = self.model.predict(data[0])
            predict_word = np.argmax(prediction[:, self.num_steps - 1, :])
            true_print_out += self.reversed_dictionary[self.test_data[self.num_steps + dummy_iters + i]] + " "
            pred_print_out += self.reversed_dictionary[predict_word] + " "
        print(true_print_out)
        print(pred_print_out)

    def predict_sent(self, sent, text="", data_ids={}, num_predict=10):
        data_ids = self.word2id(chunk_text(text), self.word_ids) if not data_ids else data_ids
        sent_ids = self.word2id(chunk_text(sent), self.word_ids)
        try:
            sent_ind = self.get_first_index_of_sequence(sent_ids, data_ids)
        except IndexError as e:
            return []
        keras_gen = KerasBatchGenerator(data_ids, self.num_steps, 1, self.vocabulary, skip_step=1, idx=sent_ind)

        predicted_words = []
        for i in range(num_predict):
            data = next(keras_gen.generate())
            prediction = self.model.predict(data[0])
            predict_word = np.argmax(prediction[:, self.num_steps - 1, :])
            predicted_words.append(predict_word)
        return predicted_words

    def get_first_index_of_sequence(self, search_seq, haystack):
        len_search = len(search_seq)
        for i in range(len(haystack)):
            if haystack[i:i + len_search] == search_seq:
                return i
        raise IndexError("Sequence not in list")
