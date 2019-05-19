import os
from modules.KerasProcessor import KerasProcessor
from helpers.helpers import *

model_path = os.getcwd() + "/models/lstm_model.hdf5"

train_data = read_file('data/lstm_data.train.txt')
valid_data = read_file('data/lstm_data.valid.txt')
test_data = read_file('data/lstm_data.test.txt')

trainer = KerasProcessor(train_data, valid_data, test_data, model_path)
trainer.build_model()
trainer.train_model()