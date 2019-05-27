import pandas as pd
import random
from nltk import sent_tokenize

df = pd.read_csv('data/articles1.csv')

sentences = []

test_data = []
valid_data = []
train_data = []

for text in df.content[0:500]:
	sentences.extend(sent_tokenize(text))

for sent in sentences:
	rand_val = random.randint(1, 10)
	if rand_val == 9:
		valid_data.append(sent)
	elif rand_val == 10:
		test_data.append(sent)
	train_data.append(sent)

with open('data/lstm_data.test.txt', encoding='utf-8', mode='w') as f:
	f.write('\n'.join(test_data))

with open('data/lstm_data.valid.txt', encoding='utf-8', mode='w') as f:
	f.write('\n'.join(valid_data))

with open('data/lstm_data.train.txt', encoding='utf-8', mode='w') as f:
	f.write('\n'.join(train_data))

with open('data/sentences.txt', encoding='utf-8', mode='w') as f:
	f.write('\n'.join(sentences))