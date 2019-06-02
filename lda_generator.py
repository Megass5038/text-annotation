import os
import pandas as pd
from modules.LDAExtractor import LDAExtractor

df = pd.read_csv('data/articles1.csv')

model_path = os.getcwd() + "/models/lda_model"
extractor = LDAExtractor(model_path)
print(df.content[0])
print(extractor.extract_key_sentences((df.content[0])))
