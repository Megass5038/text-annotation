import os
import pandas as pd
from modules.LDAExtractor import LDAExtractor

df = pd.read_csv('data/lda_train.csv', sep=';')

model_path = os.getcwd() + "/models/lda_model"
extractor = LDAExtractor(model_path)
print(df.cleaned[0])
print(extractor.extract_keywords(df.cleaned[0]))