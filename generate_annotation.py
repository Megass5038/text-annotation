import os
import pandas as pd
from text_annotator import TextAnnotator


df = pd.read_csv('data/articles1.csv')
lda_model_path = os.getcwd() + "/models/lda_model"
lstm_model_path = os.getcwd() + "/models/lstm_model.hdf5"

annotator = TextAnnotator(lda_model_path, lstm_model_path)

for content in df.content[0:500]:
    print(annotator.generate_annotation(content))
