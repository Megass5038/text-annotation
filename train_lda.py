import pandas as pd
import os
from modules.LDATrainer import LDATrainer

data = pd.read_csv('data/lda_train.csv', sep=';')

trainer = LDATrainer(data.cleaned.values.tolist())

trainer.prepare_data()
trainer.train_model()
trainer.save_model(os.getcwd() + "/models/lda_model")
