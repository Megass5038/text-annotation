import pandas as pd
import os
from modules.LDATrainer import LDATrainer

data = pd.read_csv('data/articles1.csv')

trainer = LDATrainer(data.content.values.tolist())

trainer.prepare_data()
trainer.train_model()
trainer.save_model(os.getcwd() + "/models/lda_model")
