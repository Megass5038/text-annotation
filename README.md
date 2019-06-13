# text-annotation

Выполните pipenv install для установки зависимостей <br>

Данные из датасета необходимо поместить в папку data <br>

Датасет - https://www.kaggle.com/snapcrack/all-the-news

Что бы использовать библиотеку надо выполнить следующие шаги:

1. Обучить LDA модель - запустить LDATrainer.train_model
2. Обучить LSTM сеть - KerasProcessor.train_model
3. Получить результаты - TextAnnotator.generate_annotation