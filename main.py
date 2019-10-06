# -*- coding: utf-8 -*-

from tensorflow import keras
import data_work as data
import confusion
from keras.utils import to_categorical

# Тестировать по своим изображениям (1 - да, остальное - нет).
OWN_IMAGES = 0
# Строить матрицу ошибок (1 - да, остальное - нет)
CONFUSION = 0

# Выполнение обученой моделью тестовых данных
def execute(model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test)
    print('Accuracy ', acc)

if __name__ == '__main__':
    trained_model = keras.models.load_model('trained_model.h5')
    if OWN_IMAGES == 1:
        (x_test, y_test) = data.load_own_images()
    else:
        (x_train, y_train), (x_test, y_test) = data.data_load()

    y_test = to_categorical(y_test)
    execute(trained_model,x_test, y_test)
    predict = trained_model.predict(x_test)
    data.results_to_json(predict, y_test)

    if CONFUSION == 1:
        confusion.matrix(predict, y_test, 10)