# -*- coding: utf-8 -*-

import numpy as np
from keras.datasets import mnist
from scipy import misc
import json, os

# Загрузка тестового dataset из keras
def data_load():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return(x_train, y_train), (x_test,y_test)

# Запись результата в json
def results_to_json(predict, result):
    data = []
    for i in range(len(predict)):
        res = {
            'Real:': str(np.argmax(result[i])),
            'Await': str(predict[i]),
        }
        data.append(res)

    with open('result.json', 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Загрузка dataset по собственным изображениям
def load_own_images():
    test_x = []
    test_y = []
    count = 0
    for file in os.listdir('images'):
        img = misc.imread('images/' + file)
        imgarr = np.array(img)

        resultimgarr = np.zeros((28,28))

        for i in range(len(imgarr)):
            for j in range(len(imgarr[i])):
                resultimgarr[i][j] = imgarr[i][j][3]
        test_x.append(resultimgarr)
        test_y.append(int(file.split('.')[0]))
        count = count + 1


    test_x = np.array(test_x)
    test_x = test_x.astype('float32')
    test_x /= 255
    test_y = np.array(test_y)
    test_x = test_x.reshape(count, 784)

    return test_x, test_y