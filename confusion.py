# -*- coding: utf-8 -*-

import numpy as np
import tensorflow
import pandas
import matplotlib
import seaborn

# Построить матрицу ошибок
def matrix(predict, y, count):

    predict = list(map(np.argmax, predict) )
    y = list(map(np.argmax, y) )
    matrix = tensorflow.math.confusion_matrix(labels=y, predictions=predict).numpy()
    matrix = np.around(matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis], decimals=2)

    columns = []
    for i in range(0, count):
        columns.append(i)

    frame = pandas.DataFrame(matrix,
                         index = columns,
                         columns = columns)

    seaborn.heatmap(frame, annot=True,cmap=matplotlib.pyplot.cm.Blues)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.ylabel('True label')
    matplotlib.pyplot.xlabel('Predicted label')
    matplotlib.pyplot.show()