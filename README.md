# MnistML
Mnist with Keras (handwrited digits recognition)
Необходимые пакеты: tensorflow Keras scipy numpy pandas matplotlib seaborn

Начало работы: Обученная модель хранится в файле trained_model.h5. Нейросеть можно обучить заново, для этого надо выполнить файл learn.py. Чтобы работать с обученной изначально моделью, надо запустить файл main.py.

learn.py: Здесь происходит создание трех-слойной модели, а затем ее обучение на тестовом dataset из библиотеки keras. Обученная модель сохраняется в файл trained_model.h5.

main.py: Файл, в котором происходит работа с обученной моделью. В начале задаются две переменные : OWN_IMAGES и CONFUSION. OWN_IMAGES = 1 соответсвует режиму, при котором модель будет тестироваться на пользовательских изображениях, которые хранятся в директории images. Важно, чтобы эти файлы были формата png и размерностью 28x28. CONFUSION = 1 соответсвует режиму, при котором строится матрица ошибок (confusion matrix). Если запускать на режиме без собствнных изображений, то data set будет браться из keras. В конце результат загружается в файл result.json.

data_work.py: Файл, в котором определены методы работы с data set-ми. Метод data_load: загрузка данных из keras. Испоользуется для обучения или для тестирования без свои изображений. Метод load_own_images: загрузка data set из собственных изображений, которые расположены в папке images. Можно загружать любое количество изображений более нуля. Метод results_to_json: запись результатов в файл result.json.

confusion.py: построение матрицы ошибок. Используется pandas, matplotlib и seaborn.