import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

PATH = 'ph 1200\\Ph_for_exp_big\\'
IMAGES = os.listdir(PATH)
BLACKTRSHOLD = 10

if not os.path.exists('data'):
    os.mkdir('data')

# Создаем датафрейм с коллонками: Код файла (число в начале файла), Диагональное, Горизонтальное, Вертикальное
# НА каждой строчке в диагональном итд должен быть словарь: Мат.ожидание, СКО, Точность, Крит.Ошибка

# Создаем датафрейм для сохранения данных
df = pd.DataFrame(columns=['Код файла', 'Диагональное', 'Горизонтальное', 'Вертикальное'])

# Обрабатываем каждое изображение
for i in tqdm(range(0, len(IMAGES))):
    # Открываем изображение и переводим его в массив как чб
    image_name = IMAGES[i]
    img = cv2.imread(PATH + image_name, cv2.IMREAD_GRAYSCALE)

    # TODO: Динамический BLACKTRSHOLD в зависимости от ширины дорожек
    # FIXME: 045_ang и ang1 плохо считаются

    treshold = 160

    roads = np.array([])
    for row in img:
        r = np.array([], dtype=int)
        couterWhite = 0
        couterBlack = 0
        for pix in row:
            if pix > treshold:
                couterWhite += 1
            else:
                couterBlack += 1
                if couterBlack > BLACKTRSHOLD and couterWhite != 0:
                    r = np.append(r, couterWhite)
                    couterWhite = 0
                    couterBlack = 0
        roads = np.append(roads, r)

    roads = roads[roads > 5]

    # Очищаем Plot
    plt.clf()

    sns.histplot(roads, stat="probability", discrete=True)
    plt.title(image_name)

    # Сохраняем график
    plt.savefig('data/' + image_name + '.png')

    # Сохраняем данные в датафрейм
    code = image_name.split('_')[0]
    angle = image_name.split('_')[1].split('.')[0]

    if angle == 'ang':
        angleTable = 'Диагональное'
    elif angle == 'gor':
        angleTable = 'Горизонтальное'
    elif angle == 'vert':
        angleTable = 'Вертикальное'
    else:
        raise Exception('Неизвестный угол')
    
    # Переводим ширины дорожек из пикселей (кол-во dpi - код файла) в мм
    dpi = int(code)
    roads = roads / (dpi * 25.4) * 1000
    
    mean = np.mean(roads)
    std = np.std(roads)
    accuracy = np.count_nonzero(roads == roads[0]) / len(roads)
    error = np.count_nonzero(roads != roads[0]) / len(roads)
    data = {'Мат.ожидание': mean, 'СКО': std, 'Точность': accuracy, 'Крит. ошибка': error}

    # Округляем все данные до 3 знака после запятой
    data = {k: round(v, 3) for k, v in data.items()}

    if code in df['Код файла'].values:
        # Convert existing data to dictionary
        df.loc[df['Код файла'] == code, angleTable] = df.loc[df['Код файла'] == code, angleTable].apply(lambda x: {**x, **data} if isinstance(x, dict) else data)
    else:
        df = df._append({'Код файла': code, angleTable: data}, ignore_index=True)
    

df.to_csv('data.csv', index=False)

    

        

