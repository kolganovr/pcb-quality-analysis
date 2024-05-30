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
df = pd.DataFrame(columns=['Код', 'Диагональное_Мат.Ожид', 'Диагональное_СКО', 'Диагональное_Точность', 'Диагональное_Крит.Ошибка',
                           'Горизонтальное_Мат.Ожид', 'Горизонтальное_СКО', 'Горизонтальное_Точность', 'Горизонтальное_Крит.Ошибка',
                           'Вертикальное_Мат.Ожид', 'Вертикальное_СКО', 'Вертикальное_Точность', 'Вертикальное_Крит.Ошибка'])

# Обрабатываем каждое изображение
for i in tqdm(range(0, len(IMAGES))):
    # Открываем изображение и переводим его в массив как чб
    image_name = IMAGES[i]
    img = cv2.imread(PATH + image_name, cv2.IMREAD_GRAYSCALE)

    # TODO: Динамический BLACKTRSHOLD в зависимости от ширины дорожек
    # FIXME: 045_ang и ang1 плохо считаются

    treshold = 120

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
    mean = np.mean(roads)
    std = np.std(roads)

    # remove all valuees that greater then mean + 3 std or less then mean - 3 std
    roads = roads[roads < mean + 3 * std]
    roads = roads[roads > mean - 3 * std]

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
    dpi = 1200
    roads = roads / (dpi * 25.4) * 1000
    
    mean = np.mean(roads)
    std = np.std(roads)
    accuracy = np.count_nonzero(roads == roads[0]) / len(roads)
    error = np.count_nonzero(roads != roads[0]) / len(roads)

    data = [mean, std, accuracy, error]

    # Округляем все данные до 3 знака после запятой
    data = [np.round(x, 3) for x in data]

    # Записываем данные в датафрейм
    # Если этот код уже есть в датафрейме, а угол другой, то добавляем в соотвествующие столбцы

    if code in df['Код'].values:
        df.loc[df['Код'] == code, angleTable + '_Мат.Ожид'] = data[0]
        df.loc[df['Код'] == code, angleTable + '_СКО'] = data[1]
        df.loc[df['Код'] == code, angleTable + '_Точность'] = data[2]
        df.loc[df['Код'] == code, angleTable + '_Крит.Ошибка'] = data[3]
    else:
        df = df._append({'Код': code,
                        angleTable + '_Мат.Ожид': data[0],
                        angleTable + '_СКО': data[1],
                        angleTable + '_Точность': data[2],
                        angleTable + '_Крит.Ошибка': data[3]}, ignore_index=True)
        
    

df.to_csv('data.csv', index=False)

    

        

