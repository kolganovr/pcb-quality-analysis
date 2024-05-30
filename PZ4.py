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

# Создаем датафрейм с коллонками: Идеальная ширина (мкм) файла (число в начале файла), Диагональное, Горизонтальное, Вертикальное
# НА каждой строчке в диагональном итд должен быть словарь: Мат.ожидание, СКО, Точность, Крит.Ошибка

# Создаем датафрейм для сохранения данных
df = pd.DataFrame(columns=['Идеальная ширина (мкм)', 'Диагональное_Мат.Ожид', 'Диагональное_СКО', 'Диагональное_Точность', 'Диагональное_Крит.Ошибка',
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
    
    # Переводим ширины дорожек из пикселей (кол-во dpi - Идеальная ширина (мкм) файла) в мм
    dpi = 1200
    roads = roads / (dpi / 25.4) * 1000
    
    mean = np.mean(roads)
    std = np.std(roads)

    idealWidth = int(code) * 10
    
    goodCount = 0
    for road in roads:
        if road*0.8 < idealWidth < road*1.2:
            goodCount += 1

    accuracy = goodCount / len(roads)
    error = 1 - accuracy
    data = [mean, std, accuracy, error]

    # Округляем все данные до 3 знака после запятой
    data = [np.round(x, 3) for x in data]

    # Записываем данные в датафрейм
    # Если этот Идеальная ширина (мкм) уже есть в датафрейме, а угол другой, то добавляем в соотвествующие столбцы

    if idealWidth in df['Идеальная ширина (мкм)'].values:
        df.loc[df['Идеальная ширина (мкм)'] == idealWidth, angleTable + '_Мат.Ожид'] = data[0]
        df.loc[df['Идеальная ширина (мкм)'] == idealWidth, angleTable + '_СКО'] = data[1]
        df.loc[df['Идеальная ширина (мкм)'] == idealWidth, angleTable + '_Точность'] = data[2]
        df.loc[df['Идеальная ширина (мкм)'] == idealWidth, angleTable + '_Крит.Ошибка'] = data[3]
    else:
        df = df._append({'Идеальная ширина (мкм)': idealWidth,
                        angleTable + '_Мат.Ожид': data[0],
                        angleTable + '_СКО': data[1],
                        angleTable + '_Точность': data[2],
                        angleTable + '_Крит.Ошибка': data[3]}, ignore_index=True)
        
    
        
    
# Делаем df['Идеальная ширина (мкм)'] интом
df['Идеальная ширина (мкм)'] = df['Идеальная ширина (мкм)'].astype(int)

df.to_csv('data.csv', index=False)

# Плотим данные о мат.ожидании ско и точности для каждого угла по 3 графика на одном графике

x = df['Идеальная ширина (мкм)']
plt.plot(x, df['Диагональное_Мат.Ожид'], label='Диагональное')
plt.plot(x, df['Горизонтальное_Мат.Ожид'], label='Горизонтальное')
plt.plot(x, df['Вертикальное_Мат.Ожид'], label='Вертикальное')
plt.legend()
plt.title('Мат.ожидание')
plt.xlabel('Идеальная ширина (мкм)')
plt.ylabel('Мат.ожидание')
plt.grid()
plt.show()

plt.plot(x, df['Диагональное_СКО'], label='Диагональное')
plt.plot(x, df['Горизонтальное_СКО'], label='Горизонтальное')
plt.plot(x, df['Вертикальное_СКО'], label='Вертикальное')
plt.legend()
plt.title('СКО')
plt.xlabel('Идеальная ширина (мкм)')
plt.ylabel('СКО')
plt.grid()
plt.show()

plt.plot(x, df['Диагональное_Точность'], label='Диагональное')
plt.plot(x, df['Горизонтальное_Точность'], label='Горизонтальное')
plt.plot(x, df['Вертикальное_Точность'], label='Вертикальное')
plt.legend()
plt.title('Точность')
plt.xlabel('Идеальная ширина (мкм)')
plt.ylabel('Точность')
plt.grid()
plt.show()

    

        

