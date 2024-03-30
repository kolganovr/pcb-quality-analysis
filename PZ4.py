import cv2
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

PATH = 'ph 1200\\Ph_for_exp_big\\'
IMAGES = os.listdir(PATH)
BLACKTRSHOLD = 10

if not os.path.exists('data'):
    os.mkdir('data')


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


    roads = roads.astype(int)
    roads = roads[roads > 5]

    # Очищаем Plot
    plt.clf()

    sns.histplot(roads, stat="probability", discrete=True)
    plt.title(image_name)

    # Сохраняем график
    plt.savefig('data/' + image_name + '.png')
        

