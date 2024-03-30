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
    # Открываем изображение и переводим его в массив
    image_name = IMAGES[i]
    img = cv2.imread(PATH + image_name)

    def getVals(treshhold = None):
        vals = []

        for line in img:
            row = []
            for pix in line:
                if treshhold is not None:
                    row.append(1 if sum(pix) > treshhold else 0)
                else:
                    row.append(sum(pix))
            vals.append(row)

        return np.array(vals)

    after = getVals(treshhold=160)

    roads = np.array([])
    for row in after:
        r = np.array([], dtype=int)
        couterWhite = 0
        couterBlack = 0
        for pix in row:
            if pix == 1:
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
        

