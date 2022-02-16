from detecto import core, utils
import numpy as np
from torchvision import transforms as tf
import os
import PIL

# Wczytanie danych
cus_signs = tf.Compose([tf.ToPILImage(), tf.Resize(400), tf.ToTensor(), utils.normalize_transform()])
train_sg = core.Dataset('Train/annotations', 'Train/images', transform=cus_signs)

# Uczenie, zakomentować w przypadku wczytania modelu
model = core.Model(['crosswalk', 'trafficlight', 'stop', 'speedlimit'])
model.fit(train_sg, lr_step_size=10, learning_rate=0.0001, verbose=True)
model.save('model_sg.pth')

# Model do wczytania
# model = core.Model.load('model_sg.pth', ['crosswalk', 'trafficlight', 'stop', 'speedlimit'])

dir = 'Test/images'

for filename in os.listdir(dir):
    if filename.endswith('.png'):
        #sprawdzenie obrazka
        image = utils.read_image('Test/images/' + filename)
        labels, boxes, scores = model.predict(image)

        # filtrowanie wyników
        filtr = np.where(scores > 0.75)
        filtr_scr = scores[filtr]
        filtr_boxes = boxes[filtr]

        num_list = list(filtr[0])
        filtr_labels = []
        for i in num_list:
            filtr_labels.append(labels[i])

        #zapisywanie liczby i koordynatów znaków przejść dla pieszych
        crosswalk_count = 0
        boxes_print = np.empty((0, 4), int)

        for i in range(filtr_labels.__len__()):
            lab = filtr_labels[i]

            if lab == 'crosswalk':
                crosswalk_count += 1
                boxes_tmp = []
                boxes_tmp += filtr_boxes[i]

                xmin = boxes_tmp[0].numpy().round().astype(int)
                xmax = boxes_tmp[2].numpy().round().astype(int)
                ymin = boxes_tmp[1].numpy().round().astype(int)
                ymax = boxes_tmp[3].numpy().round().astype(int)

                # sprawdzanie czy znak zajmuje 1/10 szerokości i wysokości zdjęcia
                image_hs = PIL.Image.open('Test/images/' + filename)
                width, height = image_hs.size

                if width * 0.1 <= xmax - xmin and height * 0.1 <= ymax - ymin:
                    boxes_print = np.vstack([boxes_print, [xmin, xmax, ymin, ymax]])
                else:
                    crosswalk_count -= 1

        #wypisywanie rezultatów

        if crosswalk_count > 0:
            print(filename)
            print(crosswalk_count)

        for i in range(len(boxes_print[:, 1])):
            p = ''
            for j in range(4):
                p += str(boxes_print[i][j]) + ' '
            print(p)