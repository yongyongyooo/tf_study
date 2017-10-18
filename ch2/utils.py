import random
import os
import numpy as np
from PIL import Image

train_path = './convert/'
test_path = './test/'

data_size = 3000
c_size = 256
imgs = list(filter(lambda x: x != 'Thumbs.db', os.listdir(train_path)))
random.shuffle(imgs)
imgs = imgs[:data_size]

def get_test():
    datas = []
    test_dir = list(filter(lambda x: x != 'Thumbs.db', os.listdir(test_path)))
    for d in test_dir:
        im = Image.open(test_path + d)
        datas.append(np.array(im))

    datas = np.array(datas).reshape((100, c_size * c_size))

    labels = []

    for d in test_dir:
        tmp = [0, 0]
        if d.split('.')[0] == 'cat':
            tmp[0] = 1
        else:
            tmp[1] = 1

        labels.append(tmp)

    labels = np.array(labels)
    
    return (datas, labels)


def get_img(page, size):
    t = []

    for d in imgs[page*size:(page+1)*size]:
        im = Image.open(train_path + d)
        t.append(np.array(im))

    t = np.array(t).reshape((size, c_size * c_size))

    return t

# cat : 0
# dog : 1
def get_label(page, size):
    t = []

    for d in imgs[page*size:(page+1)*size]:
        tmp = [0, 0]
        if d.split('.')[0] == 'cat':
            tmp[0] = 1
        else:
            tmp[1] = 1

        t.append(tmp)

    t = np.array(t)
    return t

def get_batch(page, size):
    return (get_img(page, size), get_label(page, size))
