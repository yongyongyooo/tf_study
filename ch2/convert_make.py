from PIL import Image
import random
import sys
import os

train_path = './train/'

convert_path = './convert/'
test_path = './test/'

set_size = 3000
test_size = 1000

if len(sys.argv) == 2:
    set_size = int(sys.argv[1])

c_size = 128
origin_data = os.listdir('./train')
random.shuffle(origin_data)

print('train set make')
for idx, d in enumerate(origin_data[:set_size]):
    Image.open(train_path + d).resize((c_size, c_size)).save(convert_path + d)
    print('{} idx image finish {} remain'.format(idx, set_size - idx))

print('test set make')
for idx, d in enumerate(origin_data[-test_size:]):
    Image.open(train_path + d).resize((c_size, c_size)).save(test_path + d)
    print('{} idx image finish {} remain'.format(idx, test_size - idx))
