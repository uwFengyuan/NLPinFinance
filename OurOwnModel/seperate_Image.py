from random import shuffle, seed
import os
import multiprocessing as mp
from PIL import Image

images = os.listdir('/data/liufengyuan/NLPinFinance/Image')
shuffle(images)
val = images[:50000]
test = images[50000:100000]
train = images[100000:]

def transfer_image(item, target):
    with open(os.path.join('/data/liufengyuan/NLPinFinance/Image/', item), 'r+b') as f:
        try:
            with Image.open(f) as img:
                img.save(os.path.join('/data/liufengyuan/NLPinFinance/trainvalImage/' + target, item), img.format)
        except:
            item

def do_parallel(data, name):
    pool = mp.Pool(48)
    for item in data:
        pool.apply_async(transfer_image, args = (item, name))
    pool.close()
    pool.join()

do_parallel(val, 'val')
do_parallel(test, 'test')
do_parallel(train, 'train')