'''
# 从COCO数据集中抽取7000张图片作为实验对象
import json
from random import shuffle, seed
import os
import shutil

# Make it reproducible
seed( 123 )

# Change this param for quick training                                                                     
num_val = 1000
num_test = 1000
num_total = 7000

# Add the 'data/' to keep the path right                                                                     
val = json.load( open(r'data/annotations/captions_val2014.json', 'r') )
train = json.load( open(r'data/annotations/captions_train2014.json', 'r') )

# Merge together
imgs = val['images'] + train['images']
annots = val['annotations'] + train['annotations']

# Shuffle the dataset
shuffle( imgs )

# Split into val, test, train
# Change the train set to 5000 for quick training, and print the length as well.                            -----wjy
dataset = {}
dataset[ 'val' ] = imgs[ :num_val ]
dataset[ 'test' ] = imgs[ num_val: num_val + num_test ]
dataset[ 'train' ] = imgs[ num_val + num_test: num_total]
print(f'训练集的大小为{len(dataset["train"])}')
print(f'验证集的大小为{len(dataset["val"])}')
print(f'测试集的大小为{len(dataset["test"])}')

# Extract the related images as dataset to parttrain and partval                                            -----wjy
filename = {}
filename['train'] = []
filename['val'] = []
for i in range(len(dataset['train'])):
    filename['train'].append(imgs[num_val + num_test+i]['file_name'])
for i in range(len(dataset['val'])):
    filename['val'].append(imgs[i]['file_name'])
print('所抽取的图片名字列表为:\n',filename)

#The path of the part_dataset
train_path = './data/train2014'
train_part_path = './data/train_part'
val_path = './data/val2014'
val_part_path = './data/val_part'           

for file in filename['train']:
    if file.count('train') ==1:
        shutil.copy(os.path.join(train_path,file),train_part_path)
    if file.count('val') ==1:
        shutil.copy(os.path.join(val_path,file),train_part_path)    

for file in filename['val']:
    if file.count('train') ==1:
        shutil.copy(os.path.join(train_path,file),val_part_path)
    if file.count('val') ==1:
        shutil.copy(os.path.join(val_path,file),val_part_path)
'''
# 验证是否准确：
import os
path =r"D:\nlp for e-commerce\Code\Adaptive\data\resized\train2014"  #文件夹路径
count1 = 0
for file in os.listdir(path): #file 表示的是文件名
        count1 = count1+1
print(count1)

path =r"D:\nlp for e-commerce\Code\Adaptive\data\resized\val2014"  #文件夹路径
count2 = 0
for file in os.listdir(path): #file 表示的是文件名
        count2 = count2+1
print(count2)
