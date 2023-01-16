# coding: utf-8
# # Karpathy Split for MS-COCO Dataset
import json
from random import shuffle, seed
import os
import shutil

seed( 123 ) # Make it reproducible
# Change this param for quick training                                                                       -----wjy
num_val = 5000
num_test = 5000

# Add the 'data/' to keep the path right                                                                     -----wjy
val = json.load( open('/data/liufengyuan/NLPinFinance/COCOdata/annotations/captions_val2014.json', 'r') )
train = json.load( open('/data/liufengyuan/NLPinFinance/COCOdata/annotations/captions_train2014.json', 'r') )

# Merge together
imgs = val['images'] + train['images']
annots = val['annotations'] + train['annotations']

shuffle( imgs )

# Split into val, test, train
# Change the train set to 9000 for quick training, and print the length as well.                            -----wjy
dataset = {}
dataset[ 'val' ] = imgs[ :num_val ]
dataset[ 'test' ] = imgs[ num_val: num_val + num_test ]
dataset[ 'train' ] = imgs[ num_val + num_test: ]

'''# Extract the related images as dataset to parttrain and partval                                            -----wjy
filename = {}
filename['train'] = []
filename['val'] = []
for i in range(len(dataset['train'])):
    filename['train'].append(imgs[num_val + num_test+i]['file_name'])
for i in range(len(dataset['val'])):
    filename['val'].append(imgs[i]['file_name'])
#print(filename)

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
# Group by image ids
itoa = {}
for a in annots:
    imgid = a['image_id']
    if not imgid in itoa: itoa[imgid] = []
    itoa[imgid].append(a)

# write the val,test,train files                                                                             -----wjy
json_data = {}
info = train['info']
licenses = train['licenses']

split = [ 'val', 'test', 'train' ]

for subset in split:
    
    json_data[ subset ] = { 'type':'caption', 'info':info, 'licenses': licenses,
                           'images':[], 'annotations':[] }
    
    for img in dataset[ subset ]:
        
        img_id = img['id']
        anns = itoa[ img_id ]
        
        json_data[ subset ]['images'].append( img )
        json_data[ subset ]['annotations'].extend( anns )
        
    json.dump( json_data[ subset ], open( '/data/liufengyuan/NLPinFinance/COCOdata/annotations/karpathy_split_' + subset + '.json', 'w' ) )
