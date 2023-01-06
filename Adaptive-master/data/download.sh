#!/bin/sh
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P ./data
wget -c http://images.cocodataset.org/zips/train2014.zip -P ./data
wget -c http://images.cocodataset.org/zips/val2014.zip -P ./data

unzip ./data/captions_train-val2014.zip -d ./data
rm ./data/captions_train-val2014.zip
unzip ./data/train2014.zip -d ./data
rm ./data/train2014.zip 
unzip ./data/val2014.zip -d ./data
rm ./data/val2014.zip 
