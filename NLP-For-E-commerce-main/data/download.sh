#!/bin/sh
curl http://images.cocodataset.org/annotations/annotations_trainval2014.zip -o captions_train-val2014.zip
curl http://images.cocodataset.org/zips/train2014.zip -o train2014.zip 
curl http://images.cocodataset.org/zips/val2014.zip -o val2014.zip

unzip ./captions_train-val2014.zip
rm ./captions_train-val2014.zip
unzip ./train2014.zip 
rm ./train2014.zip 
unzip ./val2014.zip
rm ./val2014.zip 
