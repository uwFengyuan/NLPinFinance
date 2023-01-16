#!/bin/sh
#curl http://images.cocodataset.org/annotations/annotations_trainval2014.zip -o captions_train-val2014.zip
#curl http://images.cocodataset.org/zips/train2014.zip -o train2014.zip 
#curl http://images.cocodataset.org/zips/val2014.zip -o val2014.zip

#unzip ./annotations_trainval2014.zip
#rm ./annotations_trainval2014.zip
#unzip ./train2014.zip 
#rm ./train2014.zip 
#unzip ./val2014.zip
#rm ./val2014.zip 
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P /data/liufengyuan/NLPinFinance/COCOdata
wget -c http://images.cocodataset.org/zips/train2014.zip -P /data/liufengyuan/NLPinFinance/COCOdata
wget -c http://images.cocodataset.org/zips/val2014.zip -P /data/liufengyuan/NLPinFinance/COCOdata

unzip /data/liufengyuan/NLPinFinance/COCOdata/annotations_trainval2014.zip -d /data/liufengyuan/NLPinFinance/COCOdata
#rm /data/liufengyuan/NLPinFinance/COCOdata/annotations_trainval2014.zip

unzip /data/liufengyuan/NLPinFinance/COCOdata/train2014.zip -d /data/liufengyuan/NLPinFinance/COCOdata
#rm /data/liufengyuan/NLPinFinance/COCOdata/train2014.zip 

unzip /data/liufengyuan/NLPinFinance/COCOdata/val2014.zip -d /data/liufengyuan/NLPinFinance/COCOdata
#rm /data/liufengyuan/NLPinFinance/COCOdata/val2014.zip 

python KarpathySplit.py
python build_vocab.py
mkdir /data/liufengyuan/NLPinFinance/COCOdata/resized
python resize.py
python train.py