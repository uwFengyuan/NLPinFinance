import json
import os
import math
import multiprocessing as mp

def select_data(item, new_data_val, new_data_test, new_data_train):
    id = item['asin'] + '.jpg'
    if id in val:
        new_data_val.append(item)
    elif id in test:
        new_data_test.append(item)
    elif id in train:
        new_data_train.append(item)

train = os.listdir('/data/liufengyuan/NLPinFinance/trainvalImage/train')
test = os.listdir('/data/liufengyuan/NLPinFinance/trainvalImage/test')
val = os.listdir('/data/liufengyuan/NLPinFinance/trainvalImage/val')


f_save = open('/data/liufengyuan/NLPinFinance/Combined_Data.json', 'r')
total_data = json.load(f_save)
f_save.close()

manager = mp.Manager()
new_data_val = manager.list()
new_data_test = manager.list()
new_data_train = manager.list()

total_data_length = len(total_data)
each_step = 100000
print(f'Total number of round is {math.ceil(total_data_length/each_step)}')
for i in range(math.ceil(total_data_length/each_step)):
    pool = mp.Pool(48)
    count = 0
    start = each_step*i
    end = each_step*(i+1)
    if end > total_data_length:
        end = total_data_length
    for item in total_data[start:end]:
        count += 1
        if count % 2000 == 0:
            print("[%d/%d] Tokenized the captions." %(count, len(each_step)))
        pool.apply_async(select_data, args = (item, new_data_val, new_data_test, new_data_train))
    print('Close pool')
    pool.close()
    print('Join pool')
    pool.join()
    print(f'Round {i} : new_data_val is {len(new_data_val)}, new_data_test is {len(new_data_test)}, new_data_train is {len(new_data_train)}')

print('Save new_data_val')
f_save = open('/data/liufengyuan/NLPinFinance/Data_Image/new_data_val.json', 'w')
json.dump(list(new_data_val), f_save)
f_save.close()

print('Save new_data_test')
f_save = open('/data/liufengyuan/NLPinFinance/Data_Image/new_data_test.json', 'w')
json.dump(list(new_data_test), f_save)
f_save.close()

print('Save new_data_train')
f_save = open('/data/liufengyuan/NLPinFinance/Data_Image/new_data_train.json', 'w')
json.dump(list(new_data_train), f_save)
f_save.close()