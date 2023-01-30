import json
import pickle
import math
import multiprocessing as mp
import gzip
import os

def select_data(item, cloth_karpathy_split_train_list,cloth_ids):
    id = item['id']
    if id in cloth_ids:
        cloth_karpathy_split_train_list.append(item)

cloth_filename = open('/data/liufengyuan/NLPinFinance/Unziped_Filtered_Data/Clothing_Shoes_and_Jewelry.pkl','rb')
cloth_data = pickle.load(cloth_filename)
cloth_filename.close()
cloth_ids = []
for data in cloth_data:
    cloth_ids.append(data['asin'])
# print(len(cloth_ids))

karpathy_split_train_path = open('/data/liufengyuan/NLPinFinance/LFYdata/annotations/karpathy_split_train.json','r')
karpathy_split_train = json.load(karpathy_split_train_path)
karpathy_split_train_path.close()


manager = mp.Manager()
cloth_karpathy_split_train_list = manager.list()
# new_data_test = manager.list()
# new_data_train = manager.list()

total_data = karpathy_split_train['images']
total_data_length = len(total_data)
each_step = 1000
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
            print("[%d/%d] Filtered the cloth data." %(count, len(total_data)))
        pool.apply_async(select_data, args = (item, cloth_karpathy_split_train_list, cloth_ids))
    print('Close pool')
    pool.close()
    print('Join pool')
    pool.join()
    print(f'Round {i} : cloth_train_data is {len(cloth_karpathy_split_train_list)}')

print('Save cloth_train_data')
f_save = open('/data/liufengyuan/NLPinFinance/LFYdata/annotations/cloth_karpathy_split_train.json', 'w')
cloth_karpathy_split_train = {'type': 'caption',
                              'images':list(cloth_karpathy_split_train_list)}
json.dump(cloth_karpathy_split_train, f_save)
f_save.close()

print('Done')