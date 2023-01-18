import json
import os
import pickle

global_path = '/data/liufengyuan/NLPinFinance/Filtered Data/'
filenames = os.listdir(global_path)
total_data = []
for file in filenames:
    this_file = pickle.load( open(global_path + file, 'rb') )
    total_data.extend(this_file)
    print(f'File {file} has {len(this_file)} number of files and the total length is {len(total_data)}.')

json.dump( total_data, open( '/data/liufengyuan/NLPinFinance/Combined_Data.json', 'w' ) )