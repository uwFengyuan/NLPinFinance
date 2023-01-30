import os
import re
import pickle
import json
import gzip

filenames = os.listdir(r'/data/liufengyuan/NLPinFinance/Data')
total = 0
for name in filenames:

    data = {}
    filename_path = os.path.join(r'/data/liufengyuan/NLPinFinance/Data',name)
    category = re.findall('meta_(.*?).json.gz',name)[0]
    data[category] = []

    with gzip.open(filename_path) as f:

        count = 0
        for l in f:
            count += 1
            pre_data = json.loads(l.strip())
            if 'imageURLHighRes' in pre_data.keys() and 'feature' in pre_data.keys() and 'description' in pre_data.keys() and 'title' in pre_data.keys() and 'asin' in pre_data.keys():
                if len(pre_data['imageURLHighRes']) != 0 and len(pre_data['description']) != 0 and len(pre_data['feature']) != 0 and len(pre_data['title']) != 0 and len(pre_data['asin']) != 0:
                    data[category].append({k : pre_data[k] for k in ('asin','title','feature', 'description', 'imageURLHighRes')})

    print(f'{category}中满足条件的共{len(data[category])}条,占比{round(100*len(data[category])/count,2)}%')
    total += len(data[category])
    # 字典保存
    path = os.path.join(r'/data/liufengyuan/NLPinFinance/Unziped_Filtered_Data',category)
    f_save = open(path + '.pkl', 'wb')
    pickle.dump(data[category], f_save)
    f_save.close() # 1276549
print(total)