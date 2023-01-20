import os
from urllib.request import urlopen
import re
import pickle
import requests
from NLP_For_E_commerce_main.build_vocab import Vocabulary

# 下载图片
with open( '/data/liufengyuan/NLPinFinance/COCOdata/vocab.pkl', 'rb') as f: # vocab.pkl
    vocab = pickle.load( f )
print(vocab)

"""
# 创建图片文件夹路径
if not os.path.exists(r'Image'):
    os.mkdir(r'Image')

filenames = os.listdir(r'Data')
for iter in range(20):

    # 拿到文件名和种类
    filename = re.findall('meta_(.*?).json.gz',filenames[iter])[0] + '.pkl'
    filepath = os.path.join(r'Filtered Data',filename)

    # 读取相应保存字典
    f_read = open(filepath, 'rb')
    All_data = pickle.load(f_read)
    f_read.close()

    
    # 拿到网址列表
    if len(All_data) > 0:
        for j in range(len(All_data)):
            url_list = All_data[j]['imageURLHighRes']

            # 爬取图片
            headers={
                'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36 Edg/106.0.1370.42'
            }
            for idx, url in enumerate(url_list):
                
                img_name = All_data[j]['asin'] + '_' + str(idx) + '.jpg'
                img_data = requests.get(url=url,headers=headers).content
                img_path = os.path.join(r'Image',img_name)
                
                with open (img_path,'wb') as fp:
                    fp.write(img_data)
"""