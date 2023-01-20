import os
import pickle
import requests
import time
import multiprocessing as mp

def loading_image(url, number_code):

    # 爬取图片
    headers={
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36 Edg/106.0.1370.42'
    }
        
    img_name = number_code + '.jpg'
    img_data = requests.get(url=url,headers=headers).content
    img_path = os.path.join(r'/data/liufengyuan/NLPinFinance/Image',img_name)
    
    with open (img_path,'wb') as fp:
        fp.write(img_data)

# 创建图片文件夹路径
if not os.path.exists(r'/data/liufengyuan/NLPinFinance/Image'):
    os.mkdir(r'/data/liufengyuan/NLPinFinance/Image')

filenames = os.listdir(r'/data/liufengyuan/NLPinFinance/Filtered Data')

for filename in filenames[10:]:

    # 拿到文件名和种类
    #filename = re.findall('meta_(.*?).json.gz',filenames[iter])[0] + '.pkl'
    filepath = os.path.join(r'/data/liufengyuan/NLPinFinance/Filtered Data',filename)

    # 读取相应保存字典
    f_read = open(filepath, 'rb')
    All_data = pickle.load(f_read)
    f_read.close()

    print(f"Number {iter}: Loading {len(All_data)} images from {filename} dataset.")
    # 拿到网址列表
    if len(All_data) > 0:
        start = time.time()

        pool = mp.Pool(48)
        for j in range(len(All_data)):
            url = All_data[j]['imageURLHighRes'][0]
            number_code = All_data[j]['asin']
            process = pool.apply_async(loading_image, args = (url, number_code))
        pool.close()
        pool.join()

        print(f'耗时:{time.time() - start}')