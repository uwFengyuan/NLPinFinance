# loadRes
import json
import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from skimage.draw import polygon
import copy

class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset = []
        self.imgToAnns = {}
        self.imgs = []
        if not annotation_file == None:
            print ('loading annotations into memory...')
            time_t = datetime.datetime.utcnow()
            dataset = json.load(open(annotation_file, 'r'))
            print (datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print ('creating index...')
        imgToAnns = {ann['asin']: [] for ann in self.dataset}
        for ann in self.dataset:
            imgToAnns[ann['asin']] = ann['title']

        print ('index created!')

        # create class members
        self.imgToAnns = imgToAnns

    def getImgIds(self):

        ids = []
        for i in range(len(self.dataset)):
            ids.append(self.dataset[i]['asin'])
        return ids


    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        print ('Loading and preparing results...     ')
        res.dataset = json.load(open(resFile, 'r'))
        print('Done')
        
        # time_t = datetime.datetime.utcnow()
        # anns    = json.load(open(resFile))
        """
        assert type(anns) == list, 'results in not an array of objects'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                ann['area']=sum(ann['segmentation']['counts'][2:-1:2])
                ann['bbox'] = []
                ann['id'] = id
                ann['iscrowd'] = 0
        """
        # print ('DONE (t=%0.2fs)'%((datetime.datetime.utcnow() - time_t).total_seconds()))

        ################################################################### 这边有问题要改 #########################################################
        # res.dataset['title'] = anns
        res.createIndex()
        return res


    @staticmethod
    def decodeMask(R):
        """
        Decode binary mask M encoded via run-length encoding.
        :param   R (object RLE)    : run-length encoding of binary mask
        :return: M (bool 2D array) : decoded binary mask
        """
        N = len(R['counts'])
        M = np.zeros( (R['size'][0]*R['size'][1], ))
        n = 0
        val = 1
        for pos in range(N):
            val = not val
            for c in range(R['counts'][pos]):
                R['counts'][pos]
                M[n] = val
                n += 1
        return M.reshape((R['size']), order='F')

    @staticmethod
    def encodeMask(M):
        """
        Encode binary mask M using run-length encoding.
        :param   M (bool 2D array)  : binary mask to encode
        :return: R (object RLE)     : run-length encoding of binary mask
        """
        [h, w] = M.shape
        M = M.flatten(order='F')
        N = len(M)
        counts_list = []
        pos = 0
        # counts
        counts_list.append(1)
        diffs = np.logical_xor(M[0:N-1], M[1:N])
        for diff in diffs:
            if diff:
                pos +=1
                counts_list.append(1)
            else:
                counts_list[pos] += 1
        # if array starts from 1. start with 0 counts for 0
        if M[0] == 1:
            counts_list = [0] + counts_list
        return {'size':      [h, w],
               'counts':    counts_list ,
               }

    @staticmethod
    def segToMask( S, h, w ):
         """
         Convert polygon segmentation to binary mask.
         :param   S (float array)   : polygon segmentation mask
         :param   h (int)           : target mask height
         :param   w (int)           : target mask width
         :return: M (bool 2D array) : binary mask
         """
         M = np.zeros((h,w), dtype=np.bool)
         for s in S:
             N = len(s)
             rr, cc = polygon(np.array(s[1:N:2]), np.array(s[0:N:2])) # (y, x)
             M[rr, cc] = 1
         return M