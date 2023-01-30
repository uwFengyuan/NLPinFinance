__author__ = 'tylin'

from tokenizer.title_tokenizer import title_tokenize
from tokenizer.my_ptbtokenizer import PTBTokenizer
#from my_build_vocab import tokenize
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider


# from .spice.spice import Spice


class COCOEvalCap:

    ################################################################ 对coco修改 #####################################################################

    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

    ################################################################ 对coco修改 #####################################################################

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        GTS = {}
        RES = {}
        for asin, title in gts.items():
            token_list = title_tokenize(title)
            final_title = ' '.join(token_list)
            GTS[asin] = final_title
        #gts_copy = {}
        #for key, value in gts.items():
        #    gts_copy[key] = tokenize(value)
        #gts = gts_copy
        #print(gts)
        #print(GTS)
        for asin, title in res.items():
            token_list = title_tokenize(title)
            final_title = ' '.join(token_list)
            RES[asin] = final_title
        # gts = tokenizer.tokenize(gts)
        #print(res)
        #print(RES)

        #res_copy = {}
        #for key, value in res.items():
        #    res_copy[key] = tokenize(value)
        #res = res_copy

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(GTS, RES)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, GTS.keys(), m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, GTS.keys(), method)
                print("%s: %0.3f" % (method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
