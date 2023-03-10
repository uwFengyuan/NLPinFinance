import nltk
import pickle
import argparse
from collections import Counter
from coco import COCO
import pandas as pd

content = 'caption'
description = 'description'

# One of the changes.                                                                             -----wjy
# nltk.download('punkt')

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

# Tokenize the captions                                                                                 -----wjy
def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption_tit = str(coco.anns[id][content])
        tokens_tit = nltk.tokenize.word_tokenize(caption_tit.lower())
        counter.update(tokens_tit)

        caption_des = str(coco.anns[id][description])
        tokens_des = nltk.tokenize.word_tokenize(caption_des.lower())
        counter.update(tokens_des)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(json=args.caption_path,
                        threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
    data_source = 'Five_Categories_Data/Sports_and_Outdoors/tokenized'
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='/data/liufengyuan/NLPinFinance/' + data_source + '/annotations/karpathy_split_train.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='/data/liufengyuan/NLPinFinance/' + data_source + '/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=5, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
