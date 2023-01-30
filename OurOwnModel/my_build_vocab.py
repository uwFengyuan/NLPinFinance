import pickle
import re
import os
import argparse
import json
import pandas as pd
from nltk.corpus import words

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

def coarse_sizing_words(keyList, wordlist, threshold):
    for key in keyList:
        if wordlist[key] <= threshold:
            del wordlist[key]
    return list(wordlist.keys()), wordlist

def fine_sizing_words(keyList, wordlist):
    count = 0
    for key in keyList:
        count += 1
        if count % 2000 == 0:
            print("[%d/%d] Tokenized the captions." %(count, len(wordlist)))
        if not key in words.words():
            del wordlist[key]
    return list(wordlist.keys()), wordlist

def save_file(wordlist, csv_file_path):
    wordListFile = pd.DataFrame.from_dict(wordlist, orient='index')
    wordListFile = wordListFile.reset_index().rename(columns={'index': 'word', 0: 'count'})
    wordListFile = wordListFile.sort_values(by=['word'], ignore_index=True)
    wordListFile = wordListFile[~wordListFile['word'].isnull()]
    wordListFile.to_csv(csv_file_path, index=False)
    return wordListFile

def tokenize(text):
    return re.findall('[a-zA-Z]+', text.lower())

def get_wordlist(file_path):
    wordlist = {}
    f_save = open(file_path, 'r')
    file_read = json.load(f_save)
    f_save.close()
    for item in file_read:
        sentence = item['title']
        output = tokenize(sentence)
        features = item['feature']
        output.extend(tokenize(fea) for fea in features)
        for word in output:
            if word in wordlist.keys():
                wordlist[word] += 1
            else:
                wordlist[word] = 1
    return list(wordlist.keys()), wordlist

# Tokenize the captions                            
def build_vocab(file_path, threshold):
    """Build a simple vocabulary wrapper."""
    csv_file_path = '/home/liufengyuan/NLPinFinance/WordList_Title_wjy.csv'

    if not os.path.exists(csv_file_path):
        keyList, wordlist = get_wordlist(file_path)
        print(f"Raw number of words is {len(keyList)}")
        coarse_keyList, coarse_word_list = coarse_sizing_words(keyList, wordlist, threshold)
        print(f"The number of words after coarse sizing is {len(coarse_keyList)}")
        fine_keyList, fine_word_list = fine_sizing_words(coarse_keyList, coarse_word_list)
        print(f"The number of words after fine sizing is {len(fine_keyList)}")
        wordListFile = save_file(fine_word_list, csv_file_path)
    else:
        print('Tokenized wordlistFile has existed.')
        wordListFile = pd.read_csv(csv_file_path)

    words = [row['word'] for _, row in wordListFile.iterrows()]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    vocab.add_word('<sep>')

    # Adds the words to the vocabulary.
    for _, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(file_path=args.caption_path,
                        threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: %d" %len(vocab))
    print("Saved the vocabulary wrapper to '%s'" %vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='/data/liufengyuan/NLPinFinance/Combined_Data.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='/data/liufengyuan/NLPinFinance/my_vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=30, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
