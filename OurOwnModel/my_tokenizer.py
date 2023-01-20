import os
import re
import pickle
import pandas as pd
from nltk.corpus import words

# nltk.download('words')

def coarse_sizing_words(keyList, wordlist, freq):
    for key in keyList:
        if wordlist[key] <= freq:
            del wordlist[key]
    return list(wordlist.keys()), wordlist

def fine_sizing_words(keyList, wordlist):
    count = 0
    for key in keyList:
        count += 1
        if count % 500 == 0:
            print(count)
        if not key in words.words():
            del wordlist[key]
    return list(wordlist.keys()), wordlist

def save_file(wordlist):
    wordListFile = pd.DataFrame.from_dict(wordlist, orient='index')
    wordListFile = wordListFile.reset_index().rename(columns={'index': 'word', 0: 'count'})
    wordListFile = wordListFile.sort_values(by=['word'], ignore_index=True)
    #wordListFile = wordListFile[wordListFile['count'] > 30]
    wordListFile.to_csv('WordList.csv', index=False)


filenames = os.listdir(r'/data/liufengyuan/NLPinFinance/Filtered Data')
wordlist = {}
pat = '[a-zA-Z]+'
total = 0
for name in filenames:
    f_save = open('/data/liufengyuan/NLPinFinance/Filtered Data/' + name, 'rb')
    file_read = pickle.load(f_save)
    f_save.close()
    for item in file_read:
        total += 1
        sentence = item['title']
        output = re.findall(pat, sentence.lower())

        features = item['feature']
        for fea in features:
            output.extend(re.findall(pat, fea.lower()))

        for word in output:
            if word in wordlist.keys():
                wordlist[word] += 1
            else:
                wordlist[word] = 1
    print(f'Tokenize {total} sentences from {name}.')
    total = 0

keyList = list(wordlist.keys())
print(len(keyList))

coarse_keyList, coarse_word_list = coarse_sizing_words(keyList, wordlist, 30)
print(len(coarse_keyList))

fine_keyList, fine_word_list = fine_sizing_words(coarse_keyList, coarse_word_list)
print(len(fine_keyList))
save_file(fine_word_list)

