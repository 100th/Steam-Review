from __future__ import print_function
import nltk
import numpy
import openpyxl
from nltk.stem.porter import * # for Stemming
from nltk.stem import PorterStemmer # ProterStemmer
from nltk.stem import WordNetLemmatizer # for lemmatizing
from nltk.corpus import stopwords # for stopwords removal

frequency = {}

wb=openpyxl.load_workbook('newtest.xlsx', read_only=False)
ws=wb.active


for r in ws.rows:
    row_index = r[0].row # 행 인덱스?
    review = r[4].value
    print(review)

    # tokenization start
    aft_tokens = nltk.word_tokenize(review)
    print(aft_tokens)

    # stopword removal 152 words
    aft_remov = [w for w in aft_tokens if not w in stopwords.words('english')]
    print(aft_remov) #print(stopwords.words('english')[:155]) # to see what's in stopwords

    # stemming (어간추출) executed
    stemmer = PorterStemmer()
    aft_stem = [stemmer.stem(w) for w in aft_remov]
    print(aft_stem) #could be "print(' '.join(singles))""

    # frequency counting
    for word in aft_stem:
        count = frequency.get(word,0)
        frequency[word] = count+1

    frequency_list = frequency.keys()

for w in frequency_list:
    print(w," ", frequency[w])

    # # lemmatizing(원형복원) executed
    # lemmatizer = WordNetLemmatizer()
    # result2 = [lemmatizer.lemmatize(w2) for w2 in tokens]
    # print(result2)
    #
    # print(lemmatizer.lemmatize("left", pos ='v')) #일일히 명시 해줘야만?



    # tagged = nltk.pos_tag(tokens)
    # print(tagged[0:])

    # entities = nltk.chunk.ne_chunk(tagged)
    # print(entities)

#t = treebank.parsed_sents('wsj_0001.mrg')[0]
#t.draw()
