# https://datascienceschool.net/view-notebook/6927b0906f884a67b0da9310d3a581ee/
# https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt
# https://github.com/corazzon/petitionWrangling

import nltk
from nltk.corpus import movie_reviews
from gensim.models.word2vec import Word2Vec

from __future__ import print_function
import nltk
import numpy
import openpyxl
from nltk.stem.porter import * # for Stemming
from nltk.stem import PorterStemmer # ProterStemmer
from nltk.stem import WordNetLemmatizer # for lemmatizing
from nltk.corpus import stopwords # for stopwords removal

wb=openpyxl.load_workbook('newtest.xlsx', read_only=False)
ws=wb.active

sentences = []

for r in ws.rows:
    row_index = r[0].row # 행 인덱스?
    review = r[4].value
    # print(review)

    # tokenization start
    aft_tokens = nltk.word_tokenize(review)
    # print(aft_tokens)

    # stopword removal 152 words
    aft_remov = [w for w in aft_tokens if not w in stopwords.words('english')]
    # print(aft_remov) #print(stopwords.words('english')[:155]) # to see what's in stopwords
    sentences.append(aft_remov)

    # stemming (어간추출) executed
    # stemmer = PorterStemmer()
    # aft_stem = [stemmer.stem(w) for w in aft_remov]
    # print(aft_stem) #could be "print(' '.join(singles))""
    # sentences.append(aft_stem)

# nltk.download('movie_reviews')
# sentences = [list(s) for s in movie_reviews.sents()]

# sentences[:2]

%time
model = Word2Vec(sentences)
model.init_sims(replace=True)

# 두 단어의 유사도 계산
model.wv.similarity('You', 'game')

# 가장 유사한 단어를 출력
model.wv.most_similar("game")

# she + (actor - actress) = he
model.wv.most_similar(positive=['she', 'actor'], negative='actress', topn=1)

# (참고로 한국어도 가능)
