import re
import lda
import nltk
import math
import collections
import pandas as pd
import numpy as np
from nltk import ngrams
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn import linear_model
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from matplotlib import pyplot as plt

# for i in range(62641):
# re.sub("[^a-zA-Z]", "apple", str(X.review[i]))

X = pd.read_csv('C:/Users/Paramount/Desktop/GitHub/Steam-Review/dayz/dayz 2018.7-.csv', encoding='cp949')
sens=[nltk.tokenize.sent_tokenize(X['review'][i]) for i in range(0,len(X['review']))]

names=set(X['title'])
[ x.split(' ') for x in names]
names_words = []
for x in names:
    names_words += [ word.lower() for word in x.split() ]

cnt = collections.Counter()
for x in names_words:
    cnt[x] += 1


tokens=[]
for i in range(0,len(sens)):
    token=[]
    for j in range(0,len(sens[i])):
        token+=nltk.tokenize.word_tokenize(sens[i][j])
    tokens.append(list(token))
pos=[]
for t in tokens:
    pos_tokens=[token for token, pos in nltk.pos_tag(t) if pos.startswith('RB')|pos.startswith('JJ')]
    pos.append(pos_tokens)


collections.OrderedDict(sorted(cnt.items(), key=lambda t: t[1]))


#removal stopwords
stop=nltk.corpus.stopwords.words('english')
stop+=["!","...",")","(","/",".",",","?","-","''","``","'d",":",";","***","*","%","$","@","#","&","+","~","'s","n't","'m","'d"]
additionalstop = ['game','make','un', 'es', 'juego', 'la', 'el', 'con', 'lo', 'los', 'para', 'una', 'si', 'se', 'por', 'le']
stop+= additionalstop
stop += names_words


pos2=[]
for t in tokens:
    pos_tokens=[token for token, pos in nltk.pos_tag(t) if pos.startswith('NN')|pos.startswith('VERB')]
    pos2.append(pos_tokens)


stemmer2=SnowballStemmer("english",ignore_stopwords=True)
sin_snowball=[]
for p in pos2:
    singels=[stemmer2.stem(p[i]) for i in range(0,len(p))]
    sin_snowball.append(singels)


sin_snowball2=[]
for singles in sin_snowball:
    singles2=[word for word in singles if word not in stop]
    singles2=[a for a in singles2 if len(a)!=1] #한 글자 지우기
    sin_snowball2.append(singles2)


all_words2=[]
for doc in sin_snowball2:
    all_words2+=[word for word in doc]

all_words_raw = []
for doc in pos2:
    all_words_raw +=[x for x in doc if x not in stop]


#frequency analysis
fd=nltk.FreqDist(all_words2)
fd_table=pd.DataFrame(np.array(fd.most_common(len(set(all_words2)))))
fd_table[1]=fd_table[1].apply(pd.to_numeric)
fd_table=fd_table[fd_table[1]>=100]     # 100으로 줄임
fd_table.to_csv("frequency_analysis.csv")


#fd_table.to_csv('frequency_table_nopreprocessed_cr.csv')
#remove words
sin_snowball3=[]
for singles in sin_snowball2:
    singles2=[word for word in singles if word in list(fd_table[0])]
    sin_snowball3.append(singles2)

#clean document
doc2=[]
for singeles  in sin_snowball3:
    result =  " ".join(singeles)
    doc2.append(result)
countvec=CountVectorizer()
tf_lda=countvec.fit_transform(doc2)
topic_X=tf_lda.toarray()
vocab=countvec.get_feature_names() ####################sen-topic에서 topic 검색 단어로 쓰임

#########################전처리 끝(RB,JJ)#######################################

for x in [3,4,5,6,7]:
    for s in [0.1]:
        model=lda.LDA(n_topics= x ,n_iter=500,random_state=6,alpha = s)
        model.fit(topic_X)
        topic_word=model.topic_word_
        n_top_words=20
        lda_results = []
        for i, topic_dist in enumerate(topic_word):
            topic_words=np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
            print('Topic',i,topic_words)
            lda_results.append([i,topic_words])
        lda_results = pd.DataFrame(lda_results,columns = ['Topic_N','Words'])
        lda_results.to_csv('nwords_50_CLDA_results_NNVERB%s_%s.csv' % (x, s))


score = X[X.columns[3]]

# lasso reg for sentiment dictionary

for x in range(len(score)):
    if math.isnan(score[x]) == True:
        score[x] = 0

final_lasso = linear_model.Lasso(alpha=0.0005) # 오래 걸림
final_lasso.fit(topic_X, score)

fea_score = [[feature, coef] for feature, coef in
             zip(list(countvec.get_feature_names()), list(final_lasso.coef_))]
fea_score = pd.DataFrame(np.array(fea_score))
fea_score.columns = ['feature', 'sen_score']
fea_score['sen_score'] = pd.to_numeric(fea_score['sen_score'])
fea_score = fea_score[(fea_score['sen_score'] > 0) | (fea_score['sen_score'] < 0)]

sentiment_list=list(fea_score['feature'])

# 감정 사전 개수
# 양수를 앞으로 정렬
fea_score.sort_values(by=['sen_score'],axis=0,ascending=False)

# 음수를 앞으로 정렬
# fea_score.sort_values(by=['sen_score'],axis=0)[:10]

fea_score.to_csv('sensitive_score.csv')



"""
# Process - 감정단어를 검색하고 앞뒤 n개 단어를 searching
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
n = 0
dws = []

for d, docu in enumerate(sin_snowball3):
    for plo in sentiment_list:
        plo_score = list(fea_score[fea_score['feature'] == plo]['sen_score'])[0]
        plo_idx = [i for i, w in enumerate(docu) if w == plo]
        for idx in plo_idx:
            s_idx = np.where(idx - n < 0, 0, idx - n)
            e_idx = np.where(idx + n + 1 > len(docu), len(docu), idx + n + 1)
            f_ngram = docu[s_idx:idx]
            b_ngram = docu[idx + 1:e_idx]

            if len(f_ngram) != 0:
                topic_idx = [i for i, w in enumerate(f_ngram) if w in vocab]
                if len(topic_idx) != 0:
                    topic_words = f_ngram[np.max(topic_idx)]
                    twi = vocab.index(topic_words)
                    dws.append([d, twi, plo_score])
            elif len(b_ngram) != 0:
                topic_idx = [i for i, w in enumerate(b_ngram) if w in vocab]
                if len(topic_idx) != 0:
                    topic_words = b_ngram[np.min(topic_idx)]
                    twi = vocab.index(topic_words)
                    dws.append([d, twi, plo_score])
            else:
                next
    print(d)


dwsm = np.zeros(shape=(d + 1, len(vocab)))
for i in range(0, len(dws)):
    dwsm[dws[i][0]][dws[i][1]] = dwsm[dws[i][0]][dws[i][1]] + dws[i][2]


np.savetxt("C:/Users/Paramount/Desktop/GitHub/Steam-Review/data/dwsm.csv",dwsm,delimiter = ",")

senti_table = pd.read_csv('C:/Users/Paramount/Desktop/GitHub/Steam-Review/data/dwsm.csv', encoding='cp949')

#게임/토픽별 점수
grouped=X.groupby(['title'])
# game_num=[0]+list(grouped.last().sort_values('id')['id']+1)
game_num = list(X.index)
game_topic=[]
for i in range(0,len(game_num)-1):
    if game_num[i+1] <= d+1:
        s_num=game_num[i]
        f_num=game_num[i+1]
        game_topic.append(list(np.dot(senti_table[s_num:f_num], topic_word.T).sum(axis=0)))
    else:
        next


lda_topic = pd.read_csv('C:/Users/Paramount/Desktop/GitHub/Steam-Review/data/lda_topic3.csv', encoding='cp949')
lda_topic


dataframe_temp = pd.DataFrame(game_topic)
dataframe_temp.to_csv("C:/Users/Paramount/Desktop/GitHub/Steam-Review/data/game_topic.csv", header=False, index=False)

topic = pd.read_csv('C:/Users/Paramount/Desktop/GitHub/Steam-Review/data/game_topic.csv',names=['playtime', 'genre', 'hack', 'battleroyal', 'bug', 'friends', 'dontknow'])
topic.head(10)


game_seq = X['title']
game_table = pd.concat([game_seq,topic],axis=1)

senti_game_score = game_table[game_table.columns[1:]].sum()

# .groupby(game_table['title']).mean()

senti_game_score
"""
