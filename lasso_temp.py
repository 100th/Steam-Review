import numpy as np
import pandas as pd
import math
from sklearn import linear_model

X = pd.read_csv('C:/Users/Paramount/Desktop/GitHub/Steam-Review/data/reviews_sample.csv', encoding='cp949')

X.head(3)

score = X[X.columns[3]]

# lasso reg for sentiment dictionary

for x in range(len(score)):
    if math.isnan(score[x]) == True:
        score[x] = 0

final_lasso = linear_model.Lasso(alpha=0.0005)
final_lasso.fit(topic_X, score)

fea_score = [[feature, coef] for feature, coef in
             zip(list(countvectorizer.get_feature_names()), list(final_lasso.coef_))]
fea_score = pd.DataFrame(np.array(fea_score))
fea_score.columns = ['feature', 'sen_score']
fea_score['sen_score'] = pd.to_numeric(fea_score['sen_score'])
fea_score = fea_score[(fea_score['sen_score'] > 0) | (fea_score['sen_score'] < 0)]



# 감정 사전 개수 : 2568
fea_score = pd.read_csv(r"C:\Users\User\PycharmProjects\word2vec\new_fea_score.csv",encoding='cp949')
fea_score.sort_values(by=['sen_score'],axis=0,ascending=False)[:10]



fea_score.sort_values(by=['sen_score'],axis=0)[:10]



# Process - 감정단어를 검색하고 앞뒤 n개 단어를 searching

n = 3
dws = []
for d, docu in enumerate(snowball3_senti):
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




senti_table = pd.read_csv(r"C:\Users\User\Desktop\game_word2vec\dwsm.csv",encoding='cp949')

#게임/토픽별 점수
grouped=X.groupby(['kind_of_games'])
game_num=[0]+list(grouped.last().sort_values('id')['id']+1)
game_num = list(X.index)
game_topic=[]
for i in range(0,len(game_num)-1):
    if game_num[i+1] <= d+1:
        s_num=game_num[i]
        f_num=game_num[i+1]
        game_topic.append(list(np.dot(senti_table[s_num:f_num],ratio_topic.T).sum(axis=0)))
    else:
        next


lda_topic = pd.read_csv(r"C:\Users\User\Desktop\game_word2vec\lda_topic.csv",encoding = 'cp949')
lda_topic



path="C:/Users/User/Desktop/game_word2vec/"
topic = pd.read_csv(path+'game_topic_sen.csv',names=['graphic','story','character','first_person','level_design','levle_design2','perfection','graphic2','series','AI','play','play2'])
topic.head(10)



game_seq = X['kind_of_games']
game_table = pd.concat([game_seq,topic],axis=1)

senti_game_score = game_table[game_table.columns[1:]].groupby(game_table['kind_of_games']).sum()



senti_game_score = pd.read_csv(r"C:\Users\User\PycharmProjects\word2vec\topic_senti_score.csv",encoding='cp949')
senti_game_score.head(10)
