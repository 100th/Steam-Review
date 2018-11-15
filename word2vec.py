import numpy as np
import pandas as pd
import nltk.tokenize
import lda
# from collections import Counter
import collections
from nltk.stem.snowball import SnowballStemmer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import euclidean_distances

data = pd.read_csv('C:/Users/Paramount/Desktop/GitHub/Steam-Review/data/reviews_removed.csv', encoding='cp949')
data.head(3)

names = set(data['kind_of_games'])

names_words=[]
for x in names:
    if type(x) == str:
        names_words += [word.lower() for word in x.split()]

cnt = collections.Counter()

for x in names_words:
    cnt[x] += 1

# 데이터 전처리

review = [re for re in data['review']]

# stemming
stemmer = SnowballStemmer("english", ignore_stopwords=True)

stem_review = []
for ii, words in enumerate(review):
    # print("stemmer--> ", ii, "/", len(review))
    if type(words) == str:
        stem_review.append(stemmer.stem(words))

stem_review = np.array(stem_review)
def eli_stop(string, stop_word, rlc = ''):
    for sw in stop_word:
        string= string.replace(sw, rlc)
    return string


eli = ['nothing','better','way','nice','new','great','awesome','more','lot', 'many', 'other', 'time', 'same', 'everything', 'hours', 'things', 'bit', 'most','much',"good", "bad","!", "...", ")", "(", "/", ".", ",", "?", "-", "''", "``", "'d", ":", ";", "***", "*", "%", "$", "@", "#", "&", "+", "~", "'s", "n't", "'m", "game","'"]
eli_review_list = [eli_stop(str(rm),eli) for rm in stem_review]

sens = [nltk.tokenize.sent_tokenize(eli_list) for eli_list in eli_review_list]

tokens = []
for i in range(0, len(sens)):
    token = []
    for j in range(0, len(sens[i])):
        token += nltk.tokenize.word_tokenize(sens[i][j])
    tokens.append(list(token))

pos = []
for t in tokens:
    pos_tokens = [token for token, pos in nltk.pos_tag(t) if pos.startswith('NN')|pos.startswith('JJ')]
    pos.append(pos_tokens)

collections.OrderedDict(sorted(cnt.items(), key=lambda t: t[1]))

np.shape(pos)
len(np.unique(pos))

# removal stopwords
stop = nltk.corpus.stopwords.words('english')

additionalstop = ['lot', 'many', 'other', 'time', 'same', 'everything', 'hours', 'things', 'bit', 'most','much','more','good','bad','sure','fun','great','game','make', 'un', 'es', 'juego', 'la', 'el', 'con', 'lo', 'los', 'para', 'una', 'si', 'se', 'por', 'le','rollerskates']
stop += additionalstop
stop += names_words

removed_words = [a for a in pos if a not in stop]
ree = [a for i in removed_words for a in i]
removed_voc = list(set(ree))

len(set(removed_voc))


one_list = []
for i in ree:
    one_list.append([i[l] for l in range(len(i)) if len(i[l])<2])

clean_words = [a for a in removed_words if a not in one_list]

final_words = []
for a_line in removed_words:
        final_words.append([a_line[a_num] for a_num in range(len(a_line)) if len(str(a_line[a_num]))>1])

final_words_list = [a for i in final_words for a in i]

# 매트릭스 생성
# Word2Vec
embedding_model = Word2Vec(final_words, size=100, window = 3,min_count=1)

# print(embedding_model)

# Word2Vec + LDA matrix

# word2vec mat
em_mat=[]
row_name = []
for ii, list_words in enumerate(final_words):
    for i in list_words:
        if i in embedding_model:
            em_mat.append(embedding_model[str(i)])
            row_name.append(i)
    # print('turn-->', ii, "/", len(final_words))

mat_list = pd.DataFrame(em_mat,index=[row_name])

# print(len(row_name))

# mat_list = pd.read_csv(r"C:\Users\User\PycharmProjects\word2vec\word2vec_word.csv",encoding='cp949')
# mat_list.head(3)

countvec = CountVectorizer(analyzer='word',vocabulary=list(set(final_words_list)))
count = countvec.fit_transform(final_words_list)
topic_vocab=countvec.get_feature_names()
titles = names


# LDA modeling
model = lda.LDA(n_topics=12, n_iter=500, random_state=6, alpha=0.1)
model.fit(count)
topic_word = np.array(topic_vocab)[np.argsort(model.topic_word_)][:,:-1]


# lda percent
word_topic = np.array(np.argsort(model.topic_word_))[:,::-1]
np.shape(word_topic)


# combine model - topic vector --------------------
lda_mat = []
for ii, wor in enumerate(topic_word):
    # print("turn--> ", ii, '/', len(topic_word))
    wor_vec = []
    for wo in wor:
        wor_vec.append(embedding_model[wo])
    lda_mat.append(wor_vec)

topic_range=range(12)

theta_i = []
for n, i in enumerate(topic_range):
    # print("turn--> ", n, "/", len(topic_range))
    topic_theta = [topic/sum(word_topic[i]) for topic in word_topic[i]]
    theta_i.append(topic_theta)

topic_vectors = []
for a in range(12):
    cal_theta = [lda_mat[a][i]*theta_i[a][i] for i in range(len(lda_mat[a]))]
    topic_vectors.append(cal_theta)

topic_vec_final = [np.array(topic_vectors[i]).sum(axis=0) for i in topic_range]


# comb modeling - document vector ---------------------------------

fi_sent_word = pd.DataFrame(final_words)

game = np.array(data['kind_of_games'])
len(names)
data.columns[:]

review_data = data['kind_of_games']
review_data = pd.DataFrame(review_data)

total_docu_vec = []
for n, docu in enumerate(final_words):
    # print('turn--> ', n, '/', len(final_words))
    docu_vector = []
    for wo in docu:
        docu_vector.append(embedding_model[wo])
    total_docu_vec.append(docu_vector)

game_name = list(names)
game = np.array(data['kind_of_games'])

game_re_vec = []
for n, na in enumerate(game_name):
    print("turn--> ", n, "/", len(names))
    what_game = []
    for ga in range(len(game)):
        if na == game[ga]:
            what_game.append(total_docu_vec[ga])
    game_re_vec.append(what_game)

sum_mat = []
for line in range(len(game_re_vec)):
    print('turn--> ', line, '/', len(game_re_vec))
    game_line = [a for i in game_re_vec[line] for a in i]
    sum_line = []
    for num in range(100):
        tmp = [sum(game_line[n][num] for n in range(len(game_line)))/len(game_line)]
        sum_line.append(tmp)
    sum_mat.append(sum_line)


# calculate distance ------------------------------------

dist_mat = []
for p in range(len(sum_mat)):
    print("turn--> ", p, '/', len(sum_mat))
    x_vec = [a for i in sum_mat[p] for a in i]
    dist = [euclidean_distances(x_vec,topic_vectors[n]).tolist() for n in range(len(topic_vectors))]
    bb = [a for i in dist for a in i]
    dist_mat.append([a for i in bb for a in i])

print(dist_mat)

# select topic ----------------------------------------
selected_topic = [np.argsort(dist_mat[n])[:5] for n in range(len(dist_mat))]

# print(selected_topic)
