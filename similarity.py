# https://datascienceschool.net/view-notebook/6927b0906f884a67b0da9310d3a581ee/
# https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt

import nltk
from nltk.corpus import movie_reviews
from gensim.models.word2vec import Word2Vec

nltk.download('movie_reviews')

# 여기에 우리의 리뷰 데이터를 정제해서 넣으면 됨

sentences = [list(s) for s in movie_reviews.sents()]
# sentences[0]

%time
model = Word2Vec(sentences)
model.init_sims(replace=True)

# 두 단어의 유사도 계산
model.wv.similarity('actor', 'actress')

# 가장 유사한 단어를 출력
model.wv.most_similar("actor")

# she + (actor - actress) = he
model.wv.most_similar(positive=['she', 'actor'], negative='actress', topn=1)

# (참고로 한국어도 가능)
