import pandas as pd
import os
import numpy as np
# from progbar import ProgBar

path = 'C:/Users/paramount/Desktop/GitHub/Steam-Review/aclImdb/'

# pbar = ProgBar(50000)
labels = {'pos':1, 'neg':0}
df = pd.DataFrame()
for s in ('test', 'train'):
    for name in ('pos', 'neg'):
        subpath = '%s/%s' %(s, name)
        dirpath = path + subpath
        for file in os.listdir(dirpath):
            with open(os.path.join(dirpath, file), 'r') as f:
                txt = f.read()
            df = df.append([[txt, labels[name]]], ignore_index=True)
            # pbar.update()

df.columns = ['review', 'sentiment']

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('C:/Users/paramount/Desktop/GitHub/Steam-Review/movie_review.csv', index=False)

df = pd.DataFrame()
df= = pd.read_csv('movie_review.csv')
print(df.head())
print(df.taiil())
