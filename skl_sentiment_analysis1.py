import re
import pandas as pd
from time import time

# 데이터 전처리 : 특수기호, HTML 태그 등 제거 (단, 이모티콘은 남겨둠)
def preprocessor(text) :
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)|\^.?\^', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

df = pd.read_csv('/Users/paramount/Desktop/GitHub/Steam-Review/test.csv')

stime = time()
print('전처리 시작')
df['contents'] = df['contents'].apply(preprocessor)
print('전처리 완료: 소요시간 [%d] 초' % (time() - stime))
df.head()


from mylib.tokenizer import tokenizer, tokenizer_porter

text = 'runners like running and thus they run'
print(tokenizer(text))
print(tokenizer_porter(text))
