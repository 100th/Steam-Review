# import nltk
# nltk.download()
# from nltk.tokenize import word_tokenize
# print(word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

import re
s = '123abc'
i = re.findall(r'^\w', s)[0]
print(i)
