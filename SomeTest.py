import re
import numpy as np
import torch
import word2vec
import nltk

word2vec.word2vec(".\\Content.txt",".\\tem.bin",size=150,verbose=True,sample=0,
                  negative=0,min_count=0,cbow=0,window=10)
import torch.nn as nn
import torch.nn.functional as F
nn.CrossEntropyLoss()
a = nltk.word_tokenize("DDI-MedLine.d78.s1	The fluoroquinolones are a rapidly growing class of antibiotics with a broad "
                   "spectrum of activity against gram-negative and some gram-positive aerobic bacteria.")
print(a)

