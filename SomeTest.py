import re
import numpy as np
import torch
import word2vec
import nltk
import torch.nn as nn
import torch.nn.functional as F



testData = torch.from_numpy(np.random.rand(3,4,5,6)).float()
conv = torch.nn.Conv2d(in_channels=4,out_channels=7,kernel_size=3,padding=(1,1))
print(conv(testData).shape)

