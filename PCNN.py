import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class PCNN (nn.Module) :
    """
        The shape of input tensor is [b ,length]
    """

    def __init__(self,W,embeddingDim,dropoutPro,labelNumbers,embeddingWeight):
        super(PCNN,self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddingWeight).float())
        self.conv = nn.Conv2d(in_channels=1,out_channels=W,
                              kernel_size=(3,embeddingDim + 2),padding=(1,0))
        self.dropout = nn.Dropout(dropoutPro)
        self.dense = nn.Linear(3*W,labelNumbers)


    def forward(self, x, positionOfEntity, positionEmbedding,device):
        """
        :param x: Input tensor,[b,length], it needs to be long type.
        :param positionOfEntity: The shape of positionOfEntity is [b , 2].
               It corresponds the position of entities in each data. It is a numpy array.
        :param positionEmbedding : [b , length , 2]
        :param device : cpu or gpu
        :return: result
        """
        ### [b , length , embeddingDim]
        x = self.embedding(x)
        positionEmbedding = positionEmbedding
        x = torch.cat([x,positionEmbedding],dim=-1)
        x = torch.reshape(x,shape=[x.shape[0] , 1 , x.shape[1], x.shape[2]])
        ### [b , outChannels , length , 1]
        x = self.conv(x)
        b = x.shape[0]
        outChannels = x.shape[1]
        length = x.shape[2]
        ### The shape of x is [b , outChannels , length]
        x = torch.reshape(x,shape=[-1,outChannels,length])
        ### The shape of one element of batchTensorList is [1 , outChannels , length]
        batchTensorList = torch.chunk(x,chunks=b,dim=0)
        batchOutList = []
        for i,element in enumerate(batchTensorList):
            ### The shape of elementS is [outChannels , length]
            elementS = torch.squeeze(element)
            mask1 = np.zeros(shape=[outChannels,length],dtype=np.float32)
            mask1[:,positionOfEntity[i,0] + 1:] = -1e5
            mask2 = np.zeros(shape=[outChannels,length],dtype=np.float32)
            mask2[:,:positionOfEntity[i,0] + 1] = -1e5
            mask2[:,positionOfEntity[i,1] + 1 :] = -1e5
            mask3 = np.zeros(shape=[outChannels,length],dtype=np.float32)
            mask3[:,:positionOfEntity[i,1] + 1] = -1e5
            mask1 = torch.from_numpy(mask1).float().to(device)
            mask2 = torch.from_numpy(mask2).float().to(device)
            mask3 = torch.from_numpy(mask3).float().to(device)
            mask1Ele = torch.add(mask1,elementS.clone())
            mask2Ele = torch.add(mask2,elementS.clone())
            mask3Ele = torch.add(mask3,elementS.clone())
            ### [outChannels]
            mask1EleMaxPool = torch.squeeze(F.max_pool2d(torch.reshape(mask1Ele,shape=[1,1,outChannels,length]),(1,length),stride = 1,padding = 0))
            mask2EleMaxPool = torch.squeeze(F.max_pool2d(torch.reshape(mask2Ele,shape=[1,1,outChannels,length]),(1,length),stride = 1,padding = 0))
            mask3EleMaxPool = torch.squeeze(F.max_pool2d(torch.reshape(mask3Ele,shape=[1,1,outChannels,length]),(1,length),stride = 1,padding = 0))
            concatTensor = torch.cat((mask1EleMaxPool,mask2EleMaxPool,mask3EleMaxPool),dim=0)
            batchOutList.append(concatTensor)
        preTensor = torch.stack(batchOutList,dim=0)
        ### [b , 3*outChannels]
        tanhTrans = torch.tanh(preTensor)
        dropout = self.dropout(tanhTrans)
        liner = self.dense(dropout)
        return F.softmax(liner,dim=1)

if __name__ == "__main__":
    testInput = torch.from_numpy(np.ones(shape=[3,10],dtype=np.long))
    testPositionOfEntity = np.array([[3,4],
                                     [5,6],
                                     [1,2]])
    testEmbeddingWeight = np.ones(shape=[10,5],dtype=np.float32)
    testPositionEmbedding = np.random.rand(3,10,2)
    model = PCNN(W=3,embeddingDim=5,dropoutPro=0.5,labelNumbers=4,embeddingWeight=testEmbeddingWeight)
    result = model(testInput,testPositionOfEntity,testPositionEmbedding)
    print(result)