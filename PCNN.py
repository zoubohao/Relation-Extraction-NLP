import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ConvolutionCombine(nn.Module) :

    def __init__(self,inChannels,outChannels,group = 4):
        super(ConvolutionCombine,self).__init__()
        self.conv = nn.Conv2d(in_channels=inChannels,out_channels=outChannels,kernel_size=1)
        self.bn = nn.GroupNorm(group,outChannels)
        self.trans = nn.PReLU(init=0.)

    def forward(self, x):
        xI = x.clone()
        x = self.conv(x)
        x = self.bn(x)
        x = self.trans(x)
        return torch.add(x,xI)

class PCNN (nn.Module) :
    """
        The shape of input tensor is [b ,length]
    """

    def __init__(self,convNumber,denseNumber,W,embeddingDim,labelNumbers,embeddingWeight,dropoutPro = 0.4, convolutionKernelHeight = 5):
        super(PCNN,self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddingWeight)
        self.denseSeq = nn.Sequential()
        for c in range(denseNumber) :
            if c == 0 :
                self.denseSeq.add_module("dense" + str(c),module=nn.Linear(in_features=3 * W,out_features=W))
            else:
                self.denseSeq.add_module("dense" + str(c),
                                         module=nn.Linear(in_features=W,out_features=W))
            self.denseSeq.add_module("BatchNormLayer" + str(c), module=nn.BatchNorm1d(W))
            self.denseSeq.add_module("Activation" + str(c),module=nn.PReLU(init=0.))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=W,
                                       kernel_size=(convolutionKernelHeight, embeddingDim + 2))
        self.BN1 = nn.BatchNorm2d(W)
        self.PRelu1 = nn.PReLU(init=0.)
        self.convSeq = nn.Sequential()
        for c in range(convNumber):
            self.convSeq.add_module("ConvComb" + str(c),module=ConvolutionCombine(inChannels=W,outChannels=W))
        self.finalDropOut = nn.Dropout(dropoutPro)
        self.dense = nn.Linear(W,labelNumbers)


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
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.PRelu1(x)
        xI = x.clone()
        x = self.convSeq(x)
        x = torch.add(x,xI)
        ###
        b = x.shape[0]
        outChannels = x.shape[1]
        length = x.shape[2]
        ### The shape of x is [b , outChannels , length]
        x = torch.reshape(x,shape=[-1,outChannels,length])
        ### The shape of one element of batchTensorList is [1 , outChannels , length]
        ### torch.split()
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
        seq = self.denseSeq(preTensor)
        dr = self.finalDropOut(seq)
        liner = self.dense(dr)
        return F.softmax(liner,dim=1)

if __name__ == "__main__":
    device0 = torch.device("cuda:0")
    testInput = torch.from_numpy(np.ones(shape=[3,10],dtype=np.long)).long().to(device0)
    testPositionOfEntity = np.array([[3,4],
                                     [5,6],
                                     [1,2]])
    testEmbeddingWeight = torch.from_numpy(np.ones(shape=[10,5],dtype=np.float32)).float().to(device0)
    testPositionEmbedding = torch.from_numpy(np.array(np.random.rand(3,10,2),dtype=np.float32)).float().to(device0)
    model = PCNN(2,W=3,embeddingDim=5,dropoutPro=0.5,labelNumbers=4,embeddingWeight=testEmbeddingWeight).to(device0)
    result = model(testInput,testPositionOfEntity,testPositionEmbedding,device0)
    print(result)