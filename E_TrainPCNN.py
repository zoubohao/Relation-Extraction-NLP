import torch
import torch.nn as nn
import torch.optim as optim
import re
import numpy as np
from PCNN import PCNN
import math
from  sklearn.metrics import f1_score


### Config
embeddingWeightFilePath = "D:\MyProgram\Data\dataset\ddi\\all_data\\weight.txt"
testingDataPath = "D:\MyProgram\Data\dataset\ddi\\all_data\\testProcessedData.txt"
labelInforFilePath = "D:\MyProgram\Data\dataset\ddi\\all_data\\relationInfor.txt"
### The parameters below  must be a folder path , not a file path.
relationTrainDataFolderPath = "D:\MyProgram\Data\dataset\ddi\\all_data\\" # It from the output folder of D step
checkPointAndPredictFileFolder = "D:\MyProgram\Data\dataset\ddi\\all_data\\"
### Model config
epoch = 16
timesInOneEpoch = 9000
lr = 1e-3
wordsInOneSentence = 90
checkPointTimes = 9000
decayRate = 0.96
decayTimes = 9000
displayTimes = 1000


### Operation
if relationTrainDataFolderPath.endswith("\\") is False:
    relationTrainDataFolderPath = relationTrainDataFolderPath + "\\"
if checkPointAndPredictFileFolder.endswith("\\") is False:
    checkPointAndPredictFileFolder = checkPointAndPredictFileFolder + "\\"
vocabularyMap = {}
vocabularyWeight = []
lenOfEmbeddingDim = 0
### If there is no more words in one sentence that can not compose one completed training data, we should
### padding zero to this sentence. And the zero list is in the first row of weight.
with open(embeddingWeightFilePath,"r") as eh:
    for i , line in enumerate(eh):
        oneLine = line.strip("\n")
        inforList = re.split("\\s+", oneLine)
        if i == 0:
            lenOfEmbeddingDim = len(inforList[1:-1])
            vocabularyWeight.append(np.zeros(shape=[lenOfEmbeddingDim], dtype=np.float32))
        vocab = inforList[0]
        vocabularyMap[vocab] = i + 1
        vector = list(map(float, inforList[1:-1]))
        if len(vector) == lenOfEmbeddingDim:
            vocabularyWeight.append(vector)
device = torch.device("cuda:0")
vocabularyWeight = torch.from_numpy(np.array(vocabularyWeight,dtype=np.float32)).float().to(device)
print("There are " + str(len(vocabularyMap)) + " words in map.")
print("The shape of embedding weight is ",vocabularyWeight.shape)
labelInforMap = {}
with open(labelInforFilePath,"r") as lh:
    for line in lh :
        inforList = re.split("\\s+",line.strip("\n"))
        labelInforMap[inforList[0]] = int(inforList[1])
labelNumber = len(labelInforMap)
labelList = list(labelInforMap.keys())
### trainModel
### Because the training data is too huge to push its all in memory, so , the training will be
### in the loop of reading data.
def DataGenerator(dataPath):
    while True:
        with open(dataPath, "r") as dh:
            d = 0
            tempDataList = []
            for thisLine in dh:
                oneDataLine = thisLine.strip()
                if oneDataLine != "":
                    tempDataList.append(oneDataLine)
                    d += 1
                    if d % 4 == 0 and d != 0:
                        sentence = re.split("\\s+", tempDataList[0])
                        aPosiRe = re.split("\\s+", tempDataList[1])
                        bPosiRe = re.split("\\s+", tempDataList[2])
                        if sentence[-1] == "":
                            sentence = sentence[0:-1]
                        if aPosiRe[-1] == "":
                            aPosition = list(map(int, aPosiRe[0:-1]))
                        else:
                            aPosition = list(map(int, aPosiRe[0:]))
                        if bPosiRe[-1] == "":
                            bPosition = list(map(int, bPosiRe[0:-1]))
                        else:
                            bPosition = list(map(int, bPosiRe[0:]))
                        labelData = tempDataList[3]
                        assert len(sentence) == len(aPosition) == len(bPosition), "The length of sentence and position are not same."
                        oneSentenceData = []
                        onePositionData = []
                        for p in range(len(sentence)):
                            oneSentenceData.append(vocabularyMap[sentence[p]])
                            onePositionData.append([aPosition[p], bPosition[p]])
                        labelOneHot = np.zeros(shape=[labelNumber], dtype=np.float32)
                        labelOneHot[labelList.index(labelData)] = 1.
                        # print(tempDataList[0])
                        # print("aPosition : ",aPosition)
                        # print("bPosition : ",bPosition)
                        # print(labelData)
                        tempDataList = []
                        d = 0
                        ### shape of each return:
                        ### one sentence: [length] ,
                        ### positionEmbedding : [length , 2],
                        ### entityPosition : [2] ,
                        ### relationData : [labelNumber]
                        yield oneSentenceData, onePositionData, \
                              np.array([aPosition.index(0), bPosition.index(0)], dtype=np.int64), labelOneHot
def TestDataGenerator():
    with open(testingDataPath,"r") as rh :
        d = 0
        tempDataList = []
        for thisLine in rh:
            oneDataLine = thisLine.strip()
            if oneDataLine != "":
                tempDataList.append(oneDataLine)
                d += 1
                if d % 4 == 0 and d != 0:
                    sentence = re.split("\\s+", tempDataList[0])
                    aPosiRe = re.split("\\s+", tempDataList[1])
                    bPosiRe = re.split("\\s+", tempDataList[2])
                    if sentence[-1] == "":
                        sentence = sentence[0:-1]
                    if aPosiRe[-1] == "":
                        aPosition = list(map(int, aPosiRe[0:-1]))
                    else:
                        aPosition = list(map(int, aPosiRe[0:]))
                    if bPosiRe[-1] == "":
                        bPosition = list(map(int, bPosiRe[0:-1]))
                    else:
                        bPosition = list(map(int, bPosiRe[0:]))
                    labelData = tempDataList[3]
                    assert len(sentence) == len(aPosition) == len(
                        bPosition), "The length of sentence and position are not same."
                    oneSentenceData = []
                    onePositionData = []
                    for p in range(len(sentence)):
                        oneSentenceData.append(vocabularyMap[sentence[p]])
                        onePositionData.append([aPosition[p], bPosition[p]])
                    labelOneHot = np.zeros(shape=[labelNumber], dtype=np.float32)
                    labelOneHot[labelList.index(labelData)] = 1.
                    # print(tempDataList[0])
                    # print("aPosition : ",aPosition)
                    # print("bPosition : ",bPosition)
                    # print(labelData)
                    tempDataList = []
                    d = 0
                    ### shape of each return:
                    ### one sentence: [length] ,
                    ### positionEmbedding : [length , 2],
                    ### entityPosition : [2] ,
                    ### relationData : [labelNumber]
                    yield oneSentenceData, onePositionData, \
                          np.array([aPosition.index(0), bPosition.index(0)], dtype=np.int64), labelOneHot , sentence
pcnnModel = PCNN(denseNumber=1,W=16,embeddingDim=lenOfEmbeddingDim,dropoutPro=0.5,labelNumbers=labelNumber,embeddingWeight=vocabularyWeight).to(device)
# labelWeight = []
# for i,number in enumerate(labelList):
#     if i == 0 :
#         labelWeight.append(1. / labelInforMap[number])
#     else:
#         labelWeight.append(1. / labelInforMap[number])
# print(labelWeight)
crit = nn.CrossEntropyLoss(reduction="mean").to(device)
opt = optim.Adam(pcnnModel.parameters(),lr= lr,weight_decay=0.001,eps=1e-6)
dataGeneratorList = []
for label in labelList:
    dataGeneratorList.append(DataGenerator(relationTrainDataFolderPath + label + ".txt"))
trainingTimes = 0
pcnnModel.train()
for e in range(epoch):
    for timeTh in range(timesInOneEpoch):
        sentenceData = []
        positionEmbeddingData = []
        entityPositionData = []
        relationLabelData = []
        for generator in dataGeneratorList:
            oneSentence, onePositionEmb, oneEntityPosition, oneLabel = generator.__next__()
            if len(oneSentence) < wordsInOneSentence:
                paddingZeros = [0 for i in range(wordsInOneSentence - len(oneSentence))]
                sentenceData.append(oneSentence + paddingZeros)
                paddingZerosPosition = [[0, 0] for i in range(wordsInOneSentence - len(oneSentence))]
                positionEmbeddingData.append(onePositionEmb + paddingZerosPosition)
                entityPositionData.append(oneEntityPosition)
                relationLabelData.append(np.squeeze(np.argmax(oneLabel)))
            else:
                if oneEntityPosition[0] > wordsInOneSentence and oneEntityPosition[1] > wordsInOneSentence:
                    continue
                else:
                    sentenceData.append(oneSentence[0:wordsInOneSentence])
                    positionEmbeddingData.append(onePositionEmb[0:wordsInOneSentence])
                    entityPositionData.append(oneEntityPosition)
                    relationLabelData.append(np.squeeze(np.argmax(oneLabel)))
        sentenceInput = torch.from_numpy(np.array(sentenceData,dtype=np.long)).long().to(device)
        posiEmbed = torch.from_numpy(np.array(positionEmbeddingData,dtype=np.float32)).float().to(device)
        entityPosi = torch.from_numpy(np.array(entityPositionData,dtype=np.int)).int().to(device)
        relaLabel = torch.from_numpy(np.array(relationLabelData,dtype=np.long)).long().to(device)
        # print(sentenceInput.shape)
        # print(posiEmbed.shape)
        # print(entityPosi.shape)
        # print(relaLabel.shape)
        # print("#################")
        opt.zero_grad()
        predict = pcnnModel(sentenceInput, entityPosi, posiEmbed,device)
        loss = crit(predict, relaLabel)
        loss.backward()
        opt.step()
        trainingTimes += 1
        if trainingTimes % displayTimes == 0:
            print("#############")
            print("Training Times ", trainingTimes)
            print("Loss is ", loss)
            stateDic = opt.state_dict()
            thisLR = stateDic["param_groups"][0]["lr"]
            print("Learning rate is ",thisLR)
            print("Predict is ", predict)
            print("Label is ", relaLabel)
        if trainingTimes % decayTimes == 0 and trainingTimes != 0 :
            lr = lr * math.pow(decayRate, trainingTimes / decayTimes + 0.0)
            stateDic = opt.state_dict()
            stateDic["param_groups"][0]["lr"] = lr
            opt.load_state_dict(stateDic)
        if trainingTimes % checkPointTimes == 0 and trainingTimes != 0:
            trueLabels = []
            predictLabels = []
            print("Begin to test model .")
            count = 0
            allTest = 0
            pcnnModel.eval()
            ph = open( checkPointAndPredictFileFolder + "ckpt_" + str(trainingTimes) + "_Predict.txt","w")
            for testSen,testPosiEmbed,testEntiPosi,testLabel , testWordsSentence in TestDataGenerator():
                for word in testWordsSentence :
                    ph.write(word + " ")
                ph.write("\n")
                ph.write(testWordsSentence[testEntiPosi[0]] + "\t" + testWordsSentence[testEntiPosi[1]] + "\n")
                testSentenceData = []
                testPositionEmbeddingData = []
                testEntityPositionData = []
                if len(testSen) < wordsInOneSentence:
                    paddingZeros = [0 for i in range(wordsInOneSentence - len(testSen))]
                    testSentenceData.append(testSen + paddingZeros)
                    paddingZerosPosition = [[0, 0] for i in range(wordsInOneSentence - len(testSen))]
                    testPositionEmbeddingData.append(testPosiEmbed + paddingZerosPosition)
                else:
                    if testEntiPosi[0] > wordsInOneSentence or testEntiPosi[1] > wordsInOneSentence:
                        continue
                    else:
                        testSentenceData.append(testSen[0:wordsInOneSentence])
                        testPositionEmbeddingData.append(testPosiEmbed[0:wordsInOneSentence])
                testEntityPositionData.append(testEntiPosi)
                testsentenceInput = torch.from_numpy(np.array(testSentenceData,dtype=np.long)).long().to(device)
                testposiEmbed = torch.from_numpy(np.array(testPositionEmbeddingData,dtype=np.float32)).float().to(device)
                testentityPosi = torch.from_numpy(np.array(testEntityPositionData,dtype=np.int)).int().to(device)
                # print(testsentenceInput)
                # print(testPosiEmbed)
                # print(testposiEmbed)
                # print(testentityPosi)
                testPredict = np.argmax(np.squeeze(pcnnModel(testsentenceInput,testentityPosi,testposiEmbed,device).cpu().detach().numpy()))
                testTrue = np.argmax(testLabel)
                ph.write("Predict : " + labelList[int(testPredict)] + "\n")
                ph.write("Label : " + labelList[int(testTrue)] + "\n")
                trueLabels.append(testTrue)
                predictLabels.append(testPredict)
                if allTest % 100 == 0:
                    print(allTest)
                    print("PCNN predict is ", testPredict)
                    print("True label is ", testTrue)
                if testPredict == testTrue:
                    count += 1
                allTest += 1
            ph.close()
            microF1 = f1_score(y_pred=predictLabels,y_true=trueLabels,average='micro')
            print("Micro F1 SCORE : ",microF1)
            torch.save(pcnnModel.state_dict(),checkPointAndPredictFileFolder +
                       "ckpt_" + str(trainingTimes) + "_ACC" + str(count * 1.0 / allTest) + "_MicroF1" + str(microF1) + ".pt")
            pcnnModel.train()





