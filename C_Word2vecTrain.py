import word2vec
import re

### Config
trainDataPath = "D:\MyProgram\Data\dataset\ddi\\all_data\\train_data.txt"
trainProcessedDataOutputPath = "D:\MyProgram\Data\dataset\ddi\\all_data\\trainProcessedData.txt"
testDataPath = "D:\MyProgram\Data\dataset\ddi\\all_data\\test_data.txt"
testProcessedDataOutputPath = "D:\MyProgram\Data\dataset\ddi\\all_data\\testProcessedData.txt"
weightOutputPath = "D:\MyProgram\Data\dataset\ddi\\all_data\\weight.txt"
vocabularyStatisticInforPath = "D:\MyProgram\Data\dataset\ddi\\all_data\\vocabularyInfor.txt"
relationLabelStatisticInforPath = "D:\MyProgram\Data\dataset\ddi\\all_data\\relationInfor.txt"
### It must be a folder path , not a file path.
temporaryFileFolder = "D:\MyProgram\Data\dataset\ddi\\all_data\\"
embeddingDim =150

### Operation
if temporaryFileFolder.endswith("\\") is False:
    temporaryFileFolder = temporaryFileFolder + "\\"
content = ""
patternOfWord = "[a-zA-Z]{1,}"
vocabMap = {}
relationMap ={}
print("Train data begins to process.")
with open(trainDataPath,"r") as rh:
    with open(trainProcessedDataOutputPath,"w") as wh:
        tempList = []
        count = 0
        oneLine = rh.readline()
        tempList.append(oneLine.strip("\n"))
        count += 1
        breakSignal = False
        while breakSignal is False :
            if count % 1000 == 0:
                print(count)
            line = rh.readline()
            if line == "" :
                breakSignal = True
            line = line.strip("\n")
            tempList.append(line)
            count += 1
            if count % 4 == 0 :
                sentence = tempList[0]
                entities = tempList[1]
                relation = tempList[2]
                entitiesList = re.split("\t",entities)
                wordList = []
                for word in re.split("\\s+",sentence):
                    if word.__contains__(".") is False:
                        matchObj = re.match(patternOfWord, word)
                        if matchObj is not None:
                            matchStr = matchObj.group().lower()
                            if matchStr != "ddi":
                                if vocabMap.__contains__(matchStr) is False:
                                    vocabMap[matchStr] = 1
                                else:
                                    vocabMap[matchStr] = vocabMap[matchStr] + 1
                                wordList.append(matchStr)
                if wordList.__contains__("DRUGA".lower()) is False or wordList.__contains__("DRUGB".lower()) is False:
                    tempList = []
                    continue
                drugAPosition = wordList.index("DRUGA".lower())
                drugBPosition = wordList.index("DRUGB".lower())
                if relationMap.__contains__(relation.lower()) is False:
                    relationMap[relation.lower()] = 1
                else:
                    relationMap[relation.lower()] = relationMap[relation.lower()] + 1
                entity1List = re.split("\\s+",entitiesList[0])
                entity2List = re.split("\\s+",entitiesList[2])
                if len(entity1List) == 1:
                    wordList[drugAPosition] = entity1List[0].strip().lower()
                else:
                    temWord = ""
                    for ele in entity1List:
                        temWord = temWord + ele.strip().lower()
                    wordList[drugAPosition] = temWord
                if len(entity2List) == 1:
                    wordList[drugBPosition] = entity2List[0].strip().lower()
                else:
                    temWord = ""
                    for ele in entity2List:
                        temWord = temWord + ele.strip().lower()
                    wordList[drugBPosition] = temWord
                positionA = ""
                positionB = ""
                for i, word in enumerate(wordList):
                    positionA = positionA + str(i - drugAPosition) + " "
                    positionB = positionB + str(i - drugBPosition) + " "
                    content = content + word + " "
                    wh.write(word + " ")
                wh.write("\n")
                wh.write(positionA + "\n")
                wh.write(positionB + "\n")
                wh.write(relation + "\n")
                tempList = []
print("Train data has been processed completed .")
print("Test data begins to process. ")
with open(testDataPath,"r") as rh:
    with open(testProcessedDataOutputPath,"w") as wh:
        tempList = []
        count = 0
        oneLine = rh.readline()
        tempList.append(oneLine.strip("\n"))
        count += 1
        breakSignal = False
        while breakSignal is False :
            if count % 1000 == 0:
                print(count)
            line = rh.readline()
            if line == "" :
                breakSignal = True
            line = line.strip("\n")
            tempList.append(line)
            count += 1
            if count % 4 == 0 :
                sentence = tempList[0]
                entities = tempList[1]
                relation = tempList[2]
                entitiesList = re.split("\t",entities)
                wordList = []
                for word in re.split("\\s+",sentence):
                    if word.__contains__(".") is False:
                        matchObj = re.match(patternOfWord, word)
                        if matchObj is not None:
                            matchStr = matchObj.group().lower()
                            if matchStr != "ddi":
                                if vocabMap.__contains__(matchStr) is False:
                                    vocabMap[matchStr] = 1
                                else:
                                    vocabMap[matchStr] = vocabMap[matchStr] + 1
                                wordList.append(matchStr)
                if wordList.__contains__("DRUGA".lower()) is False or wordList.__contains__("DRUGB".lower()) is False:
                    tempList = []
                    continue
                drugAPosition = wordList.index("DRUGA".lower())
                drugBPosition = wordList.index("DRUGB".lower())
                if relationMap.__contains__(relation.lower()) is False:
                    relationMap[relation.lower()] = 1
                else:
                    relationMap[relation.lower()] = relationMap[relation.lower()] + 1
                entity1List = re.split("\\s+",entitiesList[0])
                entity2List = re.split("\\s+",entitiesList[2])
                if len(entity1List) == 1:
                    wordList[drugAPosition] = entity1List[0].strip().lower()
                else:
                    temWord = ""
                    for ele in entity1List:
                        temWord = temWord + ele.strip().lower()
                    wordList[drugAPosition] = temWord
                if len(entity2List) == 1:
                    wordList[drugBPosition] = entity2List[0].strip().lower()
                else:
                    temWord = ""
                    for ele in entity2List:
                        temWord = temWord + ele.strip().lower()
                    wordList[drugBPosition] = temWord
                positionA = ""
                positionB = ""
                for i, word in enumerate(wordList):
                    positionA = positionA + str(i - drugAPosition) + " "
                    positionB = positionB + str(i - drugBPosition) + " "
                    content = content + word + " "
                    wh.write(word + " ")
                wh.write("\n")
                wh.write(positionA + "\n")
                wh.write(positionB + "\n")
                wh.write(relation + "\n")
                tempList = []
print("Test data has been processed .")
print("Begin to train word2vec.")
with open(temporaryFileFolder + "Content.txt","w") as wh:
    wh.write(content)
word2vec.word2vec(temporaryFileFolder + "Content.txt", temporaryFileFolder + "tem.bin",
                  size=embeddingDim,verbose=True,sample=1e-4,
                  negative=10,min_count=0,cbow=0,window=8)
print("Training is completed .")
model = word2vec.load(temporaryFileFolder + 'tem.bin')
for voca in model.vocab:
    if voca == "<\s>":
        print(voca)
with open(weightOutputPath,"w") as wh :
    for voca in model.vocab:
        if voca != "</s>":
            thisVector = model[voca]
            wh.write(voca + " ")
            for number in thisVector:
                wh.write(str(number) + " ")
            wh.write("\n")
with open(vocabularyStatisticInforPath,"w") as wh :
    for key, value in vocabMap.items():
        wh.write(key  + " " + str(value) + "\n")
with open(relationLabelStatisticInforPath,"w") as wh:
    for key, value in relationMap.items():
        wh.write(key  + " " + str(value) + "\n")





