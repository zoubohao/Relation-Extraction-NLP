import re

### Config
trainProcessedPath = "D:\MyProgram\Data\semeval_task9_train_pair\\trainProcessedData.txt"
relationInforPath = "D:\MyProgram\Data\semeval_task9_train_pair\\relationInfor.txt"


### operation
labelList = []
with open(relationInforPath,"r") as rh :
    for line in rh:
        oneLine = line.strip()
        labelList.append(re.split("\s",oneLine)[0])
labelFileHandleList = []
for label in labelList:
    labelFileHandleList.append(open(".\\" + label + ".txt","w"))
with open(trainProcessedPath,"r") as rh:
    tempList = []
    for i,line in enumerate(rh):
        if i % 4 == 0 and i != 0:
            flag = tempList[3]
            for tempInfor in tempList:
                labelFileHandleList[labelList.index(flag)].write(tempInfor + "\n")
            tempList = []
        tempList.append(line.strip())
for handle in labelFileHandleList:
    handle.close()












