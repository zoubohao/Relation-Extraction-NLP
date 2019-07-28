import re

### Config
trainProcessedPath = "D:\MyProgram\Data\dataset\ddi\\all_data\\trainProcessedData.txt"
relationInforPath = "D:\MyProgram\Data\dataset\ddi\\all_data\\relationInfor.txt"
### It must be a folder path , not a file path.
relationTrainFolderPath = "D:\MyProgram\Data\dataset\ddi\\all_data\\"


### operation
if relationTrainFolderPath.endswith("\\") is False:
    relationTrainFolderPath = relationTrainFolderPath + "\\"
labelList = []
with open(relationInforPath,"r") as rh :
    for line in rh:
        oneLine = line.strip()
        labelList.append(re.split("\s",oneLine)[0])
labelFileHandleList = []
for label in labelList:
    labelFileHandleList.append(open(relationTrainFolderPath + label + ".txt","w"))
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












