
import numpy as np

filterFilePath = "D:\MyProgram\Data\dataset\ddi\\all_data\\trainProcessedData.txt"
filterOutputPath = "D:\MyProgram\Data\dataset\ddi\\all_data\\trainFilterProcessedData.txt"

with open(filterFilePath,"r") as rh :
    with open(filterOutputPath,"w") as wh:
        now = []
        for i , line in enumerate(rh):
            if i % 4 == 0 and i != 0 :
                if now[-1].lower() != "false":
                    for ele in now:
                        wh.write(ele + "\n")
                else:
                    random = np.random.rand()
                    if random > 0.6:
                        for ele in now:
                            wh.write(ele + "\n")
                now = []
            oneLine = line.strip()
            if i % 4 == 0:
                now.append(oneLine)
            elif i % 4 == 1:
                now.append(oneLine)
            elif i % 4 == 2:
                now.append(oneLine)
            else:
                now.append(oneLine)





