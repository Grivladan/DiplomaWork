import numpy as np
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=3, linewidth = 120)

F = open("ALLDATA.DAT", "r" )
testData = F.readlines()

X = np.zeros((36,16))
y = np.zeros(36)
healthX = np.zeros((10,16))

for i in range(4,40):
    tmpArr = testData[i].split()
    y[i-4]=float(tmpArr[1])
    for j in range (2,18):
        X[i-4][j-2] = float(tmpArr[j])

corrX = np.corrcoef(X.T)
print(corrX)

#for i in range(44, 54):
#    print(testData[i])
