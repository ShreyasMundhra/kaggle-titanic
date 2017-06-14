import os
from constants import outDir
import pandas as pd
from sklearn.svm import LinearSVC

def getProcessedData(filename):
    return pd.read_csv(os.path.join(outDir, filename + '.csv'))

def train():
    trainDf = getProcessedData('train')
    trainDf_X = trainDf.drop('Survived',axis=1)
    trainDf_X = trainDf_X.drop('PassengerId', axis=1)
    trainDf_y = trainDf['Survived']

    # print type(trainDf_X.values.tolist())
    # print type(trainDf_X.values.tolist()[0])
    # print type(trainDf_X.values.tolist()[0][0])

    X = trainDf_X.values.tolist()
    y = trainDf_y.values.tolist()

    classifier = LinearSVC()
    classifier.fit(X,y)
    return classifier

def test(cf):
    testDf = getProcessedData('test')
    testDf_X = testDf.drop('PassengerId', axis=1)

    X = testDf_X.values.tolist()

    y = cf.predict(X)

    predsDf = pd.DataFrame()
    predsDf['PassengerId'] = testDf['PassengerId']
    predsDf['Survived'] = y

    dfToCSV(predsDf, 'preds')

# Save a dataset as csv file to disk
def dfToCSV(df, fileName):
    with open(os.path.join(outDir,fileName + '.csv'),'wb') as file:
        df.to_csv(file, index=False)



# Main function to run
if __name__ == '__main__':
    cf = train()
    test(cf)
