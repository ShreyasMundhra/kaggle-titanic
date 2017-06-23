import numpy as np
import os
from constants import outDir
import pandas as pd
from sklearn.svm import LinearSVC

def getProcessedData(filename):
    return pd.read_csv(os.path.join(outDir, filename + '.csv'))

def train():
    trainDf = getProcessedData('train')

    trainDf = trainDf[trainDf['Fare'].notnull()]
    trainDf['Fare'] = trainDf['Fare'].replace(to_replace=0, value=0.1)
    trainDf['Fare'] = np.log(trainDf['Fare'].astype('float64'))

    trainDf_X = trainDf.drop('Survived',axis=1)
    trainDf_X = trainDf_X.drop('PassengerId', axis=1)
    trainDf_y = trainDf['Survived']

    X = trainDf_X.values.tolist()
    y = trainDf_y.values.tolist()

    classifier = LinearSVC()
    classifier.fit(X,y)
    return classifier

def test(cf):
    testDf = getProcessedData('test')

    testDf['Fare'] = testDf['Fare'].replace(to_replace='nan', value=0.1)
    testDf['Fare'] = testDf['Fare'].replace(to_replace=0, value=0.1)

    testDf['Fare'] = np.log(testDf['Fare'].astype('float64'))

    testDf['Fare'] = pd.to_numeric(testDf['Fare'])

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

def dfToScratchpad(df, fileName):
    with open(os.path.join('Scratchpad',fileName + '.csv'),'wb') as file:
        df.to_csv(file, index=False)


# Main function to run
if __name__ == '__main__':
    cf = train()
    test(cf)
