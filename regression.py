import numpy as np
import pandas as pd
import os
from constants import dataDir
from sklearn.linear_model import LinearRegression

def linRegAge(trainDf=None):
    # trainDf = pd.read_csv(os.path.join(dataDir, 'train' + '.csv'))
    # cleanTrainDf = trainDf[trainDf['Age'].notnull()]

    derivedDf = pd.DataFrame()
    derivedDf['SibSp'] = trainDf['SibSp']
    derivedDf['Parch'] = trainDf['Parch']
    derivedDf['Fare'] = trainDf['Fare']
    derivedDf['Age'] = trainDf['Age']

    cleanDerivedDf = derivedDf[derivedDf['Age'].notnull()]
    missingDerivedDf = derivedDf[derivedDf['Age'].isnull()]

    ageList = cleanDerivedDf['Age']

    cleanDerivedDf = cleanDerivedDf.drop('Age',axis=1)
    missingDerivedDf = missingDerivedDf.drop('Age', axis=1)

    cleanX = cleanDerivedDf.values.tolist()
    missingX = missingDerivedDf.values.tolist()

    # ageList = trainDf['Age']
    # model = LinearRegression().fit(cleanX,np.array(np.array([age]) for age in ageList))
    model = LinearRegression().fit(cleanX, ageList)
    missingDerivedDf['Age'] = model.predict(missingX)


    trainDf['Age'] = trainDf['Age'].fillna(value=missingDerivedDf['Age'])


    # with open(os.path.join('Scratchpad','age_regression' + '.csv'),'wb') as file:
    #     missingDerivedDf.to_csv(file, index=False)
    #     trainDf.to_csv(file, index=False)


    return trainDf


# Main function to run
# if __name__ == '__main__':
#     linRegAge()