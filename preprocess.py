import numpy as np
import pandas as pd
import os
from collections import Counter
from constants import dataDir, outDir

# Load raw datasets
def loadRawData():
    # load main files in dataframe
    fileNames = ['train', 'test', 'gender_submission']
    # Dictionary of dataframes loaded from the raw data files as provided in the competition data
    rawDfs = {}
    for file in fileNames:
        rawDfs[file] = pd.read_csv(os.path.join(dataDir, file + '.csv'))
    return rawDfs

# Save a dataset as csv file to disk
def dfToCSV(df, fileName):
    with open(os.path.join(outDir,fileName + '.csv'),'wb') as file:
        df.to_csv(file, index=False)

def preprocess(fileName):
    rawDfs = loadRawData()

    df = rawDfs[fileName]

    # df = df.drop('PassengerId', axis=1)
    df = df.drop('Name', axis=1)

    df = yesNotoBinary(df, 'Sex')

    df = df[df['Embarked'].notnull()]
    df = categoricalToNumeric(df,'Embarked',multiple=False)
    df = df.drop('Embarked', axis=1)

    #Change later
    df['Age'] = df['Age'].replace(to_replace='nan', value='0')
    df = df.drop('Ticket', axis=1)
    df = df.drop('Cabin', axis=1)
    df = df.drop('Embarked XXX_other', axis=1)
    df['Fare'] = df['Fare'].replace(to_replace='nan',value='0')
    #######################################

    rawDfs[fileName] = df
    dfToCSV(df, fileName)
    return rawDfs

# Convert binary to numerical e.g. Yes to 1 and No/NA to 0
def yesNotoBinary(df, feature):
    labels = {}

    for row in range(0,len(df)):
        label = str.lower(str(df.get_value(row,feature)))

        if label not in labels.keys():
            labels[label] = 1 - len(labels)

        df.set_value(row,feature,labels[label])

    pd.to_numeric(df[feature])
    return df

#Convert categorical features to numerical features
def categoricalToNumeric(df, col_name, multiple = False, min_seen_count = 0, extractFeatures=True, sourceDf=None):
    counter = None
    val_to_int = None
    int_to_val = None

    # Build a map from string feature values to unique integers.
    # Assumes 'other' does not occur as a value.
    val_to_int = {'XXX_other': 0}
    int_to_val = ['XXX_other']
    next_index = 1
    counter = Counter()

    if(extractFeatures):
        extractable = df
    else:
        extractable = sourceDf

    for val in extractable[col_name]:
        if multiple:
            # val is a list of categorical values.
            counter.update(val)
        else:
            # val is a single categorical value.
            counter[val] += 1

    for val, count in counter.iteritems():
        if count >= min_seen_count:
            val_to_int[val] = next_index
            int_to_val.append(val)
            next_index += 1

    feats = np.zeros((len(df), len(val_to_int)))
    for i, orig_val in enumerate(df[col_name]):
        if multiple:
            # orig_val is a list of categorical values.
            list_of_vals = orig_val
        else:
            # orig_val is a single categorical value.
            list_of_vals = [orig_val]
        for val in list_of_vals:
            if val in val_to_int:
                feats[i, val_to_int[val]] += 1
            else:
                feats[i, val_to_int['XXX_other']] += 1

    feat_names = ['{} {}'.format(col_name, val) for val in int_to_val]

    # df = df.drop(col_name, axis=1)
    return pd.concat([df, pd.DataFrame(feats, index=df.index, columns=feat_names)], axis=1)


# Main function to run
if __name__ == '__main__':
    preprocess('train')
    preprocess('test')