"""
Neural Net Comparison using Sklearn
Multi-layer Perceptron Classifier
Function = sigmoid
Predicts testing data correctly 73% of the time
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
import csv
import string

#readData- reads in letter-recognition_data.csv
#Separates data into 15,000 training values and 5,000 testing values
def readData(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        data = list(csv_reader)
        data = np.array(data)

    #Break data into x and y
    data_y = np.empty(20000,dtype=str)
    data_x = np.empty([20000,16]).astype(float)
    count = 0

    for arr in data:
        data_y[count] = arr[0]
        data_x[count] = arr[1:]
        count = count + 1

    #convert from letters to numbers
    y_toInt = np.empty(20000,dtype=int)
    for i in range(20000):
        y_toInt[i] = string.ascii_lowercase.index(data_y[i].lower())

    data_y = y_toInt

    #represent y as a 1x26 value array, all 0's except for 1 in corresponding letter (i.e. 'B' = (0,1,0,...,0))
    temp_y = np.empty([20000,26]).astype(int)
    i = 0
    for arr in temp_y:
        val = data_y[i]
        arr[val] = 1
        i += 1

    data_y = temp_y

    count = 0
    for arr in data_x:
        test =  (arr - np.min(arr)) / np.ptp(arr)
        data_x[count] = test
        count += 1

    train_x = data_x[:15000]
    train_y = data_y[:15000]
    test_x = data_x[15000:20000]
    test_y = data_y[15000:20000]

    return train_x, train_y, test_x, test_y

def agent():
    train_x, train_y, test_x, test_y = readData('letter-recognition_data.csv')

    mlp = MLPClassifier(activation='logistic')
    mlp.fit(train_x, train_y)

    pred = mlp.predict_proba(test_x)
    correct = 0
    for i in range(5000):
        prediction = pred[i]
        result1 = np.where(prediction == np.amax(prediction))
        result2 = np.where(test_y[i] == np.amax(test_y[i]))
        if result1 == result2:
            correct += 1

    perc_correct = (correct/5000)*100
    print(perc_correct)
    #73.39% correct

if __name__ == "__main__":
    agent()