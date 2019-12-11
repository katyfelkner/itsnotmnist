import csv
import numpy as np
import string
import matplotlib.pyplot as plt
from sklearn import preprocessing
from textwrap import wrap

#Feedforward function
#Calculates y hat
def forProp(input,actual,weights1,weights2,weights3,bias):

    z1 = np.dot(input, weights1)
    z1 = 1 / (1 + np.exp(-z1))

    z2 = np.dot(z1, weights2)
    z2 = 1 / (1 + np.exp(-z2))

    pred = np.dot(z2, weights3)
    pred = 1 / (1 + np.exp(-pred))

    return pred,z1,z2

#Backward propagation function
#Calculate error and update weights
def backProp(x,y,z,pred,weights1,weights2,weights3):

    error = pred - y
    sig_der_pred = (1 - (1 / (1 + np.exp(-pred)))) * (1 / (1 + np.exp(-pred)))
    sig_der_z1 = (1 - (1 / (1 + np.exp(-z1)))) * (1 / (1 + np.exp(-z1)))
    sig_der_z2 = (1 - (1 / (1 + np.exp(-z2)))) * (1 / (1 + np.exp(-z2)))

    w3 = np.dot(z2.T, 2 * error * sig_der_pred)
    w2 = np.dot(z1.T, np.dot(2 * error * sig_der_pred, weights3.T) * sig_der_z2)
    w1 = np.dot(X.T, np.dot(np.dot(2 * error * sig_der_pred, weights3.T) * sig_der_z2, weights2.T)
                * sig_der_z1)

    #Update weights
    weights1 += w1
    weights2 += w2
    weights3 += w3

    #Normalize when weights > 1 to keep weights lower (usually end up at 10^9)
    if np.amax(weights1) > 1 or np.amax(weights2) > 1 or np.amax(weights3) > 1:
        weights1 = preprocessing.normalize(weights1)
        weights2 = preprocessing.normalize(weights2)
        weights3 = preprocessing.normalize(weights3)

    return weights1, weights2, weights3

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

    #I picked 15 because why not
    numNodes = 15

    weights1 = np.random.rand(16, numNodes)
    weights2 = np.random.rand(numNodes, numNodes)
    weights3 = np.random.rand(numNodes, 26)

    #iterate through each training example
    for i in range(15000):
        pred,z1,z2 = forProp(train_x[i],train_y[i],weights1,weights2,weights3,1)
        weights1, weights2, weights3 = backProp(train_x[i],train_y[i],z,pred,weights1,weights2,weights3)

    i = 0
    correct = 0
    avgCorrect = np.empty(5000)

    for i in range(5000):
        pred= forProp(test_x[i],test_y[i],weights1,weights2,weights3,1)
        result1 = np.where(pred == np.amax(pred))
        result2 = np.where(test_y[i] == np.amax(test_y[i]))
        if result1 == result2:
            correct += 1

        avgCorrect[i] = (correct / (i + 1))*100

    plt.plot(avgCorrect)
    plt.xlabel('Test Samples')
    plt.ylabel('Percentage Correct')
    title = 'Percentage Letters Predicted Correctly Over Test Samples - 2 Hidden Layers & Sigmoid Function'
    plt.title('\n'.join(wrap(title, 60)), fontsize=10)
    plt.show()


if __name__ == "__main__":
    agent()