import csv
import numpy as np
import string

def sigmoid(x):
    return 1 / (1 + np.e**(-x))

def sigmoid_der(x):
    return (1-sigmoid(x))*sigmoid(x)

#Feedforward function
#Calculates y hat and loss
def forProp(input,actual,weights1,weights2,bias):

    #get the z values by dot product of inputs and weight1's
    #z = 1x5 array
    temp1 = np.dot(input,weights1)+bias
    z = sigmoid(temp1)

    #get predicted values by dot product of z values and weight2's
    #pred = 1x26 array
    temp2 = np.dot(z,weights2)+bias
    pred = sigmoid(temp2)

    return pred,z

def backProp(x,y,z,pred,weights1,weights2):

    #error = 1x26 array
    error = y - pred

    #z = np.reshape(z,(z.shape[0],-1))

    #TODO this piece of shit won't work no matter what I do
    #weights2 -=  np.dot(error,np.dot(sigmoid_der(z),z))
    weights2 -= np.dot(z,error*sigmoid_der(z))

    #TODO update weights1 somehow

    return weights1, weights2


#readData- reads in letter-recognition_data.csv
#Separates data into 15,000 training values and 5,000 testing values
def readData(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        data = list(csv_reader)
        data = np.array(data)

    #Break data into x and y
    #data_y = np.empty(20000,dtype=str)
    #data_x = np.empty([20000,16]).astype(int)
    data_y = np.zeros(shape=(20000,1),dtype=str)
    data_x = np.zeros(shape=(20000,16),dtype=int)

    count = 0

    for arr in data:
        data_y[count] = arr[0]
        data_x[count] = arr[1:]
        count = count + 1

    #convert from letters to numbers
    #y_toInt = np.empty(20000,dtype=int)
    y_toInt = np.zeros(shape=(20000,1))

    for i in range(20000):
        y_toInt[i] = string.ascii_lowercase.index(data_y[i][0].lower())

    data_y = y_toInt

    #represent y as a 1x26 value array, all 0's except for 1 in corresponding letter (i.e. 'B' = (0,1,0,...,0))
    #temp_y = np.empty([15000,26]).astype(int)
    temp_y = np.zeros(shape=(15000,26))
    i = 0
    for arr in temp_y:
        val = data_y[i][0]
        arr[val] = 1
        i += 1

    data_y = temp_y

    train_x = data_x[:15000]
    train_y = data_y[:15000]
    test_x = data_x[15000:20000]
    test_y = data_y[15000:20000]

    return train_x, train_y, test_x, test_y

def agent():
    train_x, train_y, test_x, test_y = readData('letter-recognition_data.csv')

    #I picked 5 because why not
    numNodes = 5

    #weights1 16x5 array, weights2 5x26 array
    weights1 = np.random.rand(16,numNodes)
    weights2 = np.random.rand(numNodes,26)

    #iterate through each training example
    for i in range(15000):
        pred,z = forProp(train_x[i],train_y[i],weights1,weights2,1)
        weights1, weights2 = backProp(train_x[i],train_y[i],z,pred,weights1,weights2)


if __name__ == "__main__":
    agent()