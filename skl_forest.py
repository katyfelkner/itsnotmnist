from sklearn.ensemble import RandomForestClassifier
import pandas

# load data set and partition it into training, validation, and testing
full_dataset = pandas.read_csv("./letter-recognition.csv", header=None)

# only partitioning into train-val and testing because we are not validating hyperparameters, we are just comparing this
# to my implementation with the same hyperparameters
n_testing = int(full_dataset.shape[0] * .25)
test_set = full_dataset.tail(n_testing)
train_val_set = full_dataset.head(full_dataset.shape[0] - n_testing)


# cut into x and y
# make x and y dfs for testing set
train_val_x = train_val_set.iloc[:, 1:17]
train_val_y = train_val_set.iloc[:, 0]

final_forest = RandomForestClassifier(n_estimators=50, criterion="gini",
                                      max_depth=10, max_features=4)

final_forest.fit(train_val_x, train_val_y)

# make x and y dfs for testing set
test_x = test_set.iloc[:, 1:17]
test_y = test_set.iloc[:, 0]

correct = 0
for i in range(test_set.shape[0]):
    label = final_forest.predict(test_x.iloc[i, :].values.reshape(1, -1))[0]
    if label == test_y.iloc[i]:
        correct += 1
    if (i % 500 == 0):
        print("testing prediction finished for row", i)

print(correct, "testing examples correct out of", n_testing)
print("testing success rate:", correct/n_testing)