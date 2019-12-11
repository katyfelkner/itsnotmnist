# random forest - basically a bunch of trees
# Imports
import pandas
import numpy
import math
import tree_node
import string
import random
from scipy import stats

# constant: list of capital letter chars
LETTERS = list(string.ascii_uppercase)
# constant: possible attribute indices. we sample from this list at each growth step
ATTR_INDICES = [i for i in range(1, 17)]
MAX_DEPTH = 10 # 20 was best value, but it's not realistic to run 50+ trees at depth = 20
MIN_EXAMPLES = 20 # chosen by validation on single tree
# NOTE: some of the code for CSV processing and handling datasets is taken from my answers to HW5.
# Actual ML code was written for this project.

# function to traverse a single tree!
# this function takes in a row from the CSV (i.e. including the label) but ignores the label
# also pass in the root node
def predict_with_tree(tree, data):
    # if there's a label, we're at a leaf, so return that
    if tree.label is not None:
        return tree.label

    # else, we need to recurse down to a child
    if data.iloc[tree.split_on] <= tree.split_value:
        # recurse to left child
        return predict_with_tree(tree.left_child, data)
    else:
        # recurse to right child
        return predict_with_tree(tree.right_child, data)


# use the forest to make a prediction
# this function takes in a row from the CSV (i.e. including the label) but ignores the label
# also pass in the forest itself
def predict_with_forest(forest, data):
    outputs = []
    for tree in forest:
        outputs.append(predict_with_tree(tree, data))

    # now find the mode of the outputs and return it. If multiple modes, randomly select one.
    modes = stats.mode(outputs)
    return modes[0]


# bagging - choose n examples with replacement
def get_bagged_data(df):
    # random state = none will give us a different sample each time
    new_data = df.sample(n = len(df), replace = True, random_state = None)
    return new_data


# grow_rf_tree
def grow_rf_tree(examples, attributes, default, depth, l):
    # for this case, we have numeric (real valued) features and a categorical result
    #print("starting to process a node at depth", depth)
    # check stopping conditions - if that's the case, return a leaf node

    # no examples left or less than minimum
    if len(examples) <= MIN_EXAMPLES:
        leaf = tree_node.TreeNode()
        leaf.depth = depth
        leaf.label = default
        #print("returning a leaf at depth", depth)
        return leaf

    # all examples have same label
    if examples.iloc[:, 0].nunique() == 1:
        # return the label of the first one, since they're all the same
        leaf = tree_node.TreeNode()
        leaf.depth = depth
        leaf.label = examples.iloc[0, 0]
        #print("returning a leaf at depth", depth)
        return leaf

    # maximum depth
    if depth >= MAX_DEPTH:
        leaf = tree_node.TreeNode()
        leaf.depth = depth
        leaf.label = default
        #print("returning a leaf at depth", depth)
        return leaf

    # else, grow recursively

    # sample some attributes
    sampled_attrs = numpy.random.choice(ATTR_INDICES, l, replace=False)
    best = choose_best_attr_sampled(sampled_attrs, examples)

    # check for None - if we have it, no acceptable split was found, and we should make this a leaf
    if best[0] is None or best[1] is None:
        leaf = tree_node.TreeNode()
        leaf.depth = depth
        leaf.label = default
        #print("returning a leaf at depth", depth)
        return leaf

    # else, there is an OK split
    tree = tree_node.TreeNode()
    tree.depth = depth
    tree.split_on = best[0]
    tree.split_value = best[1]

    # split examples
    left_examples = examples[examples.iloc[:, tree.split_on] <= tree.split_value]
    right_examples = examples[examples.iloc[:, tree.split_on] > tree.split_value]

    # if multiple modes, arbitrarily choose the first one
    left_label = left_examples.iloc[:, 0].mode()[0]
    right_label = right_examples.iloc[:, 0].mode()[0]

    # passing entire set of attributes because they are real-valued
    tree.left_child = grow_rf_tree(left_examples, attributes, left_label, depth + 1, l)
    tree.right_child = grow_rf_tree(right_examples, attributes, right_label, depth + 1, l)

    #print("finishing a node at depth", depth)
    return tree

# make rf

def train_forest(df, n):
    # train a forest with n trees, MAX_DEPTH and MIN_EXAMPLES taken from global constants
    forest = []
    l_value = 4 # log2 of 16, there are 16 attributes
    for i in range(0, n):
        print("Now training tree", i)
        df_i = get_bagged_data(df)
        tree = grow_rf_tree(df_i, None, None, 0, l_value)
        forest.append(tree)
    return forest

# auxiliary functions
# choose best attribute and split value
def choose_best_attr_sampled(indices, ex):
    max_gain = None
    max_index = None
    max_split = None
    for i in indices:
        # arbitrarily, I am choosing to use Info gain right now. Will test against Gini later.
        # test every observed value of this attribute
        for value in ex.iloc[:, i].unique():
            this_gain = gini(ex, i, value)
            if this_gain is None:
                continue
            if max_gain is None or this_gain > max_gain:
                max_gain = this_gain
                max_index = i
                max_split = value

    # return index and split value as tuple
    return (max_index, max_split)

# functions for GINI (testing this to see if it's better than information gain)
def gini(df, column_index, split_value):
    total_num = len(df)

    # find number of examples on each side
    left_df = df[df.iloc[:, column_index] <= split_value]
    right_df = df[df.iloc[:, column_index] > split_value]
    left_num = len(left_df)
    right_num = len(right_df)
    left_weight = left_num / total_num
    right_weight = right_num / total_num

    # if one side is too small, throw it out - else, we get a really tall, skinny, useless tree
    if left_num < MIN_EXAMPLES or right_num < MIN_EXAMPLES:
        return None

    # calculate proportions on the left
    prop_l = []
    for l_left in LETTERS:
        l_count_left = len(left_df[left_df.iloc[:, 0] == l_left])
        prop_l.append(l_count_left / left_num)

    # calculate proportions on the right
    prop_r = []
    for l_right in LETTERS:
        l_count_right = len(right_df[right_df.iloc[:, 0] == l_right])
        prop_r.append(l_count_right / right_num)

    # calculate gini for both sides
    left_gini = 1 - sum([math.pow(p, 2) for p in prop_l])
    right_gini = 1 - sum([math.pow(p, 2) for p in prop_r])

    # return weighted sum of gini scores
    return (left_weight * left_gini + right_weight * right_gini)


# load data set and partition it into training, validation, and testing
full_dataset = pandas.read_csv("./letter-recognition.csv", header=None)
# use a smaller dataset for early debugging
#full_dataset = pandas.read_csv("./letter-recognition.csv", header=None, nrows=5000)

n_testing = int(full_dataset.shape[0] * .25)
test_set = full_dataset.tail(n_testing)
train_val_set = full_dataset.head(full_dataset.shape[0] - n_testing)
n_val = int(train_val_set.shape[0] * .25)
val_set = train_val_set.tail(n_val)
train_set = train_val_set.head(train_val_set.shape[0] - n_val)

'''
# train debugging forest of 5 trees
forest = train_forest(train_val_set, 5)
# make some predictions on the training set
correct = 0
for index, row in val_set.iterrows():
    predicted = predict_with_forest(forest, row)
    # print("Actual value: " + row.iloc[0])
    # print("Predicted value: " + predicted)
    if predicted == row.iloc[0]:
        correct += 1

print(correct, "training examples correct out of", len(train_val_set))
print("training success rate:", correct / len(train_val_set))

# make some predictions on the testing set
correct = 0
for index, row in test_set.iterrows():
    predicted = predict_with_forest(forest, row)
    #print("Actual value: " + row.iloc[0])
    #print("Predicted value: " + predicted)
    if predicted == row.iloc[0]:
        correct += 1

print(correct, "testing examples correct out of", n_testing)
print("testing success rate:", correct/n_testing)
'''

# train for different values of num_trees
max_success = 0
best_num_trees = 50 # from validation run

'''for i in [1, 2, 5, 10, 50, 100, 200, 300, 400]:
    print("now testing NUM_TREES value of:", i)
    forest = train_forest(train_set, i)

    # make some predictions on the validation set
    correct = 0
    for index, row in val_set.iterrows():
        predicted = predict_with_forest(forest, row)
        # print("Actual value: " + row.iloc[0])
        # print("Predicted value: " + predicted)
        if predicted == row.iloc[0]:
            correct += 1

    print("num_trees currently set to:", i)
    print(correct, "training examples correct out of", len(val_set))
    print("training success rate:", correct / len(val_set))

    if (correct / len(val_set)) > max_success:
        best_num_trees = i
        max_success = (correct / len(val_set))'''


# train best example on the train-val set
print("num_trees best value is:", best_num_trees)
tuned_forest = train_forest(train_val_set, best_num_trees)

# make some predictions on the train-val set
correct = 0
for index, row in train_val_set.iterrows():
    predicted = predict_with_forest(tuned_forest, row)
    # print("Actual value: " + row.iloc[0])
    # print("Predicted value: " + predicted)
    if predicted == row.iloc[0]:
        correct += 1

print(correct, "training examples correct out of", len(train_val_set))
print("training success rate:", correct / len(train_val_set))


# make some predictions on the testing set
correct = 0
for index, row in test_set.iterrows():
    predicted = predict_with_forest(tuned_forest, row)
    #print("Actual value: " + row.iloc[0])
    #print("Predicted value: " + predicted)
    if predicted == row.iloc[0]:
        correct += 1

print(correct, "testing examples correct out of", n_testing)
print("testing success rate:", correct/n_testing)
