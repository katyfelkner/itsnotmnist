# learn a tree. this first script will learn a single unweighted binary decision tree

# Imports
import pandas
import numpy
import math
import tree_node
import string

# constant: list of capital letter chars
LETTERS = list(string.ascii_uppercase)
MAX_DEPTH = 20
MIN_EXAMPLES = 10
# NOTE: some of the code for CSV processing and handling datasets is taken from my answers to HW5.
# Actual ML code was written for this project.

# function to traverse the tree!
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


# functions to do actual tree learning

# grow decision tree, based on algorithm given in class
# passing in a depth value for convenience
def grow_decision_tree(examples, attributes, default, depth):
    # for this case, we have numeric (real valued) features and a categorical result
    print ("starting to process a node at depth", depth)
    # check stopping conditions - if that's the case, return a leaf node

    # no examples left or less than minimum
    if len(examples) <= MIN_EXAMPLES:
        leaf = tree_node.TreeNode()
        leaf.depth = depth
        leaf.label = default
        print("returning a leaf at depth", depth)
        return leaf

    # all examples have same label
    if examples.iloc[:, 0].nunique() == 1:
        # return the label of the first one, since they're all the same
        leaf = tree_node.TreeNode()
        leaf.depth = depth
        leaf.label = examples.iloc[0, 0]
        print("returning a leaf at depth", depth)
        return leaf

    # no attributes (not relevant now, may be implemented later)

    # maximum depth
    if depth >= MAX_DEPTH:
        leaf = tree_node.TreeNode()
        leaf.depth = depth
        leaf.label = default
        print("returning a leaf at depth", depth)
        return leaf

    # else, grow recursively
    best = choose_best_attr(attributes, examples)

    # check for None - if we have it, no acceptable split was found, and we should make this a leaf
    if best[0] is None or best[1] is None:
        leaf = tree_node.TreeNode()
        leaf.depth = depth
        leaf.label = default
        print("returning a leaf at depth", depth)
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
    left_label = left_examples.iloc[:,0].mode()[0]
    right_label = right_examples.iloc[:,0].mode()[0]

    # passing entire set of attributes because they are real-valued
    tree.left_child = grow_decision_tree(left_examples, attributes, left_label, depth + 1)
    tree.right_child = grow_decision_tree(right_examples, attributes, right_label, depth + 1)

    print("finishing a node at depth", depth)
    return tree

# choose best attribute and split value
def choose_best_attr(attrs, ex):

    max_gain = None
    max_index = None
    max_split = None
    #for a in attrs:
    # for now, the attributes are always the list of all attributes because we aren't removing them yet
    for i in range(1, 17):
        # arbitrarily, I am choosing to use Info gain right now. Will test against Gini later.
        # test every observed value of this attribute
        for value in ex.iloc[:, i].unique().tolist():
            this_gain = gini(ex, i, value)
            if this_gain is None:
                continue
            if max_gain is None or this_gain > max_gain:
                max_gain = this_gain
                max_index = i
                max_split = value

    # return index and split value as tuple
    return (max_index, max_split)


# info gain function
def info_gain_remainder(df, column_index, split_value):
    total_num = len(df)

    # get proportions of entire data set
    prop = []
    for l in LETTERS:
        l_count = len(df[df.iloc[:, 0] == l])
        prop.append(l_count / total_num)

    i_total = info_gain(prop)

    # get info gain for each side of the binary split
    left_df = df[df.iloc[:, column_index] <= split_value]
    right_df = df[df.iloc[:, column_index] > split_value]
    left_num = len(left_df)
    right_num = len(right_df)
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

    # calculate info gains
    i_left = info_gain(prop_l)
    i_right = info_gain(prop_r)

    gain = i_total - (i_left + i_right)

    return gain

def info_gain(proportions):
    # pass in a list of the proportion of each letter in the same
    ret = 0
    for i in range(0, len(proportions)):
        if proportions[i] > 0:
            ret += -1 * proportions[i] * math.log(proportions[i], 2)
    return ret

# function for GINI (testing this to see if it's better than information gain)
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
# full_dataset = pandas.read_csv("./letter-recognition.csv", header=None, nrows=500)

n_testing = int(full_dataset.shape[0] * .25)
test_set = full_dataset.tail(n_testing)
train_val_set = full_dataset.head(full_dataset.shape[0] - n_testing)
n_val = int(train_val_set.shape[0] * .25)
val_set = train_val_set.tail(n_val)
train_set = train_val_set.head(train_val_set.shape[0] - n_val)

# train for different values of min_examples
# passing None for attrs because we may end up wanting that param later
max_success = 0
best_min_examples = 0

for i in range(10, 110, 10):
    print("now testing MIN_EXAMPLES value of:", i)
    MIN_EXAMPLES = i
    tree = grow_decision_tree(train_set, None, None, 0)

    #tree.print_tree()

    # make some predictions on the validation set
    correct = 0
    for index, row in val_set.iterrows():
        predicted = predict_with_tree(tree, row)
        # print("Actual value: " + row.iloc[0])
        # print("Predicted value: " + predicted)
        if predicted == row.iloc[0]:
            correct += 1

    print("min_examples currently set to:", MIN_EXAMPLES)
    print(correct, "training examples correct out of", len(val_set))
    print("training success rate:", correct / len(val_set))

    if (correct / len(val_set)) > max_success:
        best_min_examples = MIN_EXAMPLES
        max_success = (correct / len(val_set))


# train best example on the train-val set
MIN_EXAMPLES = best_min_examples
print("min_examples best value is:", MIN_EXAMPLES)
tuned_tree = grow_decision_tree(train_val_set, None, None, 0)

tree.print_tree()

# make some predictions on the train-val set
correct = 0
for index, row in train_val_set.iterrows():
    predicted = predict_with_tree(tuned_tree, row)
    # print("Actual value: " + row.iloc[0])
    # print("Predicted value: " + predicted)
    if predicted == row.iloc[0]:
        correct += 1

print(correct, "training examples correct out of", len(train_val_set))
print("training success rate:", correct / len(train_val_set))


# make some predictions on the testing set
correct = 0
for index, row in test_set.iterrows():
    predicted = predict_with_tree(tuned_tree, row)
    #print("Actual value: " + row.iloc[0])
    #print("Predicted value: " + predicted)
    if predicted == row.iloc[0]:
        correct += 1

print(correct, "testing examples correct out of", n_testing)
print("testing success rate:", correct/n_testing)

