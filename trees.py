# learn a tree. this first script will learn a single unweighted binary decision tree

# Imports
import pandas
import numpy
import tree_node

# NOTE: some of the code for CSV processing and handling datasets is taken from my answers to HW5.
# Actual ML code was written for this project.

# load data set and partition it into training, validation, and testing
full_dataset = pandas.read_csv("./letter-recognition.csv", header=None)

n_testing = int(full_dataset.shape[0] * .25)
test_set = full_dataset.tail(n_testing)
train_val_set = full_dataset.head(full_dataset.shape[0] - n_testing)
n_val = int(train_val_set.shape[0] * .25)
val_set = train_val_set.tail(n_val)
train_set = train_val_set.head(train_val_set.shape[0] - n_val)


# grow decision tree, based on algorithm given in class
# passing in a depth value for convenience
def grow_decision_tree(examples, attributes, default, depth):
    # for this case, we have numeric (real valued) features and a categorical result

    # TODO: check stopping conditions

    # else, grow recursively
    best = choose_best_attr(attributes, examples)
    tree = tree_node.TreeNode()
    tree.split_on = best[0]
    tree.split_value = best[1]

    # split examples
    left_examples = examples[examples.iloc[:, tree.split_on] <= tree.split_value]
    right_examples = examples[examples.iloc[:, tree.split_on] > tree.split_value]

    # if multiple modes, arbitrarily choosing the first one
    # this case shouldn't happen too much - it means learning isn't going well
    left_label = left_examples.iloc[:,0].mode()[0]
    right_label = right_examples.iloc[:,0].mode()[0]

    # passing entire set of attributes because they are real-valued
    tree.left_child = grow_decision_tree(left_examples, attributes, left_label, depth + 1)
    tree.right_child = grow_decision_tree(right_examples, attributes, right_label, depth + 1)

    return tree

# choose best attribute and split value
def choose_best_attr(attrs, ex):
    # TODO: write this method
    pass