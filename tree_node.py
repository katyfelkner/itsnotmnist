# this class represents a single node of a decision tree

class TreeNode():

    def __init__(self):
        # column index of the attribute on which this node splits
        # valid values for handwriting data set: 1-16
        self.split_on = 0

        # value at which to split
        # nonnegative value, so we use -1 for empty
        self.split_value = -1

        # left child node (example[split_on] <= split_value)
        self.left_child = None

        # right child node (example[split_on] > split_value)
        self.right_child = None