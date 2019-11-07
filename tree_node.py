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

        # label - only relevant for leaves
        self.label = None
        self.depth = -1 # convenience

    def print_tree(self):
        spaces = "    " * self.depth
        if self.label is not None:
            # this is a leaf
            desc = "leaf at depth: " + str(self.depth) + ", label: " + self.label
            print(spaces + desc)
        else:
            # not a leaf
            desc = "split on attribute " + str(self.split_on) + " at value " + str(self.split_value)
            print(spaces + desc)
            self.left_child.print_tree()
            self.right_child.print_tree()
