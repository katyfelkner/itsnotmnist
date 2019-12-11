# plot of accuracy vs min_examples

import matplotlib.pyplot as mpl

accuracies = [0.1136, 0.11973333333333333, 0.1144, 0.1088, 0.10453333333333334, 0.10053333333333334, 0.092, 0.09546666666666667, 0.0888, 0.0848]
minimums = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

mpl.plot(minimums, accuracies, linestyle='solid', color='blue')
mpl.xlabel("Minimum examples per node")
mpl.ylabel("fraction of predictions correct")
mpl.title("Accuracy vs. Minimum Examples")
mpl.show()

accuracies_forest = [0.05226666666666667, 0.07653333333333333, 0.06346666666666667, 0.0784, 0.11013333333333333, 0.10186666666666666, 0.1072]
n_trees = [1, 2, 5, 10, 50, 100, 200]

mpl.plot(n_trees, accuracies_forest, linestyle='solid', color='red')
mpl.xlabel("Number of trees in random forest")
mpl.ylabel("fraction of predictions correct")
mpl.title("Accuracy vs. Number of Trees")
mpl.show()