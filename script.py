from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# DISCLAIMER
# You can change the training/validation split by changing the random state (currently 100)
# Ideally, the graph will look the same no matter how you split up the training set and test set.
# This data set is fairly small, so there is slightly more variance than usual.

breast_cancer_data = load_breast_cancer()

# INSPECT THE DATA

# print(breast_cancer_data.data[0])
# print(breast_cancer_data.feature_names)

# print(breast_cancer_data.target)
# print(breast_cancer_data.target_names)

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data,
                                                                                      breast_cancer_data.target,
                                                                                      test_size=0.2,
                                                                                      random_state=100)

# create classifier with K=3 neighbors
classifier = KNeighborsClassifier(n_neighbors=3)
# train classifier
classifier.fit(training_data, training_labels)
# print accuracy on validation set
print(classifier.score(validation_data, validation_labels))

# find optimal accuracy on validation set with K between 1 and 100
best_K = max(
                [
                    [
                        # create classifier with K as i
                        KNeighborsClassifier(n_neighbors=i)
                        # fit with training data and training labels
                        .fit(training_data, training_labels)
                        # get accuracy on validation data
                        .score(validation_data, validation_labels)
                        # also store the K value
                        , i
                     ]
                    for i
                    in range(1, 101)
                ]
            # key into the score
            , key=lambda x: x[0]
             )
print('Best K is: {} with an accuracy of {}'.format(best_K[1], best_K[0]))

# visualizing K accuracy scores

k_list = range(1, 101)
accuracies = []
for i in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(training_data,training_labels)
    accuracies.append(classifier.score(validation_data, validation_labels))

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()
