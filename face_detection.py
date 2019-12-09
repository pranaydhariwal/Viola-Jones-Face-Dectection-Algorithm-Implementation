from viola_jones import ViolaJones
from cascade import CascadeClassifier
import numpy as np
import pickle
import time

def evaluate(clf, data):
    correct = 0
    all_negatives, all_positives = 0, 0
    true_negatives, false_negatives = 0, 0
    true_positives, false_positives = 0, 0
    classification_time = 0

    first = True
    for x, y in data:
        if y == 1:
            all_positives += 1
        else:
            all_negatives += 1

        start = time.time()
        prediction = clf.classify(x, first)
        first = False
        classification_time += time.time() - start
        if prediction == 1 and y == 0:
            false_positives += 1
        if prediction == 0 and y == 1:
            false_negatives += 1

        correct += 1 if prediction == y else 0

    print("False Positive Rate: %d/%d (%f)" % (false_positives, all_negatives, false_positives/all_negatives))
    print("False Negative Rate: %d/%d (%f)" % (false_negatives, all_positives, false_negatives/all_positives))
    print("Accuracy: %d/%d (%f)" % (correct, len(data), correct/len(data)))
    print("Average Classification Time: %f" % (classification_time / len(data)))

#Part-1:
#Trains adaboost detector for 1, 3, 5, 10 rounds.

with open("cmu_train.pkl", 'rb') as f:
    training = pickle.load(f)

with open("cmu_test.pkl", 'rb') as f:
    test = pickle.load(f)

for i in [1, 3, 5, 10]:
    viola_jones = ViolaJones(i)
    viola_jones.train_adaboost(training, 499, 2000)
    evaluate(viola_jones, training)
    viola_jones.save(str(i))
    classifer = ViolaJones.load(str(i))
    evaluate(classifer, test)

#train and test cascade classifer with upto 40 rounds

cascade = CascadeClassifier([5, 5, 10, 15])
cascade.train(training)
evaluate(cascade, training)
cascade.save("Cascade")
cascade_test = CascadeClassifier.load("Cascade")
evaluate(cascade_test, test)

# classifer = ViolaJones.load("10")
# evaluate(classifer, test)
