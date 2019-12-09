import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from feature_box import FeatureBox
from weak_classifier import WeakClassifier

class ViolaJones:
    def __init__(self, no_of_detectors = 5):
        self.no_of_detectors = no_of_detectors
        self.alphas = []
        self.clfs = []

    def train_adaboost(self, training, pos_num, neg_num):
        weights = np.zeros(len(training))
        training_data = []
        print("Computing integral images")
        for x in range(len(training)):
            training_data.append((integ_img(training[x][0]), training[x][1]))
            if training[x][1] == 1:
                weights[x] = 1.0 / (2 * pos_num)
            else:
                weights[x] = 1.0 / (2 * neg_num)

        print("Building features")
        print(training_data[0][0].shape)
        features = self.build_features(training_data[0][0].shape)
        print(len(features))
        print("Applying features to training examples")
        X, y = self.apply_features(features, training_data)

        for t in range(self.no_of_detectors):
            weights = weights / np.linalg.norm(weights)
            weak_classifiers = self.train_each_weak_classifier(X, y, features, weights)
            clf, error, accuracy = self.select_best(weak_classifiers, weights, training_data)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            clf.accuracy = (len(accuracy) - sum(accuracy))/len(accuracy)
            self.alphas.append(alpha)
            self.clfs.append(clf)
            print("Chose classifier: %s with accuracy: %f and alpha: %f and error: %f" % (str(clf), (len(accuracy) - sum(accuracy))/len(accuracy), alpha, error))

    def train_each_weak_classifier(self, X, y, features, weights):
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w

        classifiers = []
        total_features = X.shape[0]
        for index, feature in enumerate(X):
            if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
                print("Trained %d classifiers out of %d" % (len(classifiers), total_features))

            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])

            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1

                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w

            clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers

    def build_features(self, image_shape):
        height, width = image_shape
        features = []
        feature1 = 0
        feature2 = 0
        feature3 = 0
        feature4 = 0
        feature5 = 0
        for w in range(1, width + 1):
            for h in range(1, height + 1):
                i = 0
                while i + w < width + 1:
                    j = 0
                    while j + h < height + 1:
                        #2 rectangle features
                        immediate = FeatureBox(i, j, w, h)
                        right = FeatureBox(i+w, j, w, h)
                        if i + 2 * w < width + 1: #Horizontally Adjacent
                            feature1 = feature1 + 1
                            features.append(([right], [immediate]))

                        bottom = FeatureBox(i, j+h, w, h)
                        if j + 2 * h < height + 1: #Vertically Adjacent
                            feature2 = feature2 + 1
                            features.append(([immediate], [bottom]))

                        right_2 = FeatureBox(i+2*w-1, j, w, h)
                        #3 rectangle features
                        if i + 3 * w < width + 1: #Horizontally Adjacent
                            feature3 = feature3 + 1
                            features.append(([right], [right_2, immediate]))

                        bottom_2 = FeatureBox(i, j+2*h-1, w, h)
                        if j + 3 * h < height + 1: #Vertically Adjacent
                            feature4 = feature4 + 1
                            features.append(([bottom], [bottom_2, immediate]))

                        #4 rectangle features
                        bottom_right = FeatureBox(i+w-1, j+h-1, w, h)
                        if i + 2 * w < width + 1 and j + 2 * h < height + 1:
                            feature5 = feature5 + 1
                            features.append(([right, bottom], [immediate, bottom_right]))

                        j += 1
                    i += 1
        print("The total number of Haar Features is: ", len(features), ".")
        print("There are ", feature1, " type 1 (two vertical) features.")
        print("There are ", feature2, " type 2 (two horizontal) features.")
        print("There are ", feature3, " type 3 (three vertical) features.")
        print("There are ", feature4, " type 4 (three horizontal) features.")
        print("There are ", feature5, " type 5 (four) features.")
        return np.array(features)

    def select_best(self, classifiers, weights, training_data):
        best_clf, best_error, best_accuracy = None, float('inf'), None
        for clf in classifiers:
            error, accuracy = 0, []
            for data, w in zip(training_data, weights):
                correctness = abs(clf.classify(data[0]) - data[1])
                accuracy.append(correctness)
                error += w * correctness
            error = error / len(training_data)
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy
        return best_clf, best_error, best_accuracy

    def apply_features(self, features, training_data):
        X = np.zeros((len(features), len(training_data)))
        y = np.array(list(map(lambda data: data[1], training_data)))
        i = 0
        for positive_regions, negative_regions in features:
            feature = lambda ii: sum([pos.compute_feature(ii) for pos in positive_regions]) - sum([neg.compute_feature(ii) for neg in negative_regions])
            X[i] = list(map(lambda data: feature(data[0]), training_data))
            i += 1
        return X, y

    def classify(self, image, first):
        total = 0
        ii = integ_img(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            if first is True:
                fig,ax = plt.subplots(1)
                ax.imshow(image)
                for k in clf.positive_regions:
                    rect1 = patches.Rectangle((k.x,k.y),k.width,k.height,linewidth=1,facecolor='yellow')
                    ax.add_patch(rect1)
                for k in clf.negative_regions:
                    rect2 = patches.Rectangle((k.x,k.y),k.width,k.height,linewidth=1,facecolor='purple')
                    ax.add_patch(rect2)
                plt.show()
                print(str(clf))
            total += alpha * clf.classify(ii)
        return 1 if total >= 0.5 * sum(self.alphas) else 0

    def save(self, filename):
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)


def integ_img(image):
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii
