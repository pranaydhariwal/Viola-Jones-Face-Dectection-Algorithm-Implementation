class WeakClassifier:
    def __init__(self, positive_regions, negative_regions, threshold, polarity, accuracy=None):
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity
        self.accuracy = accuracy

    def classify(self, x):
        feature = lambda ii: sum([pos.compute_feature(ii) for pos in self.positive_regions]) - sum([neg.compute_feature(ii) for neg in self.negative_regions])
        return 1 if self.polarity * feature(x) < self.polarity * self.threshold else 0

    def __str__(self):
        return "Weak Classifier (threshold=%d, polarity=%d, %s, %s" % (self.threshold, self.polarity, str(self.positive_regions), str(self.negative_regions))
