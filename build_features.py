import numpy as np

def integral_image(image):
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii

def apply_features(self, features, training_data):
    X = np.zeros((len(features), len(training_data)))
    y = np.array(list(map(lambda data: data[1], training_data)))
    i = 0
    for positive_regions, negative_regions in features:
        feature = lambda ii: sum([pos.compute_feature(ii) for pos in positive_regions]) - sum([neg.compute_feature(ii) for neg in negative_regions])
        X[i] = list(map(lambda data: feature(data[0]), training_data))
        i += 1
    return X, y


with open("cmu_training.pkl", 'rb') as f:
    training = pickle.load(f)

weights = np.zeros(len(training))
training_data = []
print("Computing integral images")
for x in range(len(training)):
    training_data.append((integral_image(training[x][0]), training[x][1]))
    if training[x][1] == 1:
        weights[x] = 1.0 / (2 * pos_num)
    else:
        weights[x] = 1.0 / (2 * neg_num)

print("Building features")
features = build_features(training_data[0][0].shape)
print("Applying features to training examples")
X, y = apply_features(features, training_data)
print("Selecting best features")
indices = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True)
X = X[indices]
features = features[indices]
print("Selected %d potential features" % len(X))
