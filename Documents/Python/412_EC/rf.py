import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

letter_conv = lambda raw: float(int(raw, 16))


filename = "training.csv"
train_data = np.genfromtxt(filename, skip_header = 1, delimiter = ',', converters = {2: letter_conv},
                            missing_values = '', filling_values = float(0.0), dtype = float).T
filename = "testing.csv"
test_x = np.genfromtxt(filename, skip_header = 1, delimiter = ',', converters = {2: letter_conv},
                            missing_values = '', filling_values = float(0.0), dtype = float)

train_x = train_data[0:-1].T
train_y = train_data[-1]


print(train_x.shape, train_y.shape)

rfc = RandomForestClassifier(random_state=69247)
rfc.fit(train_x, train_y)

results = rfc.predict(test_x)

tally = Counter(results)
print(tally)
