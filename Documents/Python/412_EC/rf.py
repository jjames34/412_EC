import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

letter_conv = lambda raw: float(int(raw, 16))


filename = "training.csv"
train_data = np.genfromtxt(filename, skip_header = 1, delimiter = ',', converters = {2: letter_conv},
                            missing_values = '', filling_values = float(0.0), dtype = float).T
filename = "testing.csv"
test_data = np.genfromtxt(filename, skip_header = 1, delimiter = ',', converters = {2: letter_conv},
                            missing_values = '', filling_values = float(0.0), dtype = float).T


train_x = train_data[1:-1].T
train_y = train_data[-1]
test_x = test_data[1:].T
test_ids = test_data[0]



print(train_x.shape, train_y.shape)

rfc = RandomForestClassifier(random_state=69247)
rfc.fit(train_x, train_y)

results = rfc.predict(test_x)

f = open('results.csv', 'w')
f.write('Id,Response\n')
for i in range(len(results)):
	line = (int(test_ids[i]), ',', int(results[i]), '\n')
	for i in line:
		f.write(str(i))
f.close()

tally = Counter(results)
print(tally)
