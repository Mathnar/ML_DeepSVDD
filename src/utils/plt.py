import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
mpl.use('TkAgg')  # !IMPORTANT

with open('../../data/log/cifar10_test/planes/results_2023_02_24_21_42_43.json', 'r') as file:
    data = json.load(file)

test_scores = np.array(data['score_array'])
test_labels = np.array(data['label_array'])

fpr, tpr, thresholds = roc_curve(test_labels, test_scores)

roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # plot the random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
