import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import AdaBoostClassifier
import time

data = pd.read_csv('data/kidney.csv')
train = pd.read_csv('data/kidney_train.csv')
test = pd.read_csv('data/kidney_test.csv')

y_train = train.pop('Class')
y_test = test.pop('Class')

print("=== SVC")

svc = SVC(kernel='rbf')

t1 = time.time()
svc.fit(train, y_train)
t = time.time() - t1
print("Time :", t)

y_pred_test_svc = svc.predict(test)
y_pred_train_svc = svc.predict(train)

precision_train_svc, recall_train_svc, _, _ = precision_recall_fscore_support(y_train, y_pred_train_svc)
print("Train precision :", precision_train_svc)
print("Train recall :", recall_train_svc)

precision_test_svc, recall_test_svc, _, _ = precision_recall_fscore_support(y_test, y_pred_test_svc)
print("Test precision :", precision_test_svc)
print("Test recall :", recall_test_svc)

print("=== AdaBoost")

ada = AdaBoostClassifier()

t1 = time.time()
ada.fit(train, y_train)
t = time.time() - t1
print("Time :", t)

y_pred_test_ada = ada.predict(test)
y_pred_train_ada = ada.predict(train)

precision_train_ada, recall_train_ada, _, _ = precision_recall_fscore_support(y_train, y_pred_train_ada)
print("Train precision :", precision_train_ada)
print("Train recall :", recall_train_ada)

precision_test_ada, recall_test_ada, _, _ = precision_recall_fscore_support(y_test, y_pred_test_ada)
print("Test precision :", precision_test_ada)
print("Test recall :", recall_test_ada)
