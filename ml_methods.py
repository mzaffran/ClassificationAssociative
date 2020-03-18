import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import AdaBoostClassifier

data = pd.read_csv('data/kidney.csv')
train = pd.read_csv('data/kidney_train.csv')
test = pd.read_csv('data/kidney_test.csv')

y_train = train.pop('Class')
y_test = test.pop('Class')

svc = SVC(kernel='rbf')

svc.fit(train, y_train)

y_pred_test_svc = svc.predict(test)
y_pred_train_svc = svc.predict(train)

precision_test_svc, recall_test_svc, _, _ = precision_recall_fscore_support(y_test, y_pred_test_svc)
print(precision_test_svc)
print(recall_test_svc)

precision_train_svc, recall_train_svc, _, _ = precision_recall_fscore_support(y_train, y_pred_train_svc)
print(precision_train_svc)
print(recall_train_svc)

ada = AdaBoostClassifier()

ada.fit(train, y_train)

y_pred_test_ada = ada.predict(test)
y_pred_train_ada = ada.predict(train)

precision_test_ada, recall_test_ada, _, _ = precision_recall_fscore_support(y_test, y_pred_test_ada)
print(precision_test_ada)
print(recall_test_ada)

precision_train_ada, recall_train_ada, _, _ = precision_recall_fscore_support(y_train, y_pred_train_ada)
print(precision_train_ada)
print(recall_train_ada)
