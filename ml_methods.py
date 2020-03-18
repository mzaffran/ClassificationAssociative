import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support


data = pd.read_csv('data/kidney.csv')
train = pd.read_csv('data/kidney_train.csv')
test = pd.read_csv('data/kidney_test.csv')

y_train = train.pop('Class')
y_test = test.pop('Class')

svc = SVC(kernel='rbf')

svc.fit(train, y_train)

y_pred = svc.predict(test)
y_pred_train = svc.predict(train)

precision_test, recall_test, _, _ = precision_recall_fscore_support(y_test, y_pred)
print(precision_test)
print(recall_test)

precision_train, recall_train, _, _ = precision_recall_fscore_support(y_train, y_pred_train)
print(precision_train)
print(recall_train)
