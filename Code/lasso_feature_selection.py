import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

data = pd.read_csv('features_filled_names.csv', nrows=278849)

numerics = ['int16','int32','int64','float16','float32','float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]

#print(data.shape)

x = pd.DataFrame(data.drop(labels=['carona_result'], axis=1))
y = pd.DataFrame(data['carona_result'])

Min_Max = MinMaxScaler()
X = Min_Max.fit_transform(x)
Y= Min_Max.fit_transform(y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#print(X_train.shape)
#print(X_test.shape)

sel_L = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
sel_L.fit(X_train, np.ravel(Y_train,order='C'))
sel_L.get_support()
X_train = pd.DataFrame(X_train)

selected_feat = X_train.columns[(sel_L.get_support())]
print('total features: {}'.format((X_train.shape[1])))
print('selected features using lasso regression: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
np.sum(sel_L.estimator_.coef_ == 0)))

print()
print()
print()
print()

removed_feats = X_train.columns[(sel_L.estimator_.coef_ == 0).ravel().tolist()]
removed_feats
X_train_selected = sel_L.transform(X_train)
X_test_selected = sel_L.transform(X_test)
X_train_selected.shape, X_test_selected.shape

print(X_train_selected[:20])

# clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
# clf.fit(X_train_selected,np.ravel(Y_train,order='C'))
# y_pred = clf.predict(X_test_selected)
# print('Accuracy after feature selection using lasso regression: {}'.format(accuracy_score(Y_test, y_pred)))
# print('F1 score after feature selection using lasso regression: {}'.format(f1_score(Y_test, y_pred, average = 'macro')))
# print('Precision score after feature selection using lasso regression: {}'.format(precision_score(Y_test, y_pred, average = 'macro')))
# print('Recall after feature selection using lasso regression: {}'.format(recall_score(Y_test, y_pred, average = 'macro')))

sel_R = SelectFromModel(LogisticRegression(C=1, penalty='l2', solver='liblinear'))
sel_R.fit(X_train, np.ravel(Y_train,order='C'))
sel_R.get_support()
X_train = pd.DataFrame(X_train)

selected_feat_R = X_train.columns[(sel_R.get_support())]
print('total features: {}'.format((X_train.shape[1])))
print('selected features using ridge regression: {}'.format(len(selected_feat_R)))
print('features selected: {}'.format(selected_feat_R))

X_train_selected_R = sel_R.transform(X_train)
X_test_selected_R = sel_R.transform(X_test)
X_train_selected_R.shape, X_test_selected_R.shape

print(X_train_selected_R[:20])

# clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
# clf.fit(X_train_selected_R,np.ravel(Y_train,order='C'))
# y_pred = clf.predict(X_test_selected_R)
# print('Accuracy after feature selection using ridge regression: {}'.format(accuracy_score(Y_test, y_pred)))
# print('F1 score after feature selection using ridge regression: {}'.format(f1_score(Y_test, y_pred, average = 'macro')))
# print('Precision score after feature selection using ridge regression: {}'.format(precision_score(Y_test, y_pred, average = 'macro')))
# print('Recall after feature selection using ridge regression: {}'.format(recall_score(Y_test, y_pred, average = 'macro')))