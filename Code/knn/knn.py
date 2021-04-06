from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt


file_out = pd.read_csv('knn\data_filled.csv')

#* target labels
label = pd.read_csv('knn\labels_filled.csv')
label = label.iloc[:].values
label = np.ravel(label) #reshape to (N,)

#* feature 1: symptoms score
#* sum all of the symptoms (cough, fever,sore_throat, shortness_of_breath, head_ache)
x1 = file_out.iloc[:, 0:5].values 
x1 = np.sum(x1, axis = 1) 

#* feature 2: contact status (Contact with confirmed, Abroad, Other)
x2 = file_out.iloc[:, 7].values 

#placeholder for features
features = np.zeros((len(x1), 2))
features[:, 0] = x1
features[:, 1] = x2


X = features
y = label

# mu = np.mean(X, 0)
# sigma = np.std(X, 0)
# X = (X - mu ) / sigma

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)

# sklearn_classifier = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
# # (274955, 2) -> probabilities that the data point belongs to class 1 and class 2
# #y_score = sklearn_classifier.predict_proba(X)

# #print(sklearn_classifier.predict(X))

# #* Prediction of the labels
# y_pred = sklearn_classifier.predict(X) 

# #* Accuracy
# print("Accuracy: ", sklearn_classifier.score(X, y)) # = 0.9479805786401411

# #* precision, recall, fbeta_score, number of occurences for each label
# #print(precision_recall_fscore_support(label,y_pred))
# arr = precision_recall_fscore_support(label,y_pred, average='binary')
# print("Precision: ", arr[0])
# print("Recall: ", arr[1])
# print("F-beta score: ", arr[2])

# TODO: draw a visualization try different number of neighbors 

neighbor_list = [3, 5, 7, 9, 11]
accuracy = []
precision = []
recall = []
f_beta = []

for i in range(len(neighbor_list)):
    sklearn_classifier = KNeighborsClassifier(n_neighbors=neighbor_list[i]).fit(X_train, y_train)
    
    #* Prediction of the labels
    y_pred = sklearn_classifier.predict(X) 

    accuracy.append(sklearn_classifier.score(X, y))

    #* precision, recall, fbeta_score, number of occurences for each label
    arr = precision_recall_fscore_support(label,y_pred, average='binary')
    precision.append(arr[0])
    recall.append(arr[1])
    f_beta.append(arr[2])


#* Plot a bar graph of accuracy, precision, recall, and F-beta for different number of neighbors
x = np.arange(len(neighbor_list)) # the label locations
width = 0.15 # the label locations

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5 * width, accuracy, width, label='Accuracy')
rects2 = ax.bar(x - width/2, precision, width, label='Precision')
rects3 = ax.bar(x + width/2, recall, width, label='Recall')
rects4 = ax.bar(x + 1.5 * width, f_beta, width, label='F-beta')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_xlabel('Number of Neighbors')
ax.set_title('Accuracy, Precision, Recall, and F-beta for different number of neighbors')
ax.set_xticks(x)
ax.set_xticklabels(neighbor_list)
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.4f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

plt.show()