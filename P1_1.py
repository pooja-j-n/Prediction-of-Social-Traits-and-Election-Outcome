import datetime
import numpy as np
import pdb
from scipy.io import loadmat
from sklearn.svm import SVR
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
print("-----------------------1.1--------------------------")

dirpath = "/Applications/P4/"
#Part 1.1
train_anno = loadmat(dirpath + "train-anno.mat" )
face_landmark = train_anno['face_landmark']
trait_annotation = train_anno['trait_annotation']

#print(face_landmark.shape) (491, 160)
#print(trait_annotation.shape) (491, 14)

num = 14

X_train, X_test, y_train, y_test = train_test_split(face_landmark, trait_annotation,  test_size = 0.2, random_state=5, shuffle=True)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

accuracies = np.zeros(shape=(num,2))
precision = np.zeros(shape=(num,2))

C_range = np.power(2.0, list(range(5, 13)))
epsilon_range = np.power(2.0, list(range(-9, 1)))
gamma_range = np.power(2.0, list(range(-17, 5)))

parameters = {'C':C_range, 'epsilon':epsilon_range, 'gamma': gamma_range}
print("Parameters for model----------- ")
print(str(parameters))

for i in range(num):
    trait_train = y_train[:,i]
    trait_test = y_test[:,i]
    svrmodel = SVR()
    clf = GridSearchCV(svrmodel, parameters, cv=10, scoring='neg_mean_squared_error')
    print(str(datetime.datetime.now()) + " Training model-----------------"+str(i+1))
    clf.fit(X_train_scaled,trait_train)
    print("Best score for model-----------"+str(i+1))
    print(str(clf.best_score_))
    print("Best params for model----------" + str(i + 1))
    print(str(clf.best_params_))
    print("Calculating accuracy for model-- " +str(i+1))
    threshold = np.mean(trait_train)
    p1 = clf.predict(X_train_scaled) >= threshold
    q1 = trait_train >= threshold
    accuracies[i][0] = np.sum(p1 == q1) / len(q1)
    precision[i][0] = precision_score(q1, p1)
    p2 = clf.predict(X_test_scaled) >= threshold
    q2 = trait_test >= threshold
    accuracies[i][1] = np.sum(p2 == q2) / len(q2)
    precision[i][1] = precision_score(q2, p2)



print("Accuracies are ")
print(str(accuracies))
print("Precisions are ")
print(str(precision))

#Part 1.2
