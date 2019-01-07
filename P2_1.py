import datetime
import numpy as np
import os
import pdb
from scipy.io import loadmat
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import hog
from skimage import io
import time

print("-----------------------2.1--------------------------")
print("-----------------------SENATOR--------------------------")
dirpath = "/Applications/P4/"
img_folder = dirpath+"img/"
sen_folder = dirpath+"img-elec/senator/"
gov_folder = dirpath+"img-elec/governor/"
train_anno = loadmat(dirpath + "train-anno.mat" )
sen_anno = loadmat(dirpath + "stat-sen.mat")
sen_face_landmark = sen_anno['face_landmark']
sen_vote_diff= sen_anno['vote_diff']
gov_anno = loadmat(dirpath + "stat-gov.mat")
gov_face_landmark = gov_anno['face_landmark']
gov_vote_diff = gov_anno['vote_diff']

#senator

hog_features = []
im_list = os.listdir(sen_folder)
im_list.sort()
for image_name in im_list:
    im = io.imread(sen_folder + image_name)
    fd, hog_image = hog(im, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualize=True,
                        multichannel=True)   #default values?
    hog_features.append(fd)

hog_features = np.array(hog_features)
print(hog_features.shape)

num = 14
num_sen = len(sen_vote_diff)
num_gov = len(gov_vote_diff)

landmarks_plus_hog = np.concatenate((sen_face_landmark, hog_features), axis=1)
num_features = landmarks_plus_hog.shape[1]

X_sen = np.zeros(shape=(num_sen//2,num_features))
y_sen = np.zeros(shape=(num_sen//2,1))

ind = 0
for i in range(0,num_sen,2):
    X_sen[ind] = landmarks_plus_hog[i] - landmarks_plus_hog[i+1]
    y_sen[ind] = np.sign(sen_vote_diff[i] - sen_vote_diff[i + 1])
    ind += 1

for i in range(len(X_sen)//2):
    X_sen[i] = -1  * X_sen[i]
    y_sen[i] = -1 * y_sen[i]



X_train_sen, X_test_sen, y_train_sen, y_test_sen = train_test_split(X_sen, y_sen, test_size = 0.20, random_state=34, shuffle = True)

scaler1 = MinMaxScaler()
X_train_sen = scaler1.fit_transform(X_train_sen)
X_test_sen = scaler1.transform(X_test_sen)
X_train_sen[X_train_sen>1] = 1


print("Length of training set " + str(len(X_train_sen)))
print("Length of test set " + str(len(X_test_sen)))

y_train_sen[y_train_sen == 0] = 1
y_test_sen[y_test_sen == 0] = 1

svcmodel = LinearSVC(fit_intercept=False)
C_range = np.power(2.0, list(range(-5, 10)))
parameters = {'C':C_range}
clf = GridSearchCV(svcmodel, parameters, cv=10, n_jobs=-1)
clf.fit(X_train_sen,y_train_sen)
print("Best score for model-----------")
print(str(clf.best_score_))
print("Best params for model----------")
print(str(clf.best_params_))
print("Calculating accuracy for model-- ")
p1 = clf.predict(X_train_sen)
q1 = y_train_sen
q1 = q1.reshape(p1.shape)
train_accuracy = np.sum(p1 == q1) / len(q1)
train_precision = precision_score(q1, p1)
print("Train accuracy " + str(train_accuracy))
print("Train precision " + str(train_precision))
p2 = clf.predict(X_test_sen)
q2 = y_test_sen
q2 = q2.reshape(p2.shape)
test_accuracy = np.sum(p2 == q2) / len(q2)
test_precision = precision_score(q2, p2)
print("Test accuracy " + str(test_accuracy))
print("Test precision " + str(test_precision))
print("p2 " + str(p2))
print("q2 " + str(q2))






#governor
print("-----------------------GOVERNOR--------------------------")
hog_features = []
im_list = os.listdir(gov_folder)
im_list.sort()
for image_name in im_list:
    im = io.imread(gov_folder + image_name)
    fd, hog_image = hog(im, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualize=True,
                        multichannel=True)   #default values?
    hog_features.append(fd)

hog_features = np.array(hog_features)
print(hog_features.shape)

num = 14
num_sen = len(sen_vote_diff)
num_gov = len(gov_vote_diff)

landmarks_plus_hog = np.concatenate((gov_face_landmark, hog_features), axis=1)
num_features = landmarks_plus_hog.shape[1]

X_gov = np.zeros(shape=(num_gov//2,num_features))
y_gov = np.zeros(shape=(num_gov//2,1))

ind = 0
for i in range(0,num_gov,2):
    X_gov[ind] = landmarks_plus_hog[i] - landmarks_plus_hog[i+1]
    y_gov[ind] = np.sign(gov_vote_diff[i] - gov_vote_diff[i + 1])
    ind += 1

for i in range(len(X_gov)//2):
    X_gov[i] = -1  * X_gov[i]
    y_gov[i] = -1 * y_gov[i]



X_train_gov, X_test_gov, y_train_gov, y_test_gov = train_test_split(X_gov, y_gov, test_size = 0.20, random_state=34, shuffle = True)

scaler1 = MinMaxScaler()
X_train_gov = scaler1.fit_transform(X_train_gov)
X_test_gov = scaler1.transform(X_test_gov)
X_train_gov[X_train_gov>1] = 1

print("Length of training set " + str(len(X_train_gov)))
print("Length of test set " + str(len(X_test_gov)))

y_train_gov[y_train_gov == 0] = 1
y_test_gov[y_test_gov == 0] = 1

svcmodel = LinearSVC(fit_intercept=False)
C_range = np.power(2.0, list(range(-5, 10)))
parameters = {'C':C_range}
clf = GridSearchCV(svcmodel, parameters, cv=10, n_jobs=-1)
clf.fit(X_train_gov,y_train_gov)
print("Best score for model-----------")
print(str(clf.best_score_))
print("Best params for model----------")
print(str(clf.best_params_))
print("Calculating accuracy for model-- ")
p1 = clf.predict(X_train_gov)
q1 = y_train_gov
q1 = q1.reshape(p1.shape)
train_accuracy = np.sum(p1 == q1) / len(q1)
train_precision = precision_score(q1, p1)
print("Train accuracy " + str(train_accuracy))
print("Train precision " + str(train_precision))
p2 = clf.predict(X_test_gov)
q2 = y_test_gov
q2 = q2.reshape(p2.shape)
test_accuracy = np.sum(p2 == q2) / len(q2)
test_precision = precision_score(q2, p2)
print("Test accuracy " + str(test_accuracy))
print("Test precision " + str(test_precision))
print("p2 " + str(p2))
print("q2 " + str(q2))
