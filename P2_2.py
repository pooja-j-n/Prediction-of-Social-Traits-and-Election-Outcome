import datetime
import numpy as np
import os
import pdb
import pickle
from scipy.io import loadmat
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import hog
from skimage import io
from typing import Any

print("-----------------------2.2--------------------------")
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
params_path = "/Applications/P4/best_params2.pkl"
face_landmark = train_anno['face_landmark']
trait_annotation = train_anno['trait_annotation']

#senator
hog_features_sen = []
im_list = os.listdir(sen_folder)
im_list.sort()
for image_name in im_list:
    im = io.imread(sen_folder + image_name)
    fd, hog_image = hog(im, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualize=True,multichannel=True)
    hog_features_sen.append(fd)

hog_features_sen = np.array(hog_features_sen)
print(hog_features_sen.shape)

num = 14
num_sen = len(sen_vote_diff)
num_gov = len(gov_vote_diff)


traits_sen = np.zeros(shape=(num_sen,num))

best_params = pickle.load(open(params_path, 'rb'))

#--------------FOR TRAIN-------------------
hog_features = []
im_list = os.listdir(img_folder)
im_list.sort()
for image_name in im_list:
    im = io.imread(img_folder + image_name)
    fd, hog_image = hog(im, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualize=True,
                        multichannel=True)   #default values?
    hog_features.append(fd)

hog_features = np.array(hog_features)
print(hog_features.shape)

X_train_landmark, X_test_landmark,X_train_hog, X_test_hog, y_train, y_test = train_test_split(face_landmark, hog_features, trait_annotation, test_size = 0.2, random_state=5, shuffle=True)

scaler1 = MinMaxScaler()
X_train_landmark = scaler1.fit_transform(X_train_landmark)
X_test_landmark = scaler1.transform(X_test_landmark)
sen_face_landmark = scaler1.transform(sen_face_landmark)
X_train_landmark[X_train_landmark>1] = 1

scaler2 = MinMaxScaler()
X_train_hog = scaler2.fit_transform(X_train_hog)
X_test_hog = scaler2.transform(X_test_hog)
hog_features_sen = scaler2.transform(hog_features_sen)
X_train_hog[X_train_hog>1] = 1
landmarks_plus_hog = np.concatenate((sen_face_landmark, hog_features_sen), axis=1)

X_train = np.concatenate((X_train_landmark, X_train_hog), axis=1)
X_test = np.concatenate((X_test_landmark, X_test_hog), axis=1)
accuracies = np.zeros(shape=(num,2))

for i in range(num):
    trait_train = y_train[:,i]
    trait_test = y_test[:, i]
    params = best_params[i]
    for k in params:
        params[k] = [params.get(k)]
    svrmodel = SVR()
    clf = GridSearchCV(svrmodel, params, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
    print(str(datetime.datetime.now()) + " Training model-----------------" + str(i + 1))
    clf.fit(X_train, trait_train)
    preds = clf.predict(landmarks_plus_hog)
    traits_sen[0:num_sen, i] = preds


X_sen = np.zeros(shape=(num_sen//2,num))
y_sen = np.zeros(shape=(num_sen//2,1))

ind = 0
for i in range(0,num_sen,2):
    X_sen[ind] = traits_sen[i] - traits_sen[i+1]
    y_sen[ind] = np.sign(sen_vote_diff[i] - sen_vote_diff[i + 1])
    ind += 1

for i in range(len(X_sen)//2):
    X_sen[i] = -1  * X_sen[i]
    y_sen[i] = -1 * y_sen[i]


X_train_sen, X_test_sen, y_train_sen, y_test_sen = train_test_split(X_sen, y_sen, test_size = 0.20, random_state=67, shuffle = True)

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


print("---------GOVERNOR-----------")



hog_features_gov = []
im_list = os.listdir(gov_folder)
im_list.sort()
for image_name in im_list:
    im = io.imread(gov_folder + image_name)
    fd, hog_image = hog(im, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualize=True,multichannel=True)
    hog_features_gov.append(fd)

hog_features_gov = np.array(hog_features_gov)
print(hog_features_gov.shape)

num = 14
num_gov = len(gov_vote_diff)


traits_gov = np.zeros(shape=(num_gov,num))

best_params = pickle.load(open(params_path, 'rb'))

#--------------FOR TRAIN-------------------
hog_features = []
im_list = os.listdir(img_folder)
im_list.sort()
for image_name in im_list:
    im = io.imread(img_folder + image_name)
    fd, hog_image = hog(im, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), visualize=True,
                        multichannel=True)   #default values?
    hog_features.append(fd)

hog_features = np.array(hog_features)
print(hog_features.shape)

X_train_landmark, X_test_landmark,X_train_hog, X_test_hog, y_train, y_test = train_test_split(face_landmark, hog_features, trait_annotation, test_size = 0.2, random_state=5, shuffle=True)

scaler1 = MinMaxScaler()
X_train_landmark = scaler1.fit_transform(X_train_landmark)
X_test_landmark = scaler1.transform(X_test_landmark)
gov_face_landmark = scaler1.transform(gov_face_landmark)
X_train_landmark[X_train_landmark>1] = 1

scaler2 = MinMaxScaler()
X_train_hog = scaler2.fit_transform(X_train_hog)
X_test_hog = scaler2.transform(X_test_hog)
hog_features_gov = scaler2.transform(hog_features_gov)
X_train_hog[X_train_hog>1] = 1
landmarks_plus_hog = np.concatenate((gov_face_landmark, hog_features_gov), axis=1)

X_train = np.concatenate((X_train_landmark, X_train_hog), axis=1)
X_test = np.concatenate((X_test_landmark, X_test_hog), axis=1)
accuracies = np.zeros(shape=(num,2))

for i in range(num):
    trait_train = y_train[:,i]
    trait_test = y_test[:, i]
    params = best_params[i]
    for k in params:
        params[k] = [params.get(k)]
    svrmodel = SVR()
    clf = GridSearchCV(svrmodel, params, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
    print(str(datetime.datetime.now()) + " Training model-----------------" + str(i + 1))
    clf.fit(X_train, trait_train)
    preds = clf.predict(landmarks_plus_hog)
    traits_gov[0:num_gov, i] = preds


X_gov = np.zeros(shape=(num_gov//2,num))
y_gov = np.zeros(shape=(num_gov//2,1))

ind = 0
for i in range(0,num_gov,2):
    X_gov[ind] = traits_gov[i] - traits_gov[i+1]
    y_gov[ind] = np.sign(gov_vote_diff[i] - gov_vote_diff[i + 1])
    ind += 1

for i in range(len(X_gov)//2):
    X_gov[i] = -1  * X_gov[i]
    y_gov[i] = -1 * y_gov[i]



X_train_gov, X_test_gov, y_train_gov, y_test_gov = train_test_split(X_gov, y_gov, test_size = 0.20, random_state=42, shuffle = True)

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





#-------------2.3--------------

#senators


corr=np.zeros(shape=num)

pdb.set_trace()

for i in range(num):
    X_corr = np.zeros(shape=num_sen//2)
    y_corr = np.zeros(shape=num_sen//2)
    ind = 0
    X = traits_sen[:,i]
    for j in range(0, num_sen, 2):
        X_corr[ind] = X[j + 1] - X[j]
        y_corr[ind] = sen_vote_diff[j + 1]
        if (y_corr[ind]==0):
            y_corr[ind] = 1
        ind += 1
    corr[i] = np.correlate(X_corr, y_corr)

print(str(corr))
corr_gov=np.zeros(shape=num)

#pdb.set_trace()

for i in range(num):
    X_corr = np.zeros(shape=num_gov//2)
    y_corr = np.zeros(shape=num_gov//2)
    ind = 0
    X = traits_gov[:,i]
    for j in range(0, num_gov, 2):
        X_corr[ind] = X[j + 1] - X[j]
        y_corr[ind] = gov_vote_diff[j + 1]
        if (y_corr[ind]==0):
            y_corr[ind] = 1
        ind += 1
    corr_gov[i] = np.correlate(X_corr, y_corr)
print(str(corr_gov))