import datetime
import numpy as np
import os
import pdb
import pickle
from scipy.io import loadmat
from sklearn.svm import SVR
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import hog
from skimage import io
print("-----------------------1.2--------------------------")
dirpath = "/Applications/P4/"
img_folder = dirpath+"img/"
train_anno = loadmat(dirpath + "train-anno.mat" )
face_landmark = train_anno['face_landmark']
trait_annotation = train_anno['trait_annotation']

num = 14
params_path = "/Applications/P4/best_params2.pkl"
#scale landmarks and hog separately ?
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
X_train_landmark[X_train_landmark>1] = 1

scaler2 = MinMaxScaler()
X_train_hog = scaler2.fit_transform(X_train_hog)
X_test_hog = scaler2.transform(X_test_hog)
X_train_hog[X_train_hog>1] = 1

X_train = np.concatenate((X_train_landmark, X_train_hog), axis=1)
X_test = np.concatenate((X_test_landmark, X_test_hog), axis=1)
print(X_train.shape)
accuracies = np.zeros(shape=(num,2))
precision = np.zeros(shape=(num,2))

C_range = np.power(2.0, list(range(5, 13)))
epsilon_range = np.power(2.0, list(range(-9, 1)))
gamma_range = np.power(2.0, list(range(-17, 5)))


parameters = {'C': C_range, 'epsilon': epsilon_range, 'gamma': gamma_range}
print("Parameters for models----------- " + str(parameters))
params = []
for i in range(9,10):
    trait_train = y_train[:,i]
    trait_test = y_test[:,i]
    svrmodel = SVR()
    clf = GridSearchCV(svrmodel, parameters, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
    print(str(datetime.datetime.now()) + " Training model-----------------"+str(i+1))
    clf.fit(X_train,trait_train)
    print("Best score for model-----------"+str(i+1))
    print(str(clf.best_score_))
    print("Best params for model----------" + str(i + 1))
    print(str(clf.best_params_))
    params.append(clf.best_params_)
    print("Calculating accuracy for model-- " +str(i+1))
    threshold = np.mean(trait_train)
    p1 = clf.predict(X_train) >= threshold
    q1 = trait_train >= threshold
    accuracies[i][0] = np.sum(p1 == q1) / len(q1)
    p2 = clf.predict(X_test) >= threshold
    q2 = trait_test >= threshold
    accuracies[i][1] = np.sum(p2 == q2) / len(q2)
    print("Model " + str(i+1) +" Train accuracy " + str(accuracies[i][0]) + " Test accuracy " + str(accuracies[i][1]))
    precision[i][0] = precision_score(q1, p1)
    precision[i][1] = precision_score(q2, p2)
    print("Model " + str(i + 1) + " Train precision " + str(precision[i][0]) + " Test precision " + str(precision[i][1]))


pickle.dump(params, open(params_path, 'wb'))
print("Accuracies are ")
print(str(accuracies))

