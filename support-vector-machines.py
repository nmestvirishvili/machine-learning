# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:23:26 2022

@author: Natia_Mestvirishvili
"""

import os
import cv2
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import svm, datasets

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if (filename.endswith(".png")):
            img = cv2.imread(os.path.join(folder,filename))
            reds = []
            greens = []
            blues = []
            if img is not None:
                for i in range (len(img)):
                    for j in range (len(img[0])):
                        reds.append(img[i][j][0])
                        greens.append(img[i][j][1])
                        blues.append(img[i][j][2])
            features = []
            features.append(min(reds))
            features.append(min(greens))
            features.append(min(blues))
            features.append(round(sum(reds) / len(reds),2))
            features.append(round(sum(greens) / len(greens),2))
            features.append(round(sum(blues) / len(blues),2))
            images.append(features)
    return images

def find_optimal_hyper_params(x, y):
    parameters = {'gamma':('scale', 'auto'), 'C':[1, 10]}
    iris = datasets.load_iris()
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(iris.data, iris.target)
    C = clf.best_params_.get('C')
    gamma = clf.best_params_.get('gamma')
    return C, gamma
    
def create_training_sets():
    negatives = load_images_from_folder("negatives")
    positives = load_images_from_folder("positives")
    positives.extend(negatives)
    x = np.array(positives)
    zeros = np.zeros(30)
    ones = np.ones(30)
    y = np.concatenate((ones, zeros))
    return x,y

x,y = create_training_sets()
optimal_C, optimal_gamma = find_optimal_hyper_params(x,y)

print("Optimal Parameters:")
print("C:", optimal_C)
print("Gamma:", optimal_gamma)

test_x = x
test_y = y

print("\nExperiment results")
#Linear
clf = make_pipeline(StandardScaler(), SVC(gamma=optimal_gamma, C=optimal_C, kernel='linear'))
clf.fit(x, y)
print("Linear:", clf.score(test_x, test_y))
#Polynomial Degree 1
clf = make_pipeline(StandardScaler(), SVC(gamma=optimal_gamma, C=optimal_C, kernel='poly', degree=1))
clf.fit(x, y)
print("Polynomial degree 1:", clf.score(test_x, test_y))
#Polynomial Degree 2
clf = make_pipeline(StandardScaler(), SVC(gamma=optimal_gamma, C=optimal_C, kernel='poly', degree=2))
clf.fit(x, y)
print("Polynomial degree 2:", clf.score(test_x, test_y))
#Polynomial Degree 3
clf = make_pipeline(StandardScaler(), SVC(gamma=optimal_gamma, C=optimal_C, kernel='poly', degree=3))
clf.fit(x, y)
print("Polynomial degree 3:", clf.score(test_x, test_y))
#Polynomial Degree 4
clf = make_pipeline(StandardScaler(), SVC(gamma=optimal_gamma, C=optimal_C, kernel='poly', degree=4))
clf.fit(x, y)
print("Polynomial degree 4:", clf.score(test_x, test_y))
#Sigmoid
clf = make_pipeline(StandardScaler(), SVC(gamma=optimal_gamma, C=optimal_C, kernel='sigmoid'))
clf.fit(x, y)
print("Sigmoid:", clf.score(test_x, test_y))
#Radial basis function (RDF) kernel
clf = make_pipeline(StandardScaler(), SVC(gamma=optimal_gamma, C=optimal_C))
clf.fit(x, y)
print("Radial basis:", clf.score(test_x, test_y))
       
   

