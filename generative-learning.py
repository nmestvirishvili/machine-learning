# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:09:24 2022

@author: Natia_Mestvirishvili
"""

import os
import cv2
import numpy
from numpy import linalg as LA
import math

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

def estimate_parameters():
    negatives = load_images_from_folder("negatives")
    positives = load_images_from_folder("positives")
    phi = len(positives)/(len(positives)+len(negatives))
    miu_1 = [0, 0, 0, 0, 0, 0]
    for feature_vector in positives:
        feature_numpy = numpy.array(feature_vector)
        miu_1 = miu_1 + feature_numpy
    miu_1 = miu_1/len(positives)
    miu_0 = [0, 0, 0, 0, 0, 0]
    for feature_vector in negatives:
        feature_numpy = numpy.array(feature_vector)
        miu_0 = miu_0 + feature_numpy
    miu_0 = miu_0/len(negatives)
    
    sigma = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],]
    for feature_vector in positives:
        factor = numpy.array(feature_vector) - miu_1
        matrix = numpy.outer(factor, factor)
        sigma = sigma + matrix
        
    sigma = sigma/(len(positives)+len(negatives))
    
    return phi, miu_0, miu_1, sigma

def estimate(feature_vector, ground_truth, phi, miu_0, miu_1, sigma):
    n = 2
    covariance_norm = LA.norm(sigma)
    common_factor = 1 / ( math.pow(math.pi * 2, n/2) * math.sqrt(covariance_norm) )
    
    #Posterior probability for a positive scenario
    p_y_positive = phi
    feature_minus_miu = numpy.array(feature_vector) - miu_1
    step_1 = -0.5 * numpy.transpose(feature_minus_miu)
    step_2 = step_1.dot(numpy.linalg.inv(sigma))
    step_3 = step_2.dot(feature_minus_miu)
    conditional_prob_positive = common_factor * math.exp(step_3) * p_y_positive
    
    #Posterior probability for a negative scenario
    p_y_negative = 1 - phi
    feature_minus_miu = numpy.array(feature_vector) - miu_0
    step_1 = -0.5 * numpy.transpose(feature_minus_miu)
    step_2 = step_1.dot(numpy.linalg.inv(sigma))
    step_3 = step_2.dot(feature_minus_miu)
    conditional_prob_negative = common_factor * math.exp(step_3) * p_y_negative
    
    prediction = 0
    if (conditional_prob_positive > conditional_prob_negative):
        prediction = 1
    
    prediction_correct = False
    if (prediction == ground_truth):
        prediction_correct = True
    return prediction, prediction_correct
    

phi, miu_0, miu_1, sigma = estimate_parameters()

print("Parameters")
print("Phi", phi)
print("mean vector positives", miu_1)
print("mean vector negatives", miu_0)
print("covariance matrix", sigma)

positives = load_images_from_folder("positives")
negatives = load_images_from_folder("negatives")

correctly_predicted = 0
for example in positives:
    prediction, prediction_correct = estimate(example, 1, phi, miu_0, miu_1, sigma)
    if (prediction_correct):
        correctly_predicted = correctly_predicted + 1

for example in negatives:
    prediction, prediction_correct = estimate(example, 0, phi, miu_0, miu_1, sigma)
    if (prediction_correct):
        correctly_predicted = correctly_predicted + 1

prediction_accuracy = correctly_predicted / (len(negatives) + len(positives))
print("Prediction accuracy:", prediction_accuracy)