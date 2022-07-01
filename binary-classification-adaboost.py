# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:19:38 2022

"""
import math
import matplotlib.pyplot as plt

def read_data():
    with open('dataCircle.txt') as f:
        lines = f.readlines()
    
    examples = []
    features_1 = []
    features_2 = []
    for line in lines:
        feature_1 = float(line.split()[0])
        feature_2 = float(line.split()[1])
        label = float(line.split()[2])
        
        features_1.append(feature_1)
        features_2.append(feature_2)
        example = {'x': feature_1, 'y': feature_2, 'label': label}
        examples.append(example)
    
    
    max_x = int(round(max(features_1)))
    min_x = int(round(min(features_1)))
    max_y = int(round(max(features_2)))
    min_y = int(round(min(features_2)))

    return examples, min_x, max_x, min_y, max_y

def weak_classifiers_list():
    examples, min_x, max_x, min_y, max_y = read_data()
    weak_classifiers = []
    
    x_range = [p/100 for p in range((min_x+1)*100, max_x*100)]
    y_range = [p/100 for p in range((min_y+1)*100, max_y*100)]
    
    
    for i in x_range:
        classifier_dict = {'threshold': i, 'axis': 'x', 'parity': 1}
        weak_classifiers.append(classifier_dict)
    
    for i in y_range:
        classifier_dict = {'threshold': i, 'axis': 'y', 'parity': 1}
        weak_classifiers.append(classifier_dict)
    
    return weak_classifiers
    

def weak_classifier(axis, threshold, parity, examples, importances):
    weighted_error = 0
    classification_results = []
    
    for example_index in range(0, len(examples)):
        example = examples[example_index]
        importance = importances[example_index]
        
        label = example.get('label')
        result = -1
        if (example.get(axis) > threshold):
            result = 1
        
        if (parity < 1):
            result = result*-1
        
        sigma = 0
        if (result != label):
            sigma = 1
        
        weighted_error = weighted_error + sigma*importance
        classification_results.append(result)
        
    if (weighted_error > 0.5):
        parity = parity * -1
        weighted_error = 1 - weighted_error
        classification_results = shift_signs(classification_results)
    
    return classification_results, weighted_error, parity

def shift_signs(lst):
    shifted_lst = []
    for item in lst:
        shifted_lst.append(item*-1)
    return shifted_lst

def select_best_weak_classifier(examples, importances):
    min_error = 0.6
    best_threshold = 0
    best_axis = ''
    best_parity = 0
    best_classification_results = []
    for weak_classifier_it in weak_classifiers_list():
        threshold = weak_classifier_it.get('threshold')
        axis = weak_classifier_it.get('axis')
        parity = weak_classifier_it.get('parity')
        classification_results, weighted_error, parity = weak_classifier(axis, threshold, parity, examples, importances)
    
        if (weighted_error < min_error):
            best_threshold = threshold
            best_axis = axis
            best_parity = parity
            best_classification_results = classification_results.copy()
            min_error = weighted_error
    
    #calculate alpha
    confidence_alpha = math.log(( (1-min_error)/(min_error+0.000000001) ))*0.5
    
    #create classifier dictionary to store parameters
    best_classifier = {'threshold': best_threshold, 'axis': best_axis, 'parity': best_parity, 'results': best_classification_results, 'weighted_error': min_error, 'alpha': confidence_alpha}
    
    #calculate normalization factor
    z = 0
    for importance_index in range(0, len(importances)):
        importance = importances[importance_index]
        label = examples[importance_index].get('label')
        h = best_classification_results[importance_index]
        z = z + math.exp(-1*confidence_alpha*label*h)*importance
    
    #update importances
    importances_upd = []
    for importance_index in range(0, len(importances)):
        importance = importances[importance_index]
        label = examples[importance_index].get('label')
        h = best_classification_results[importance_index]
        importance_upd = (math.exp(-1*confidence_alpha*label*h)*importance)/z
        importances_upd.append(importance_upd)
    
    return best_classifier, importances_upd
  
def define_initial_importances(examples):
    size = len(examples)
    initial_importance = 1/size
    return [initial_importance] * size

def ada_boost(iteration_count):
    examples, min_x, max_x, min_y, max_y = read_data()
    importances = define_initial_importances(examples)
    
    best_weak_classifiers = []
    
    for i in range(iteration_count):
        best_classifier, importances = select_best_weak_classifier(examples, importances)
        best_weak_classifiers.append(best_classifier)
        
    return best_weak_classifiers, importances

def draw_report_graph(best_weak_classifiers):
    examples, min_x, max_x, min_y, max_y = read_data()
    
    #Plot examples
    positives_x = []
    positives_y = []
    negatives_x = []
    negatives_y = []
    for example in examples:
        if (example.get('label') == 1):
            positives_x.append(example.get('x'))
            positives_y.append(example.get('y'))
        else:
            negatives_x.append(example.get('x'))
            negatives_y.append(example.get('y'))
   
    plt.scatter(positives_x, positives_y, c ="green")
    plt.scatter(negatives_x, negatives_y, c ="red")
    
    for weak_classifier in best_weak_classifiers:
        axis = weak_classifier.get('axis')
        threshold = weak_classifier.get('threshold')
        if (axis == 'x'):
            plt.axvline(x=threshold, color='black')
        else:
            plt.axhline(y=threshold, color='black')
    
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.show()

def print_best_classifiers(best_weak_classifiers):
    for best_classifier in best_weak_classifiers:
        axis = best_classifier['axis']
        if (axis == 'x'):
            axis = 'Feature 1'
        else:
            axis = 'Feature 2'
        print("\n")
        print("Threshold:", str(best_classifier['threshold']) + ",", "Axis:", axis + ",", "Parity:", str(best_classifier['parity']) + ",", 'Weighted error:', str(best_classifier['weighted_error']) + ",", 'Confidence:', str(best_classifier['alpha']))

best_weak_classifiers, importances = ada_boost(100)

draw_report_graph(best_weak_classifiers)

print_best_classifiers(best_weak_classifiers)



