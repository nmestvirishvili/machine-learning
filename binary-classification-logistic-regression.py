# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:37:23 2022

@author: Natia_Mestvirishvili
"""
import random
import matplotlib.pyplot as plt
import math

def read_file():
    lines = []
    with open('data.txt') as f:
        lines = f.readlines()
  
    positives = []
    negatives = []
    for line in lines:
        line_data = line.split(" ")
        pair = {'x1': float(line_data[0]), 'x2': float(line_data[1]), 'class': float(line_data[2])}
        if ('1' in line_data[2]):
            positives.append(pair)
        else:
            negatives.append(pair)
    
    return positives, negatives

def plot_result(positives, negatives, final_weights, init_weights):
    x1_coordinates = []
    x2_coordinates = []
    trained_classifier_x = []
    trained_classifier_y = []
    random_classifier_x = []
    random_classifier_y = []
    for positive_example in positives:
        x1 = positive_example.get('x1')
        x1_coordinates.append(x1)
        x2_coordinates.append(positive_example.get('x2')) 
        trained_classifier_x.append(x1)
        random_classifier_x.append(x1)
        trained_classifier_y.append(calc_classifier_point_val(positive_example, final_weights))
        random_classifier_y.append(calc_classifier_point_val(positive_example, init_weights))
    plt.scatter(x1_coordinates, x2_coordinates, label = "Positives", c='green')
    plt.plot(trained_classifier_x, trained_classifier_y, label = "Learned classifier", c='black')
    plt.plot(random_classifier_x, random_classifier_y, label = "Initial random classifier", c='blue')
    
    x1_coordinates = []
    x2_coordinates = []
    for negative_example in negatives:
        x1_coordinates.append(negative_example.get('x1'))
        x2_coordinates.append(negative_example.get('x2'))
    plt.scatter(x1_coordinates, x2_coordinates, label = "Negatives", c='red')
    
def calc_classifier_point_val(example, weights):
    return (weights[0] + weights[1]*example.get('x1')) * (-1 / weights[2])

def init_random_weights():
    weights = []
    weights.append(random.uniform(-0.01, 0.01))
    weights.append(random.uniform(-0.01, 0.01))
    weights.append(random.uniform(-0.01, 0.01))
    return weights

def calc_polynomial(weights, example):
    return weights[0] + weights[1]*example.get('x1') + weights[2]*example.get('x2')
    
def calc_sigmoid(weights, example):
    result_polynomial = calc_polynomial(weights, example)
    exp_term = math.exp(-1*result_polynomial)
    return 1/(1+exp_term)

def x_i_j_term(weight_index, example):
    switcher = {
        0: 1,
        1: example.get('x1'),
        2: example.get('x2'),
    }
    return switcher.get(weight_index)

def update_weights(data_points, weights, learning_rate):
    for example in data_points:
        input_class = example.get('class')
        h0 = calc_sigmoid(weights, example)
        for j in range(0, len(weights)):
            weights[j] = weights[j] +  learning_rate * (input_class - h0) * x_i_j_term(j, example) 
    return weights

def train_model(data, learning_rate, number_of_iterations):
    weights = init_random_weights()
    init_weights = weights.copy()
    for i in range(0, number_of_iterations):
        weights = update_weights(data, weights, learning_rate)
    return init_weights, weights

def main(learning_rate):
    positives, negatives = read_file()
    data_points = positives + negatives
    init_weights, final_weights = train_model(data_points, learning_rate, 1000)
    print(final_weights)
    print(init_weights)
    plot_result(positives, negatives, final_weights, init_weights)
    return final_weights
    
weights = main(0.01)
print(weights)


