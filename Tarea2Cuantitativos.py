#!/usr/bin/python3
from ast import literal_eval

import numpy as np
import matplotlib.pyplot as plt

# Own modules
from linreg import grad_descent

def read_data(fileName):
    lines = []
    
    with open(fileName) as f:
        lines = f.read().splitlines()
    
    return np.matrix([literal_eval(elem) for elem in lines])

# handling for conversion to inputs for linear regression
days = {'lunes' : 0, 
        'martes' : 1,
        'miercoles' : 2,
        'jueves' : 3,
        'viernes' : 4,
        'sabado' : 5,
        'domingo': 6
        }

def time_weight(day, time):
    [hours, minutes] = [int(section) for section in time.split(':')]
    
    if days[day] != 0 or hours >= 8:
        day_weight = days[day] * 24 * 3
        hours_weight = (hours - 8) * 3
        minutes_weight = minutes / 20
        
        return (day_weight + hours_weight + minutes_weight)/10
    
    return None
    
# MAIN SCRIPT
    
# dispose the unnecesary auto-generated tk window
    
'''
from tkinter import filedialog, Tk

root = Tk()
root.withdraw()

fileName = filedialog.askopenfilename()
'''

thetas = []

import os

fileNames = [
        '341_points.txt',
        '347_points.txt',
        '350_points.txt',
        '353_points.txt',
        '2100_points.txt',
        '2679_points.txt'
        ]

for _, _, files in os.walk("."):  
    datasets = list(filter(lambda x : x.endswith('.txt'), files))
    print(datasets)
    if len(datasets) > 0 :
        fileNames = datasets
        

for fileName in fileNames:
    data = read_data(fileName)

    (rows, _) = data.shape
    
    xs = data[:, 0]
    ys = data[:, 1]
    
    plt.scatter(xs.tolist(), ys.tolist(), c='black')
    
    ones_column = np.ones(shape=(rows, 1))
    
    xs = np.append(ones_column, xs, axis=1)
    
    iters = 150
    
    theta, _ = grad_descent(xs, ys, 0.001, iters)
    
    plt.plot(xs[:, 1], theta[0] + xs[:, 1] * theta[1], 'c', linewidth=3)
    
    plt.show()
    
    thetas.append(theta)

# ---------------------PREDICTIONS-----------------------------

print('Predictions: ')

day = input('Ingrese el día: ')
time = input('Ingrese la hora [horas]:[minutos] (24h) ')

data_to_predict = np.matrix([1, time_weight(day, time)])

for theta in thetas:
    prediction = np.dot(data_to_predict, theta)

    print('Predicted motion:', prediction)