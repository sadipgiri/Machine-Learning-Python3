#!/usr/bin/env python3

"""
    linear_regression.py - Python3 program to implement linear regression from scratch.
    Author: Sadip Giri (sadipgiri@bennington.edu)
    Created: Dec. 1, 2017
"""

from numpy import *
from progressbar import ProgressBar

# this is just to make user know that program is running as it takes time to fit the model when ML(Machine Learning) stuff
bar = ProgressBar()

# this is mean squared error (MSE)
def error_of_given_points(b, m, points):
    sum_errors = 0
    number_of_observations = len(points)    # how many points we have that is number of rows in a dataset table
    # find error for all the given points (dataset)
    for i in range(number_of_observations):
        x_value = points[i, 0]  # getting x_value of a point in that row
        y_value = points[i, 1]
        sum_errors += (y_value - (m*x_value + b)) ** 2  # summing all the errors found
    return sum_errors/float(number_of_observations)

# this gives the error of fitting a best-fit line in a given points 
# so we need to minimize the error as much as we could. This is where gradient descent comes!!

# this function runs given number of iterations and returns the optimized b and m values
def iterates_gradient_descent(points, given_m_value, given_b_value, learning_rate, number_of_iterations):
    # start with intial given values of b & m
    m = given_m_value
    b = given_b_value
    for i in bar(range(number_of_iterations)):
        b, m = step_gradient_descent(b, m, learning_rate, array(points))
    return [b, m]   # returns the updated b & m value [somewhat accurate everytime]

def step_gradient_descent(b, m, learning_rate, points):
    number_of_observations = len(points) 
    N = float(len(points))  # why float?: we need accurate values [not changing them to ints] 
    b_gradient = 0
    m_gradient = 0
    for i in range(number_of_observations):
        x_value = points[i, 0]
        y_value = points[i, 1]
        # calculate the gradients first:
        b_gradient += - (2/N) * (y_value - (m * x_value + b))
        m_gradient += - (2/N) * (y_value - (m * x_value + b)) * (x_value)
    # Now, updating b and m values to return those
    new_b_value = b - (learning_rate * b_gradient)  # applying given learning rate which determines how large of a step we need to take to minimize the cost function
    new_m_value = m - (learning_rate * m_gradient)
    return [new_b_value, new_m_value]

if __name__ == '__main__':
    # Testing the implementation of gradient descent in linear regression:
    # get the points from our dataset
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    given_b_value = 1    # this is our guess
    given_m_value = 3   # guess
    total_iterations = 1000
    print("We are starting gradient descent with b: {0}, m: {1}, error: {2}".format(given_b_value, given_m_value, error_of_given_points(given_b_value, given_m_value, points)))
    [b, m] = iterates_gradient_descent(points, given_m_value, given_b_value, learning_rate, total_iterations) 
    print("After implementing Gradient Descent Algorithm: ")
    print("----------------")
    print("New value of b: {0}, m: {1}, error: {2}".format(b, m, error_of_given_points(b, m, points)))

    """
        Reference:
        - Went through Mattnedrich efficient implementation of gradient descent in linear regression in python from scratch
        https://github.com/mattnedrich/GradientDescentExample
    """
