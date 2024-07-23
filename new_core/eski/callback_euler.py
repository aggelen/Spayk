#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 23:07:11 2024

@author: gelenag
"""

import numpy as np
import matplotlib.pyplot as plt

def euler_integrator(f, y0, t):
    """
    Euler method to integrate a system of ODEs with callback mechanism.
    
    Parameters:
        f : callable
            Function defining the system of differential equations. It should accept
            arguments t (float) and y (array-like), and return dydt (array-like).
        y0 : array-like
            Initial values of y at t[0].
        t : array-like
            Array of time values for which to solve for y.
    
    Returns:
        y : array-like, shape (len(t), len(y0))
            Array of values of y corresponding to each value in t.
    """
    y = np.zeros((len(t), len(y0)))
    y[0] = y0  # Initial values
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        y[i] = y[i-1] + f(t[i-1], y[i-1]) * dt
        
        # Check callback condition (for example, if y[i] crosses a certain value)
        if callback_condition(y[i]):
            # Call callback function and update y if needed
            y[i] = callback_function(t[i], y[i])
    
    return y

def callback_condition(y):
    # Example callback condition: if y crosses a threshold
    threshold = 1.0
    return y[0] > threshold

def callback_function(t, y):
    # Example callback function: reset y to zero if condition is met
    y[:] = 0.0
    return y

# Example system of differential equations
def system_of_equations(t, y):
    dydt = np.zeros_like(y)
    dydt[0] = y[1]
    dydt[1] = -y[0]
    return dydt

# Initial conditions and time span
y0 = np.array([1.0, 0.0])  # Initial values of y
t = np.linspace(0, 10, 100)  # Time points for solution

# Solve using Euler method with callback
y_euler = euler_integrator(system_of_equations, y0, t)

# Plotting the solution
plt.figure(figsize=(10, 6))
plt.plot(t, y_euler[:, 0], label='y1(t)')
plt.plot(t, y_euler[:, 1], label='y2(t)')
plt.title('Solution of System of Differential Equations using Euler Method with Callback')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.show()