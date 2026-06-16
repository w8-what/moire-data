# Possible functions for curve fitting
import math

# ALL POSSIBLE CASES RIGHT NOW
# y < SC_Threshold
# y = log(x)
# y = x
# y = x^2
# dy/dx goes from neg -> pos
# dy/dx is neg (?) skip for now

# def sublinear(x, a, b):
#     return a*math.log(x) + b

def linear(x, a, b):
    return a*x + b

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c


