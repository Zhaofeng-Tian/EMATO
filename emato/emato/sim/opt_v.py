import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from emato.param.param import TruckParam

def calc_at(v, av, theta, k):
    """ 
    Calculate acceleration of traction.
    """
    k1, k2, k3 = k
    at = av + k1*v**2 + k2*np.cos(theta) + k3*np.sin(theta)
    at = max(0., at)
    return at

def calc_fuel(v, at, c, o):
    c0, c1, c2 = c
    o0, o1, o2, o3, o4 = o
    fu = o0 + o1*v + o2*v**2 + o3*v**3 + o4*v**4 + (c0 + c1*v + c2*v**2)*at
    return fu

param = TruckParam()
v = 27
theta = 0
k1,k2,k3 = param.k
c0,c1,c2 = param.c
o0,o1,o2,o3,o4 = param.o

kg = k2*np.cos(theta) + k3*np.sin(theta)

at = k1*v**2 + kg

A = np.array([[o0, o1, o2, o3, o4],
             [c0*kg, c1*kg, c0*k1+c2*kg,c1*k1,c2*k1]])
X = np.array([1,v,v**2,v**3,v**4])

Y = A@X
print(Y.sum())

at = calc_at(v, 0, theta, param.k)
fu = calc_fuel(v,at,param.c,param.o)

print(fu)

from scipy.optimize import minimize_scalar

# Function to minimize
def fuel_efficiency(v, k1, k2, k3, c0, c1, c2, o0, o1, o2, o3, o4):
    kg = k2*np.cos(0) + k3*np.sin(0)  # Assuming theta = 0
    at = k1*v**2 + kg
    fu = o0 + o1*v + o2*v**2 + o3*v**3 + o4*v**4 + (c0 + c1*v + c2*v**2)*at
    return fu / v

# Minimize the fuel efficiency function
result = minimize_scalar(fuel_efficiency, bounds=(1, 50), args=(k1, k2, k3, c0, c1, c2, o0, o1, o2, o3, o4), method='bounded')

optimal_v = result.x
optimal_fuel_efficiency = result.fun

print(f"Optimal velocity: {optimal_v}")
print(f"Minimum fuel efficiency: {optimal_fuel_efficiency}")