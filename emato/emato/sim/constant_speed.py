# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
# from emato.param.param import TruckParam

# def calc_at(v, av, theta, k):
#     """ 
#     Calculate acceleration of traction.
#     """
#     k1, k2, k3 = k
#     at = av + k1*v**2 + k2*np.cos(theta) + k3*np.sin(theta)
#     at = max(0., at)
#     return at

# def calc_fuel(v, at, c, o):
#     c0, c1, c2 = c
#     o0, o1, o2, o3, o4 = o
#     fu = o0 + o1*v + o2*v**2 + o3*v**3 + o4*v**4 + (c0 + c1*v + c2*v**2)*at
#     return fu

# param = TruckParam()
# k = param.k
# c = param.c
# o = param.o

# theta_list = np.arange(0, 0.02, 0.01)
# v_list = np.arange(5, 30.1, 0.05)

# fe_map = np.zeros((len(theta_list), len(v_list)))

# av = 0

# for i, theta in enumerate(theta_list):
#     for j, v in enumerate(v_list):
#         at = calc_at(v, av, theta, k)
#         fr = calc_fuel(v, at, c, o)
#         fe = 100*fr / v if v != 0 else 0  # Avoid division by zero
#         fe_map[i, j] = fe



# from scipy.optimize import minimize_scalar
# k1,k2,k3 = param.k
# c0,c1,c2 = param.c
# o0,o1,o2,o3,o4 = param.o
# # Function to minimize
# def fuel_efficiency(v, k1, k2, k3, c0, c1, c2, o0, o1, o2, o3, o4):
#     kg = k2*np.cos(0) + k3*np.sin(0)  # Assuming theta = 0
#     at = k1*v**2 + kg
#     fu = o0 + o1*v + o2*v**2 + o3*v**3 + o4*v**4 + (c0 + c1*v + c2*v**2)*at
#     return fu / v

# # Minimize the fuel efficiency function
# result = minimize_scalar(fuel_efficiency, bounds=(1, 50), args=(k1, k2, k3, c0, c1, c2, o0, o1, o2, o3, o4), method='bounded')

# optimal_v = result.x
# optimal_fuel_efficiency = result.fun

# print(f"Optimal velocity: {optimal_v}")
# print(f"Minimum fuel efficiency: {optimal_fuel_efficiency}")




# # Define the number of levels
# num_levels = 20
# levels = np.linspace(fe_map.min(), fe_map.max(), num_levels)

# # Plotting the fuel efficiency map
# plt.figure(figsize=(10, 6))

# # Filled contour plot
# contour_filled = plt.contourf(v_list, theta_list, fe_map, levels=levels, cmap='RdYlGn_r')

# # Add contour lines
# contour_lines = plt.contour(v_list, theta_list, fe_map, levels=levels, colors='black', linewidths=0.5)

# # Label the contour lines
# clabels = plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.2f')

# # Manually set the font properties for each label
# for label in clabels:
#     label.set_fontproperties(FontProperties(family='Times New Roman', size=8))

# # Add colorbar with ticks at the level values
# cbar = plt.colorbar(contour_filled, ticks=levels)
# cbar.set_label('Fuel Efficiency [L/100km]', fontname='Times New Roman', fontsize=12)

# # Set font to Times New Roman for axis labels and title with specified font size
# plt.xlabel('Velocity (v) [m/s]', fontname='Times New Roman', fontsize=14)
# plt.ylabel('Theta [rad]', fontname='Times New Roman', fontsize=14)
# plt.title('Fuel Efficiency Map w.r.t. Velocity and Theta', fontname='Times New Roman', fontsize=16)

# # Set font for tick labels
# plt.xticks(fontname='Times New Roman', fontsize=12)
# plt.yticks(fontname='Times New Roman', fontsize=12)
# cbar.ax.yaxis.set_tick_params(labelsize=10)
# cbar.ax.set_yticklabels([f'{level:.2f}' for level in levels], fontname='Times New Roman', fontsize=10)

# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.optimize import minimize_scalar
from emato.param.param import TruckParam, CarParam

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
    o0, o1, o2 = o[:3]
    fu = o0 + o1*v + o2*v**2 + (c0 + c1*v + c2*v**2)*at
    return fu

param = CarParam()
k = param.k
c = param.c
o = param.o

theta_list = np.arange(0, 0.02, 0.01)
v_list = np.arange(5, 30.1, 0.05)

fe_map = np.zeros((len(theta_list), len(v_list)))

av = 0

# Compute the fuel efficiency map
for i, theta in enumerate(theta_list):
    for j, v in enumerate(v_list):
        at = calc_at(v, av, theta, k)
        fr = calc_fuel(v, at, c, o) 
        fe = 100 * fr / v if v != 0 else 0  # Convert fuel consumption to L/100km
        fe_map[i, j] = fe

# Function to minimize
def fuel_efficiency(v, theta, k1, k2, k3, c0, c1, c2, o0, o1, o2, o3, o4):
    kg = k2*np.cos(theta) + k3*np.sin(theta)
    at = k1*v**2 + kg
    fu = o0 + o1*v + o2*v**2 + o3*v**3 + o4*v**4 + (c0 + c1*v + c2*v**2)*at
    return fu / v

# Find the optimal velocity for each theta
optimal_v_list = []
for theta in theta_list:
    result = minimize_scalar(fuel_efficiency, bounds=(5, 30), args=(theta, k[0], k[1], k[2], c[0], c[1], c[2], o[0], o[1], o[2], o[3], o[4]), method='bounded')
    optimal_v_list.append(result.x)

# Define the number of levels for the contour plot
num_levels = 20
levels = np.linspace(fe_map.min(), fe_map.max(), num_levels)

# Plotting the fuel efficiency map
plt.figure(figsize=(10, 6))

# Filled contour plot
contour_filled = plt.contourf(v_list, theta_list, fe_map, levels=levels, cmap='RdYlGn_r')

# Add contour lines
contour_lines = plt.contour(v_list, theta_list, fe_map, levels=levels, colors='black', linewidths=0.5)

# Label the contour lines
clabels = plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.2f')

# Manually set the font properties for each label
for label in clabels:
    label.set_fontproperties(FontProperties(family='Times New Roman', size=8))

# Plot the optimal velocity as a red line
plt.plot(optimal_v_list, theta_list, color='red', linewidth=2, label='Optimal Velocity')

# Add colorbar with ticks at the level values
cbar = plt.colorbar(contour_filled, ticks=levels)
cbar.set_label('Fuel Efficiency [L/100km]', fontname='Times New Roman', fontsize=12)

# Set font to Times New Roman for axis labels and title with specified font size
plt.xlabel('Velocity (v) [m/s]', fontname='Times New Roman', fontsize=14)
plt.ylabel('Theta [rad]', fontname='Times New Roman', fontsize=14)
plt.title('Fuel Efficiency Map w.r.t. Velocity and Theta', fontname='Times New Roman', fontsize=16)

# Set font for tick labels
plt.xticks(fontname='Times New Roman', fontsize=12)
plt.yticks(fontname='Times New Roman', fontsize=12)
cbar.ax.yaxis.set_tick_params(labelsize=10)
cbar.ax.set_yticklabels([f'{level:.2f}' for level in levels], fontname='Times New Roman', fontsize=10)

# Add a legend for the optimal velocity line
plt.legend(prop=FontProperties(family='Times New Roman', size=12))

plt.show()
