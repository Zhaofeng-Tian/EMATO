import numpy as np
import matplotlib.pyplot as plt

def get_gradient_profile(feature='steep', plot=False):
    if feature == 'normal':
        dslope1, dslope2, dslope3 = 0.04, 0.02, 0.00
        len_wave1, len_wave2, len_wave3 = 2870, 2136, 1976
    elif feature == 'steep':
        dslope1, dslope2, dslope3 = 0.05, 0.02, 0.01
        len_wave1, len_wave2, len_wave3 = 2380, 1860, 1430
    else:
        dslope1, dslope2, dslope3 = 0.05, 0.02, 0.00
        len_wave1, len_wave2, len_wave3 = 1870, 1136, 976

    initial_altitude = 100
    len_road = 20000
    n_points = int(len_road)
    x = np.linspace(0, len_road, n_points)

    slope = dslope1 * np.sin(2 * np.pi * x / len_wave1) + dslope2 * np.sin(2 * np.pi * x / len_wave2) \
            + dslope3 * np.sin(2 * np.pi * x / len_wave3)
    if feature == 'steep':
        slope += 0.02
    elif feature == 'flat':
        slope = np.zeros(len(slope))

    tan_slope = np.tan(slope)
    altitude = np.cumsum(tan_slope) + initial_altitude

    return altitude, slope, x

# Generate altitude profiles for "flat", "normal", and "steep"
altitude_flat, slope_flat, x = get_gradient_profile(feature='flat')
altitude_normal, slope_normal, _ = get_gradient_profile(feature='normal')
altitude_steep, slope_steep, _ = get_gradient_profile(feature='steep')

# Plot elevation profiles with fill in between
plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 25})

# Plot flat profile
plt.plot(x, altitude_flat, label='Flat', color='black')
plt.fill_between(x, altitude_flat, color='green', alpha=0.3)

# Plot normal profile
plt.plot(x, altitude_normal, label='Rolling', color='blue')
plt.fill_between(x, altitude_normal, color='blue', alpha=0.3)

# Plot steep profile
plt.plot(x, altitude_steep, label='Steep', color='red')
plt.fill_between(x, altitude_steep, color='red', alpha=0.3)

# Add titles and labels
# plt.title('Elevation Profile', fontsize=18)
plt.xlabel('Distance [m]', fontsize=25)
plt.ylabel('Elevation [m]', fontsize=25)

# Set y-limits
# plt.ylim([90, np.max(altitude_steep) + 10])
plt.ylim([50, 350])
plt.xlim([0, 20000])
# Add grid, legend, and show plot
# plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
