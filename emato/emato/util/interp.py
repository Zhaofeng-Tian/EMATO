import numpy as np
from scipy.interpolate import interp1d
import time
import matplotlib.pyplot as plt

# Original trajectory data
original_l = np.array([
    0.0, 2.43562078, 4.86234584, 7.30597778, 9.71425379,
    12.1133488, 14.50325609, 16.88400588, 19.25566144, 21.61831537,
    23.97208605, 26.31711435, 28.65356061, 30.98160183, 33.27879798,
    35.59092695, 37.89526055, 40.19201957, 42.46007458, 44.74269481,
    47.01843886, 49.28754558, 51.53020705, 53.78709554, 56.03807245,
    58.26432516, 60.50453796, 62.73955427, 64.96960422, 67.17721668,
    69.39834822, 71.61516304, 73.82785633, 76.0366091, 78.24158548,
    80.44292965, 82.64076256, 84.83517824, 87.02623999, 89.21397624,
    91.39837623, 93.59278793, 95.76993469, 97.94343153, 100.11306584,
    102.29046158, 104.45110711, 106.61799138, 108.76800022, 110.91207905,
    113.05955913
])  
original_x = np.array([
    115.09751965, 116.2107276, 117.33096168, 118.46997755, 119.60463944,
    120.74663166, 121.89589808, 123.05231167, 124.21567941, 125.38574721,
    126.56220468, 127.74469001, 128.93279481, 130.12606881, 131.31362576,
    132.51586219, 133.72171563, 134.93061586, 136.13204727, 137.34538783,
    138.55997518, 139.77519348, 140.98100076, 142.19577081, 143.40934361,
    144.6120939, 145.82165951, 147.02831817, 148.23154587, 149.42235966,
    150.61742249, 151.80768159, 152.99275831, 154.17232036, 155.34608563,
    156.51382598, 157.67537099, 158.83061153, 159.97950337, 161.12207058,
    162.25840893, 163.39536246, 164.51967408, 165.63850527, 166.75227021,
    167.86748004, 168.97254076, 170.0799689, 171.17905394, 172.27649725,
    173.3784031
])  
original_y = np.array([
    257.8372665, 260.00360365, 262.1562927, 264.31823161, 266.44245754,
    268.55231787, 270.64775184, 272.72877975, 274.79549831, 276.84807595,
    278.88674812, 280.9118126, 282.92362476, 284.92259276, 286.88901556,
    288.86400112, 290.82763826, 292.78049826, 294.70420201, 296.63763414,
    298.56215708, 300.47842685, 302.36934177, 304.27141417, 306.16723538,
    308.04062612, 309.92623065, 311.80752616, 313.68512194, 315.54402357,
    317.4162549, 319.28642902, 321.15501263, 323.02242318, 324.88902328,
    326.7551151, 328.62093467, 330.48664619, 332.35233618, 334.21800765,
    336.08357419, 337.96048105, 339.82485531, 341.68826758, 343.550213,
    345.42033577, 347.27700751, 349.13953008, 350.98738022, 352.82930643,
    354.67253029
])  
original_yaw = np.array([
    1.0961172, 1.09097147, 1.08590076, 1.08020098, 1.07468466, 1.0691345,
    1.06359246, 1.0580987, 1.05269156, 1.04740744, 1.0422807, 1.0373436,
    1.03262616, 1.02750136, 1.02398114, 1.0200796, 1.01649368, 1.01252732,
    1.01035975, 1.00781474, 1.00563839, 1.00312747, 1.00243714, 1.00139063,
    1.00005294, 1.0004384, 1.00049219, 1.00089152, 1.00105706, 1.00268342,
    1.00401067, 1.0056006, 1.00742461, 1.009451, 1.01164501, 1.01396885,
    1.01638178, 1.01884013, 1.02129734, 1.02370399, 1.02615291, 1.02813691,
    1.03006759, 1.0317215, 1.03307992, 1.03392576, 1.03436789, 1.03421562,
    1.0334613, 1.0319854, 1.0319854
])  
original_s = np.array([
    206.63670385, 208.56567561, 210.49079406, 212.41192797, 214.32899516,
    216.2419599, 218.15083038, 220.0556561, 221.95652532, 223.85356246,
    225.74692557, 227.63680373, 229.52341447, 231.40700121, 233.28783071,
    235.16619044, 237.04238607, 238.91673886, 240.78958308, 242.66126348,
    244.53213268, 246.40254861, 248.27287193, 250.14346348, 252.01468167,
    253.88687996, 255.76040422, 257.63559023, 259.51276104, 261.39222445,
    263.2742704, 265.15916844, 267.04716511, 268.93848139, 270.83331014,
    272.7318135, 274.63412034, 276.54032368, 278.45047811, 280.36459722,
    282.28265106, 284.2045635, 286.13020972, 288.05941361, 289.9919452,
    291.9275181, 293.8657869, 295.80634462, 297.74872013, 299.69237559,
    301.63670385
])  
original_d = np.array([
    -0.07411815, -0.11235094, -0.16006059, -0.21754829, -0.28499725, -0.36247897,
    -0.44995941, -0.5473052, -0.65428991, -0.7706002, -0.89584207, -1.02954707,
    -1.1711785, -1.32013765, -1.47576999, -1.6373714, -1.80419437, -1.97545423,
    -2.15033535, -2.32799738, -2.50758142, -2.6882163, -2.8690247, -3.04912947,
    -3.22765977, -3.40375732, -3.57658259, -3.74532104, -3.90918931, -4.06744146,
    -4.21937516, -4.36433793, -4.50173333, -4.63102717, -4.75175377, -4.86352212,
    -4.96602214, -5.05903083, -5.14241859, -5.2161553, -5.28031667, -5.33509035,
    -5.3807822, -5.41782248, -5.44677209, -5.46832875, -5.48333325, -5.49277562,
    -5.4978014, -5.49971782, -5.5
])

# New reallocated trajectory
new_l = np.array([
    0.0, 2.43117292, 4.8555512, 7.27773592, 9.70272705,
    12.13052461, 14.5561286, 16.97517568, 19.38578814, 21.7879373,
    24.18159468, 26.56673203, 28.94332125, 31.31133449, 33.67075282,
    36.02155749, 38.36372111, 40.6972165, 43.02201669, 45.33809491,
    47.64542461, 49.94397943, 52.2337416, 54.51469348, 56.78680922,
    59.0500632, 61.30442999, 63.54988437, 65.78640136, 68.01395616,
    70.23253216, 72.4421129, 74.64267411, 76.83419175, 79.01664198,
    81.1900012, 83.35424602, 85.50935326, 87.65529998, 89.79207091,
    91.91965093, 94.03801763, 96.14714883, 98.24795397, 100.34385354,
    102.43984755, 104.54093599, 106.65211887, 108.77709896, 110.91457905,
    113.05955913
])  

# Create interpolation functions
interp_x = interp1d(original_l, original_x, bounds_error=False, fill_value='extrapolate')
interp_y = interp1d(original_l, original_y, bounds_error=False, fill_value='extrapolate')
interp_yaw = interp1d(original_l, original_yaw, bounds_error=False, fill_value='extrapolate')
interp_s = interp1d(original_l, original_s, bounds_error=False, fill_value='extrapolate')
interp_d = interp1d(original_l, original_d, bounds_error=False, fill_value='extrapolate')

# Measure the time taken for interpolation
start_time = time.time()

# Get new trajectory points
new_x = interp_x(new_l)
new_y = interp_y(new_l)
new_yaw = interp_yaw(new_l)
new_s = interp_s(new_l)
new_d = interp_d(new_l)

dt = 0.1

# First derivatives
new_s_d = np.gradient(new_s, dt)  # Derivative of s with respect to time
new_d_d = np.gradient(new_d, dt)  # Derivative of d with respect to time
new_s_dd = np.gradient(new_s_d, dt)
new_d_dd = np.gradient(new_d_d, dt)
# Output results
print("new_s_d (first derivative of s with respect to time):", new_s_d)
print("new_d_d (first derivative of d with respect to time):", new_d_d)
print("new_s_d (first derivative of s with respect to time):", new_s_dd)
print("new_d_d (first derivative of d with respect to time):", new_d_dd)


# # First derivatives
# new_s_d = np.gradient(new_s, new_l)  # Derivative of s with respect to l
# new_d_d = np.gradient(new_d, new_l)  # Derivative of d with respect to l

# # Second derivatives
# new_s_dd = np.gradient(new_s_d, new_l)  # Derivative of s_d with respect to l
# new_d_dd = np.gradient(new_d_d, new_l)  # Derivative of d_d with respect to l

# Output results




end_time = time.time()

# Output new trajectory coordinates and time taken
print("New X:", new_x)
print("New Y:", new_y)
print("New Yaw:", new_yaw)
print("New S:", new_s)
print("New D:", new_d)
# print("new_s_d (first derivative of s):", new_s_d)
# print("new_d_d (first derivative of d):", new_d_d)
# print("new_s_dd (second derivative of s):", new_s_dd)
# print("new_d_dd (second derivative of d):", new_d_dd)
print("Time taken for interpolation: {:.6f} seconds".format(end_time - start_time))

print(new_s - original_s)
# Plotting the trajectories
plt.figure(figsize=(12, 8))

# Plot original trajectory
plt.subplot(2, 2, 1)
plt.plot(original_x, original_y, 'ko-', label='Original Trajectory')
plt.plot(original_x, original_y, 'k.', markersize=5)  # Dots for original trajector
plt.plot(new_x, new_y, 'ro-', label='New Trajectory')
plt.plot(new_x, new_y, 'r.', markersize=5)  # Dots for new trajectory
plt.title('Original Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.legend()

# Plot new trajectory
plt.subplot(2, 2, 2)
plt.plot(new_x, new_y, 'ro-', label='New Trajectory')
plt.plot(new_x, new_y, 'r.', markersize=5)  # Dots for new trajectory
plt.title('New Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.legend()

# Plot yaw
plt.subplot(2, 2, 3)
plt.plot(original_l, original_yaw, 'ko-', label='Original Yaw')
plt.plot(original_l, original_yaw, 'k.', markersize=5)  # Dots for original yaw
plt.plot(new_l, new_yaw, 'ro-', label='New Yaw')
plt.plot(new_l, new_yaw, 'r.', markersize=5)  # Dots for new yaw
plt.title('Yaw Comparison')
plt.xlabel('Length')
plt.ylabel('Yaw')
plt.grid(True)
plt.legend()

# Plot s and d
plt.subplot(2, 2, 4)
plt.plot(original_l, original_s, 'ko-', label='Original S')
plt.plot(original_l, original_s, 'k.', markersize=5)  # Dots for original s
plt.plot(new_l, new_s, 'ro-', label='New S')
plt.plot(new_l, new_s, 'r.', markersize=5)  # Dots for new s
plt.title('S Comparison')
plt.xlabel('Length')
plt.ylabel('S')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

