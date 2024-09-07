import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the saved Recorder object from the file
file_path = '/home/tian/emato/emato/data/acc/truck_HWFET_flat_NLP_1.16_14.51_38.91_gd_True_ptime_5.pkl'
with open(file_path, 'rb') as file:
    recorder = pickle.load(file)['recorder']

# Extract the time (Tt), leading car velocity (Tvl), and ego car velocity (Tve)
Tt = np.array(recorder.Tt)
Tvl = np.array(recorder.Tvl)
Tve = np.array(recorder.Tve)

# Plot the velocities of the leading car and ego car over time
plt.figure(figsize=(10, 6))

plt.plot(Tt, Tvl, label='Leading Car Velocity', linewidth=2)
plt.plot(Tt, Tve, label='Ego Car Velocity', linewidth=2)

# Setting Times New Roman font for labels and title
plt.title('Vehicle Velocities over Time', fontname='Times New Roman', fontsize=16)
plt.xlabel('Time [s]', fontname='Times New Roman', fontsize=14)
plt.ylabel('Velocity [m/s]', fontname='Times New Roman', fontsize=14)

# Setting Times New Roman font for ticks
plt.xticks(fontname='Times New Roman', fontsize=12)
plt.yticks(fontname='Times New Roman', fontsize=12)

plt.legend(loc='best', fontsize=12)
plt.grid(True)
plt.tight_layout()

plt.show()