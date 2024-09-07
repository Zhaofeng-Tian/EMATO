from emato.util.util import get_gradient_profile

import matplotlib.pyplot as plt
import numpy as np
from emato.param.param import TruckParam

param = TruckParam()
param.gd_profile_type = "rolling"



profile_altitude, profile_gradient = get_gradient_profile(feature=param.gd_profile_type)
elevation_hist = profile_altitude
label_font = {'family': 'Times New Roman', 'size': 10}
fig, ax2 = plt.subplots(figsize=(30, 18), )

traveled_s_hist = np.arange(0,len(elevation_hist), 1) 
ax2.fill_between(traveled_s_hist, elevation_hist, color = 'lightgray', edgecolor='black')
# ax2.set_ylabel('Elevation [m]')
# ax2.set_xlabel('Traveled S [m]')
# ax2.set_ylim([90,max(elevation_hist)])
# ax2.set_xlim([0, max(100, traveled_s)])
# Customize ax2 similar to ax
ax2.set_ylabel('Elevation [m]', fontdict=label_font)
ax2.set_xlabel('Traveled S [m]', fontdict=label_font)
ax2.set_ylim([min(elevation_hist)-10, max(elevation_hist)+10])
# ax2.set_xlim([0, max(100, traveled_s)])

ax2.tick_params(axis='both', which='major', labelsize=10)
for tick in ax2.get_xticklabels():
    tick.set_fontname("Times New Roman")
for tick in ax2.get_yticklabels():
    tick.set_fontname("Times New Roman")

# Set figure-level title (for both subplots)
title_font = {'family': 'Times New Roman', 'size': 50}

plt.show()