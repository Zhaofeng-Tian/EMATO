"""
This file fucos on the THEROM verification of EMATO, give a staight lane:
xs = [ss,vs,as] = [0,0,0]
xe = [se,ve,ae] = [200,0,0]
ts = 0
te = T                           # end time
dv = 20                          # desired velocity (highway)
We try how different is the performance of Polynomial method and EMATO.

"""
import numpy as np
import matplotlib.pyplot as plt
from emato.obj.poly.quintic import QuinticPolynomial, Traj
from emato.util.util import *
from emato.param.param import TruckParam
from emato.alg.emato import EMATO
# from car import Car


se = 200 # end state
dv = 25
dt = 0.1
ktlb = 0.9; ktub = 1.5; assert ktub > ktlb, "coefficient wrong!"
# __init__(self, xs, vxs, axs, xe, vxe, axe, time):
xs = [0,dv,0]; xe = [se, dv, 0]; 
T = np.arange((se/dv)*ktlb,(se/dv)*ktub, dt)
poly_list = []
traj_list = []
r_list = []
rv_list = []
rj_list = []
rfc_list = []
for i in range(len(T)):
    poly = QuinticPolynomial(xs[0],xs[1],xs[2],xe[0],xe[1],xe[2],T[i],dt = 0.1)
    poly_list.append(poly) 

param = TruckParam()
param.gd_profile_type = 'flat'
param.vr = dv
param.dt = dt
param.w1, param.w2, param.w3 = (0.16, 14,38)
profile_altitude, gd_profile = get_gradient_profile(feature=param.gd_profile_type)

fig,a = plt.subplots(2,5)
for i in range(len(poly_list)):
    p = poly_list[i]
    traj = Traj(param,p,gd_profile,dv)
    traj_list.append(traj)
    r_list.append(traj.r)
    rv_list.append(traj.rv)
    rj_list.append(traj.rj)
    rfc_list.append(traj.rfc)
    print("Trajectory time duration: " ,p.Traj_t[-1])
    
    plot_traj(a = a, traj=traj, al=0.1)
    # Energy-Model-Optimal Trajectory
    



r_index = r_list.index(min(r_list))
rv_index = rv_list.index(min(rv_list))
rj_index = rj_list.index(min(rj_list))
rfc_index = rfc_list.index(min(rfc_list))
print("r_list: ", r_list)
print("best_r: ", r_index, "th, ", r_list[r_index])
print("rv_list: ", rv_list)
print("best_rv: ", rv_index, "th, ", rv_list[rv_index])
print("rj_list: ", rj_list)
print("best_rj: ", rj_index, "th, ", rj_list[rj_index])
print("rfc_list: ", rfc_list)
print("best_rfc: ", rfc_index, "th, ", rfc_list[rfc_index])
solver = EMATO(traj_list[rfc_index], param, dv)
otraj = solver.solve()
plot_traj(a, otraj, al = 1, single_color='red', if_single_color= True)
plot_traj(a, traj_list[rfc_index], al = 1, single_color='green', if_single_color= True)
# plot_traj(a, traj_list[rv_index], al = 1, single_color='lightblue', if_single_color= True)
# plot_traj(a, traj_list[rj_index], al = 1, single_color='lightgrey', if_single_color= True)
# plot_traj(a, traj_list[rfc_index], al = 1, single_color='yellow', if_single_color= True)



a[0][0].set_title('s')
a[0][1].set_title('v')
a[0][2].set_title('a')
a[0][3].set_title('j')
a[0][4].set_title('theta')
a[1][0].set_title('at')
a[1][1].set_title('ar')
a[1][2].set_title('ab')
a[1][3].set_title('fr')
a[1][4].set_title('fc')

plt.show()