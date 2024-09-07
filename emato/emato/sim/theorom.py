import numpy as np
import matplotlib.pyplot as plt
from emato.obj.poly.quintic import QuinticPolynomial
from emato.util.util import *
from emato.obj.poly.quintic import Traj
from emato.param.param import CarParam, TruckParam
from emato.obj.car.car import Car
from e import EMATO, EMATO_R

dv = 20
dt = 0.1
param = TruckParam()
param.gd_profile_type = 'flat'
param.vr = dv
param.dt = dt
# param.w1, param.w2, param.w3 = (0.0001, 0.0001, 35)
param.w1, param.w2, param.w3 = (0.0001, 0.0001, 1)
param.jmax = 5
profile_altitude, gd_profile = get_gradient_profile(feature=param.gd_profile_type)
fig, ax = plt.subplots(2, 2)
total_s = 1000
fc_list = []
# for ds in [80, 100, 150, 200,500]:
for ds in [1000]:
    n = int(total_s/ds)
    # Clear each subplot to start fresh for each ds value
    for row in ax:
        for subplot in row:
            subplot.clear()

    temp_T = 0
    temp_fc = 0
    temp_cc_fc = 0
    ses = [(i + 1) * ds for i in range(n)]
    for se in ses:
        ktlb = 0.9
        ktub = 1.5
        assert ktub > ktlb, "coefficient wrong!"
        xs = [se - ds, dv, 0]
        xe = [se, dv, 0]
        T = np.arange((ds / dv) * ktlb, (ds / dv) * ktub, 0.1)
        poly_list = []
        traj_list = []
        r_list = []
        rv_list = []
        rj_list = []
        rfc_list = []

        for i in range(len(T)):
            poly = QuinticPolynomial(xs[0], xs[1], xs[2], xe[0], xe[1], xe[2], T[i],0.1)
            poly_list.append(poly)

        for i in range(len(poly_list)):
            p = poly_list[i]
            # p.Traj_t = p.Traj_t + temp_T
            traj = Traj(param, p, gd_profile, dv)
            traj.Traj_t += temp_T
            traj.Traj_fc += temp_fc
            if max(traj.Traj_jerk**2) < param.jerkmax**2:
                traj_list.append(traj)
                r_list.append(traj.r)
                rv_list.append(traj.rv)
                rj_list.append(traj.rj)
                rfc_list.append(traj.rfc)
                print(" rfc_list: ", rfc_list)
            # Plot the trajectory and pause to create animation effect
            # plot_traj(a=ax, traj=traj, al=0.6)
            # plt.pause(0.5)  # Pause to display each plot step
            if max(traj.Traj_jerk**2) > param.jerkmax**2: 
                plot_traj2(a=ax, traj=traj, al=1, single_color='lightgray')
            else:
                plot_traj2(a=ax, traj=traj, al=0.6, single_color='lightgreen')
        r_index = r_list.index(min(r_list))
        rv_index = rv_list.index(min(rv_list))
        rj_index = rj_list.index(min(rj_list))
        rfc_index = rfc_list.index(min(rfc_list))

        # plt.pause(0.5)  # Pause to display the plot

        fc_traj = traj_list[rfc_index]
        plot_traj2(a=ax, traj=fc_traj, al=0.6, single_color='green')
        

        # fc_traj.Traj_t += temp_T
        # fc_traj.Traj_fc += temp_fc
        temp_T = fc_traj.Traj_t[-1]
        temp_fc = fc_traj.Traj_fc[-1]

        # # solver = EMATO_R(cc_traj, param, dv)
        # plot_traj(ax, traj_list[rfc_index], al=1, single_color='green', if_single_color=True)
        # solver = EMATO_R(traj_list[rfc_index], param, dv)
        # solver = EMATO_R(cc_traj, param, dv)
        # png_traj = solver.solve()
        # png_traj.Traj_t += temp_T
        # png_traj.Traj_fc += temp_fc
        # plot_traj2(ax, png_traj, al=1, single_color='red', if_single_color=True)


        # temp_T = png_traj.Traj_t[-1]
        # temp_fc = png_traj.Traj_fc[-1]
        # temp_cc_fc = cc_traj.Traj_fc[-1]
        # temp_T = cc_traj.Traj_t[-1]
        # temp_fc = cc_traj.Traj_fc[-1]
        # plt.pause(1)  # Pause to display the final plot in this loop

        # ax[0][0].set_title('s')
        # ax[0][1].set_title('v')
        # ax[0][2].set_title('a')
        # ax[0][3].set_title('j')
        # ax[0][4].set_title('theta')
        # ax[1][0].set_title('at')
        # ax[1][1].set_title('ar')
        # ax[1][2].set_title('ab')
        # ax[1][3].set_title('fr')
        # ax[1][4].set_title('fc')
        ax[0][0].set_title('S')
        ax[0][1].set_title('V')
        ax[1][0].set_title('Jerk')
        ax[1][1].set_title('Fuel Consumption')
        # plt.pause(1)
       
    fc_list.append(temp_fc)


    """
    CC
    """
    poly = QuinticPolynomial(0, dv, 0, total_s, dv, 0, total_s / dv,0.1)
    # poly.Traj_t = poly.Traj_t
    cc_traj = Traj(param, poly, gd_profile, dv)
    # cc_traj.Traj_fc += temp_cc_fc
    plot_traj2(a=ax, traj=cc_traj, al=1, single_color='orange')

    plt.show()
    print("********** fuel consumption! ***********" ,fc_list)

# ********** fuel consumption! *********** [264.55513728527893, 267.64427581707446, 267.0818786612389, 266.6558977916516, 266.07387854998893]
# ********** fuel consumption! *********** [270.9204883495864, 272.91744526616804, 270.69860424774384, 269.58918373853174, 267.59222682195]  
# ********** fuel consumption! *********** [267.73219537988257, 267.64427581707446, 267.0818786612386, 266.6558977916522, 261.9672482990764]