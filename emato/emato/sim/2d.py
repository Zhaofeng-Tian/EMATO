
from time import time
import numpy as np
from emato.alg.quintic2d import Quintic_2d
from emato.param.param import CarParam, TruckParam
# from emato.param.uniparam import Param
from emato.obj.road.road import Road
from emato.obj.car.car import Car, Car_J
from emato.obj.car.traffic_car import TrafficCar
from emato.util.util import get_leading_profile, local_approx, get_gradient_profile, get_traj_theta
from emato.alg.nlp import NLP, NLP_J
from emato.alg.bnlp import BNLP, BNLP_R
from emato.alg.quintic import Quintic_1d, Quintic_1d_R
from emato.util.recorder import Recorder
from emato.util.plot import plot_cars, plot_traffic_traj
from emato.alg.emato import EMATO, EMATO_R
from scipy.interpolate import interp1d
import pickle

import matplotlib.pyplot as plt

def frenet_sim(if_plot, car_type, g_type, r_type, if_use_frenet_policy):

    v_list = []
    oft_v_list = []
    oft_s_d_list = []
    sim_time = 0
    total_time = 138
    time_step = 0.1
    recorder = Recorder()
    if_plot = if_plot
    if_use_frenet_policy = if_use_frenet_policy
    # param = TruckParam(cycle='HWFET')
    # param = Param(car_type = car_type)
    if car_type == 'truck':
        param = TruckParam()
    elif car_type == 'car':
        param = CarParam()
    param.gd_profile_type = g_type
    param.prediction_time = 5
    param.N = round(param.prediction_time/param.dt)+1
    # param.rsafe = 4
    param.safe_list = {'rsafe': 5, 'lon_safe': 5, 'lat_safe': 2.3}
    param.desired_v = 70/3.6
    param.desired_d = 0
    # sim_time = param.ts
    global_time = 0
    check_lb = -50
    check_ub = 100
    poft = None

    """
    Road initialization
    """
    wx = [0.0, 350.0, 700.0, 1300.0, 1700.0, 2500,2900]
    wy = [0.0, 500.0, 150.0, 65.0, 0.0,-500,0]
    # wy = [0,0,0,0,0]
    road = Road(wx, wy, num_lanes=3)
    # Road length 2116.5
    # assert 1==2 ,"sdf"
    profile_altitude, profile_gradient = get_gradient_profile(feature=param.gd_profile_type)
    bs = 100 # base s
    """
    Simulated traffic cars 
    """
    # tcar1 = TrafficCar(lane=0, sd= [bs+30,0], speed=55/3.6, road = road)
    # tcar2 = TrafficCar(lane=0, sd = [bs-20,0], speed=55/3.6 ,road = road)
    # tcar3 = TrafficCar(lane=1, sd = [bs+80,0], speed=50/3.6,road = road)
    # tcar4 = TrafficCar(lane=1, sd = [bs+120,0], speed=50/3.6 ,road = road)
    
    # tcar5 = TrafficCar(lane=2, sd = [bs+0,0], speed=50/3.6 ,road = road)
    # tcar6 = TrafficCar(lane=2, sd = [bs+80,0],speed=52/3.6,road = road)
    # tcar7 = TrafficCar(lane=2, sd = [bs+120,0],speed=53/3.6 ,road = road)
    # tcar8 = TrafficCar(lane=1, sd = [bs+150,0], speed=50/3.6 ,road = road)
    # tcar9 = TrafficCar(lane=1, sd = [bs+180,0], speed=50/3.6 ,road = road)
    # tcar10 = TrafficCar(lane=2, sd = [bs+130,0],speed=67/3.6 ,road = road)
    # tcar11 = TrafficCar(lane=1, sd = [bs+210,0], speed=55/3.6 ,road = road)
    # tcar12 = TrafficCar(lane=0, sd = [bs+170,0], speed=50/3.6 ,road = road)
    # tcar13 = TrafficCar(lane=0, sd = [bs+200,0], speed=53/3.6 ,road = road)
    
    # tcars = [tcar1, tcar2, tcar3, tcar4, tcar5, tcar6, \
    #         tcar7, tcar8, tcar9, tcar10, tcar11,tcar12,tcar13]
    tcars = []
    for bs in np.arange(0, 1800, 70):
        tcars.append(TrafficCar(lane=0, sd = [bs, 0], speed = 50/3.6, road = road))
    for bs in np.arange(30, 1800, 90):
        tcars.append(TrafficCar(lane=1, sd = [bs, 0], speed = 56/3.6, road = road))
    for bs in np.arange(40, 1800, 70):
        tcars.append(TrafficCar(lane=2, sd = [bs, 0], speed = 60/3.6, road = road))

    """"
    Ego car initialization
    """
    # Dynmical car
    start_s = 100
    # car_e's s coordinate is the path arc coordinate
    # Rendering car
    tcar_e = TrafficCar(lane=1, sd = [start_s,0],speed = 12.0, road = road)
    es,ed = tcar_e.s, tcar_e.d
    xc = [es, 70/3.6, 0, ed, 0, 0]

    """
    Traffic set up
    """
    # Traffic trajectory setup, define which cars are cared by distance
    
    s_list = []
    d_list = []
    leading_car = None
    cared_index_list = []
    traffic_traj_sd_list = []
    leading_dist = np.inf
    for i in range(len(tcars)):
        
        if tcars[i].s - es > check_lb and tcars[i].s - es < check_ub:
            cared_index_list.append(i)
            tcars[i].calc_traffic_future_sd_points(param.prediction_time+param.dt,param.dt)
            print("***** len: ", len(tcars[i].get_future_sd_points()))
            traffic_traj_sd_list.append(tcars[i].get_future_sd_points())
            tcars[i].set_xyyaw_with_sd()
            if abs(tcars[i].d) - ed < 0.6 :
                if tcars[i].s - es < leading_dist and tcars[i].s - es > 5:
                    leading_car = tcars[i]
                    leading_dist = tcars[i].s - es

        
    """
    Solver setup
    """
    emato = EMATO(param)
    solver = Quintic_2d(param, road, traffic_traj_sd_list, \
                        leading_car,\
                        xc,desired_v=param.desired_v, desired_d = param.desired_d)

    # fig, (ax, ax2) = plt.subplots(2,1,figsize=(30, 18), )
    fig = plt.figure(figsize=(10,4))
    # ,width_ratios = [1,0.7],height_ratios = [1,1] 
    gs = fig.add_gridspec(1,2 )
    ax = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])

    window_width = 10
    

    # Data recording set up
    invalid_count = 0
    traveled_l = 0
    traveled_s = 0
    traveled_s_hist = []
    traveled_l_hist = []
    elevation_hist = []
    total_fc = 0
    oft_jerks = []
    emato_jerks = []

    average_v = 0 # along arcwarts **************************************
    feasible_candidates = []
    invalid_candidates = []
    acc_feasibles = []; acc_invalids = []
    new_x = None
    new_y = None
    new_yaw = None
    new_s = None
    new_d = None
    new_s_d = None  # Derivative of s with respect to time
    new_d_d = None  # Derivative of d with respect to time
    new_s_dd = None
    new_d_dd = None
    # Simulation loop
    # for i in range(0, round(total_time/param.dt)):
    while traveled_s < 2200:
        traveled_s_hist.append(traveled_s)
        traveled_l_hist.append(traveled_l)
        elevation_hist.append(profile_altitude[int(tcar_e.s)])
        ax.clear()
        ax2.clear()
        road.plot_road(ax)
        # ***** Update 
        solver.update(xc, traffic_traj_sd_list, leading_car)
        # Solve
        feasible_candidates, invalid_candidates, acc_feasibles, acc_invalids = solver.solve()
        # ********************
        lenf = len(feasible_candidates); len_in = len(invalid_candidates)
        print(" Total candidates {}, Feasible {}, invalid {} ".format(lenf+len_in, lenf, len_in))
        print("how many feasible?" ,len(feasible_candidates))
        # print("r_complex1s list",solver.r_complex1s)
        # print("solver.r_complex2s, ",solver.r_complex2s)
        if lenf is not 0:
            invalid_count = 0
            oft1_index = np.argmin(solver.r_complex1s)
            oft2_index = np.argmin(solver.r_complex2s)
            oft3_index = np.argmin(solver.r_complex3s)
            print("which is optimal?" ,oft2_index)
            oft1 = solver.feasible_candidates[oft1_index]
            oft2 = solver.feasible_candidates[oft2_index]
            oft3 = solver.feasible_candidates[oft3_index]
            if r_type == 1:
                oft = solver.feasible_candidates[oft1_index]
            elif r_type == 2:
                oft = solver.feasible_candidates[oft2_index]
            elif r_type == 3:
                oft = solver.feasible_candidates[oft3_index]
            poft = oft
            oft_v_list.append(oft.v.copy())
            oft_s_d_list.append(oft.s_d.copy())

            if not if_use_frenet_policy:
                
                emato.update(oft)
                res = emato.solve()
                # print("res[:,N]: ", res[:param.N])
                # print("res[N,2N]: ",res[param.N : 2*param.N])
                l = res[: param.N]
                dl = np.diff(l); dl = np.append(dl, dl[-1])
                assert len(l) == len(dl), "l and dl not same len"
                v = res[param.N : param.N *2]
                v_list.append(v)
                at = res[param.N*2: param.N *3]
                o = param.o; c = param.c 
                fr = o[0] + o[1]*v + o[2]*v**2+o[3]* v**3 + o[4]*v**2+(c[0]+c[1]*v + c[2]*v**2)*at
                fc = np.cumsum(fr * param.dt)
                jerk = res[param.N*3: param.N*4]
                # Create interpolation functions
                interp_x = interp1d(oft.l, oft.x, bounds_error=False, fill_value='extrapolate')
                interp_y = interp1d(oft.l, oft.y, bounds_error=False, fill_value='extrapolate')
                interp_yaw = interp1d(oft.l, oft.yaw, bounds_error=False, fill_value='extrapolate')
                interp_s = interp1d(oft.l, oft.s, bounds_error=False, fill_value='extrapolate')
                interp_d = interp1d(oft.l, oft.d, bounds_error=False, fill_value='extrapolate')

                # Get new trajectory points
                new_x = interp_x(l)
                new_y = interp_y(l)
                new_yaw = interp_yaw(l)
                new_s = interp_s(l)
                new_d = interp_d(l)
                new_s_d = np.gradient(new_s, param.dt)  # Derivative of s with respect to time
                new_d_d = np.gradient(new_d, param.dt)  # Derivative of d with respect to time
                new_s_dd = np.gradient(new_s_d, param.dt)
                new_d_dd = np.gradient(new_d_d, param.dt)
                new_s_ddd = np.gradient(new_d_dd, param.dt)
                new_d_ddd = np.gradient(new_d_dd,param.dt)
                print("new_s_d: ", new_s_d)
                print("new_s_ddd:", new_s_ddd)
                # assert 1==2 , "sdf"
                try:
                    assert fc[-1] < oft.fc[-1], "getting worse fc!!"
                except AssertionError as e: 
                    print("keep going.", e)
                print('***************************ofc: {}, oft: {} **************'.format(fc[-1], oft.fc[-1]))
        else:
            invalid_count +=1

        """
        ******************* Plot here ***********************
        """
        if if_plot:
        # if if_plot and (sim_time > 67.0 and sim_time < 73.5):
            # for ft in invalid_candidates:
            #     if ft.if_xy_collision or ft.if_sd_collision:
            #         # print("check in 2d.py ", ft.if_xy_collision)
            #         plt.plot(ft.x,ft.y, 'red', alpha = 0.2)
            #     elif ft.if_over_curvy:
            #         plt.plot(ft.x,ft.y, 'orange', alpha = 0.5)
            #     #     print(max(ft.cur))
            #     elif ft.if_over_jerky:
            #         plt.plot(ft.x,ft.y, 'yellow', alpha = 0.5)

            for ft in feasible_candidates:
                ax.plot(ft.x, ft.y, 'yellow', alpha = 0.6)

            if len(acc_feasibles) > 0:
                ax.plot(acc_feasibles[0].x, acc_feasibles[0].y, 'blue', alpha = 1)
            # Select an traj with an optimal policy


            ax.plot(oft1.x, oft1.y, 'red', alpha = 1)
            ax.plot(oft2.x, oft2.y, 'orange', alpha = 1)
            ax.plot(oft3.x, oft3.y, 'green',alpha = 1)
            if not if_use_frenet_policy:
                ax.plot(new_x, new_y, 'green')


            # plot_traffic_traj(ax,tcars)
            plot_cars(ax,tcars,if_plot_future=True)
            plot_cars(ax,[tcar_e],if_plot_future=False, color = 'lightgreen')
            if leading_car is not None:
                ax.plot(leading_car.x, leading_car.y, '.')
            road.plot_road(ax)
            # assert len(solver.r_complex1s)> 0, "check reward list"

            # wt = 3
            # ax.set_xlim([oft.x[0]-wt*window_width, oft.x[0]+window_width*wt])
            # ax.set_ylim([oft.y[0]-wt*window_width, oft.y[0]+window_width*wt])
            ws= 80 # window size
            # ax.set_xlim([oft.x[0]-0.5*ws +0.4* ws * np.cos(oft.yaw[0]), oft.x[0]+0.5*ws +0.4* ws * np.cos(oft.yaw[0])])
            # ax.set_ylim([oft.y[0]-0.2*ws +0.4* ws * np.sin(oft.yaw[0]), oft.y[0]+0.2*ws +0.4* ws * np.sin(oft.yaw[0])])
            # ax.set_aspect('equal')
            # # ax.set_title(f"Highway time: {sim_time:.2f} [s] Altitude: {profile_altitude[int(tcar_e.s)]:.2f}")
            # ax.set_title(f"Highway time: {sim_time:.2f} [s] Altitude: {profile_altitude[int(tcar_e.s)]:.2f} [m]  V: {oft.v[0]*3.6:.3f} km/h fr: {oft.fr[0]:.3f} ml/s")
            # ax.set_title(f"Highway time: {sim_time:.2f} [s] Altitude: {profile_altitude[int(tcar_e.s)]:.2f} [m]  V: {oft.v[0]*3.6:.3f} [km/h] ")
            
            # Set the limits of the axes
            ax.set_xlim([oft.x[0] - 0.5 * ws + 0.4 * ws * np.cos(oft.yaw[0]), oft.x[0] + 0.5 * ws + 0.4 * ws * np.cos(oft.yaw[0])])
            # ax.set_ylim([oft.y[0] - 0.1 * ws + 0.1 * ws * np.sin(oft.yaw[0]), oft.y[0] + 0.1 * ws + 0.1 * ws * np.sin(oft.yaw[0])])
            ax.set_ylim([oft.y[0] - 0.5 * ws + 0.4 * ws * np.sin(oft.yaw[0]), oft.y[0] + 0.4 * ws + 0.4 * ws * np.sin(oft.yaw[0])])

            # Set equal aspect ratio
            ax.set_aspect('equal')

            # Customize the font properties
            # title_font = {'family': 'Times New Roman', 'size': 28, 'weight': 'bold'}
            title_font = {'family': 'Times New Roman', 'size': 32}
            label_font = {'family': 'Times New Roman', 'size': 10}

            # Set the title with the customized font
            # ax.set_title(f"Highway time: {sim_time:.2f} [s]   Traveld S: {traveled_s:.2f} [m]   Altitude: {profile_altitude[int(tcar_e.s)]:.2f} [m]   V: {oft.v[0]*3.6:.3f} [km/h]", fontdict=title_font)

            # Customize the axis labels with the customized font
            ax.set_xlabel('X ', fontdict=label_font)
            ax.set_ylabel('Y ', fontdict=label_font)

            # Customize the tick labels for the axes
            ax.tick_params(axis='both', which='major', labelsize=10)
            for tick in ax.get_xticklabels():
                tick.set_fontname("Times New Roman")
            for tick in ax.get_yticklabels():
                tick.set_fontname("Times New Roman")
            
            ax2.fill_between(traveled_s_hist, elevation_hist, color = 'lightgray', edgecolor='black')
            # ax2.set_ylabel('Elevation [m]')
            # ax2.set_xlabel('Traveled S [m]')
            # ax2.set_ylim([90,max(elevation_hist)])
            # ax2.set_xlim([0, max(100, traveled_s)])
            # Customize ax2 similar to ax
            ax2.set_ylabel('Elevation [m]', fontdict=label_font)
            ax2.set_xlabel('Traveled S [m]', fontdict=label_font)
            ax2.set_ylim([min(elevation_hist)-10, max(elevation_hist)+10])
            ax2.set_xlim([0, max(100, traveled_s)])
            
            ax2.tick_params(axis='both', which='major', labelsize=10)
            for tick in ax2.get_xticklabels():
                tick.set_fontname("Times New Roman")
            for tick in ax2.get_yticklabels():
                tick.set_fontname("Times New Roman")

            # Set figure-level title (for both subplots)
            title_font = {'family': 'Times New Roman', 'size': 50}
            fig.suptitle(f"Highway time: {sim_time:.2f} [s]   Traveled S: {traveled_s:.2f} [m]   "
                        f"Altitude: {profile_altitude[int(tcar_e.s)]:.2f} [m]   V: {oft.v[0]*3.6:.3f} [km/h]",
                        fontdict=title_font)

            print("time : ", sim_time)
            # for inft in acc_invalids:
                # print("r_jerk: {}, r_collision: {}, r_cur: {}, max_jerk: {}".format(inft.r_jerk,inft.r_collision, inft.r_curvature, np.max(inft.flon_jerk + inft.flat_jerk)) )
           
            print("r_jerk: {}, r_v: {}, r_collision: {}, r_cur: {}, max_jerk: {}".format(oft.r_jerk, oft.r_v, oft.r_collision, oft.r_curvature, np.max(oft.flon_jerk + oft.flat_jerk)) )
            # plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.subplots_adjust(hspace=0.5, wspace=0.2) 
            plt.pause(0.1)
            # if sim_time > 67.0 and sim_time < 73.5:
            #     plt.show()
            # img_name = f"{sim_time:.2f}"
            # plt.savefig('data/frenet/img2/'+img_name+'.svg')   
            # plt.show()

        # assert 1==2, "sdfad"


        leading_dist = np.inf
        leading_car = None
        traffic_traj_sd_list = []

        for tcar in tcars[:]:
            try:
                # Update position and calculate future points
                tcar.traffic_update_position(param.dt)
                tcar.calc_traffic_future_sd_points(param.prediction_time + param.dt, param.dt)
                
                # Check the conditions
                if tcar.s - tcar_e.s > check_lb and tcar.s - tcar_e.s < check_ub:
                    traffic_traj_sd_list.append(tcar.get_future_sd_points())
                    
                    if abs(tcar.d - tcar_e.d) < 2:
                        if tcar.s - tcar_e.s < leading_dist and tcar.s - tcar_e.s > 5:
                            leading_car = tcar
                            leading_dist = tcar.s - tcar_e.s
                
                # Set the xy and yaw values based on the calculated sd points
                tcar.set_xyyaw_with_sd()
            
            except IndexError as e:
                # Handle the IndexError by removing the problematic tcar from the list
                print(f"Skipping car due to IndexError: {e}")
                tcars.remove(tcar)

        if len(feasible_candidates) is not 0:               
            if if_use_frenet_policy:
                xc = oft.s[1],oft.s_d[1],oft.s_dd[1],oft.d[1],oft.d_d[1],oft.d_dd[1]
                tcar_e.set_xc(xc)
                tcar_e.set_xyyaw(oft.x[1], oft.y[1], oft.yaw[1])

                traveled_s = oft.s[1] - start_s
                traveled_l += oft.dl[0]
                total_fc += oft.fr[0] * param.dt

            else: # Use NLP

                xc = new_s[1],new_s_d[1],new_s_dd[1], new_d[1], new_d_d[1], new_d_dd[1]
                tcar_e.set_xc(xc)
                tcar_e.set_xyyaw(new_x[1], new_y[1], new_yaw[1])
                traveled_s = new_s[1] - start_s 
                traveled_l += dl[0]
                total_fc += fr[0]*param.dt

        else:
            xc = poft.s[1+invalid_count],poft.s_d[1+invalid_count],poft.s_dd[1+invalid_count],poft.d[1+invalid_count],poft.d_d[1+invalid_count],poft.d_dd[1+invalid_count]
            tcar_e.set_xc(xc)
            tcar_e.set_xyyaw(poft.x[1+invalid_count], poft.y[1+invalid_count], poft.yaw[1+invalid_count])
            traveled_s = oft.s[1+invalid_count] - start_s
            traveled_l += oft.dl[0+invalid_count]
            total_fc += oft.fr[0+invalid_count] * param.dt

        v_hist = np.diff(traveled_s_hist) / 0.1
        # Calculate acceleration history (second derivative)
        a_hist = np.diff(v_hist) / 0.1
        # Calculate jerk history (third derivative)
        jerk_hist = np.diff(a_hist) / 0.1
        print(" ********* Data display")
        print("Car: {}, Gtype: {}, Rtype: {}, NLP: {}".format(car_type, g_type, r_type, not if_use_frenet_policy))
        print("simed time: ", sim_time)
        print("traveled s: {}, l: {}, fc: {} ".format(traveled_s,traveled_l,total_fc))
        print("total fe_ds: {}, instant fe_ds: {}, instant fe: {}".format(total_fc/traveled_s,oft.fe_ds[0], oft.fe[0]))
        print("fuel rate: ", oft.fr[0:5])
        print("jerk:{}, lon_lat_jerk:{} ,r_jerk: {}, r_v: {}, r_fe: {}, r_complex2: {}, r_collision: {}, r_cur: {}, max_jerk: {}".format(jerk_hist if len(jerk_hist)< 4 else jerk_hist[-1], oft.flon_jerk[0] + oft.flat_jerk[0],  oft.r_jerk, oft.r_v, oft.r_fe, oft.r_complex2,oft.r_collision, oft.r_curvature, np.max(oft.flon_jerk + oft.flat_jerk)) )
        # print("tl_hist: {}, v_hist: {}, a_hist:{} ".format(traveled_s_hist, v_hist, a_hist))
        # print("jerk hist {},  jerk squre {} ".format(jerk_hist, jerk_hist**2))
        print("l based average jerk square: {}".format(np.sum(jerk_hist**2)/ len(jerk_hist)) )
        print("jerk oft sd: ", oft.flon_jerk[0] + oft.flat_jerk[0])
        # print("jerk emato sd: ", (new_s_ddd**2)[0] +(new_d_ddd**2)[0])

        if if_use_frenet:
            oft_jerks.append(oft.flon_jerk[0] + oft.flat_jerk[0])
        else:
            oft_jerks.append((new_s_ddd**2)[0] +(new_d_ddd**2)[0])

        # print("########### average oft jerk: ", np.sum(oft_jerks)/len(oft_jerks))
        # print("########### average emato jerk: ", np.sum(emato_jerks)/len(emato_jerks))
        ci = (oft.s/0.1).astype(int)
        

        

        sim_time += param.dt
    # fig2, ax2 = plt.subplots(figsize=(10, 5))
    # temp_t = 0
    # # To store the coordinates of the red dots
    # red_dots_x = []
    # red_dots_y = []

    # i = 0
    # for v_array in v_list:
    #     t_list = np.array([param.dt * i for i in range(len(v_array))])
    #     t_list = temp_t + t_list
        
    #     # Plot time series in orange
    #     ax2.plot(t_list, v_array, 'orange', alpha=0.5)  
    #     ax2.plot(t_list, oft_s_d_list[i], 'lightblue', alpha = 0.2)
        
    #     # Plot points as red dots with specified edge color and size
    #     ax2.plot(t_list[0], v_array[0], 'o', color='red', 
    #             markeredgecolor='none', markersize=2)  # Plotting the first point

    #     # Store the coordinates of the red dots to connect later
    #     red_dots_x.append(t_list[0])
    #     red_dots_y.append(v_array[0])
        
    #     temp_t += param.dt
    #     i += 1

    # # Connect the red dots with a line
    # if red_dots_x and red_dots_y:  # Check if there are any red dots to connect
    #     ax2.plot(red_dots_x, red_dots_y, color='red', linestyle='-', linewidth=1)  # Line connecting red dots

    # plt.show()
    # with open('data/frenet/v_list.pkl', 'wb') as f:
    #     pickle.dump(v_list, f)

    return sim_time, traveled_s, traveled_l, total_fc, total_fc/traveled_s, traveled_l_hist, oft_jerks

if __name__ == '__main__':
    if_plot = False
    sim_time_list = []
    traveled_s_list = []
    traveled_l_list = []
    total_fc_list = []
    total_fe_ds_list = []
    car_type_list = []
    g_type_list = []
    r_type_list = []
    use_frenet_list = []

    average_s_d_list = []
    average_v_list = []
    a_oft_jerk = []
  

    # for car_type in ['truck']:
    #     for g_type in ['steep', 'rolling', 'flat']:
    #         for r_type in [1, 2, 3]:

    # Quantative 
    # for car_type in ['truck', 'car']:
    #     for g_type in ['steep','rolling','flat']:
    #         for r_type in [1, 2, 3]:
    #             for if_use_frenet in [False, True]:

    # for car_type in ['truck']:
    #     for g_type in ['rolling']:
    #         for r_type in [1]:
    #             for if_use_frenet in [False]:
    
    # Quantative 
    for car_type in ['truck', 'car']:
        for g_type in ['steep','rolling','flat']:
            for r_type in [1, 2, 3]:
                for if_use_frenet in [False, True]:
                    car_type_list.append(car_type)
                    g_type_list.append(g_type)
                    r_type_list.append(r_type)
                    st, ts, tl, tfc, tfe, tl_hist, oft_jerks = frenet_sim(if_plot=if_plot, car_type=car_type, g_type=g_type, r_type=r_type, if_use_frenet_policy=if_use_frenet)
                    sim_time_list.append(st)
                    traveled_s_list.append(ts)
                    traveled_l_list.append(tl)
                    total_fc_list.append(tfc)
                    total_fe_ds_list.append(tfe)
                    use_frenet_list.append(if_use_frenet)

                    average_s_d_list.append(ts/st)
                    average_v_list.append(tl/st)
                    # Calculate velocity history (first derivative)
                    v_hist = np.diff(tl_hist) / 0.1
                    # Calculate acceleration history (second derivative)
                    a_hist = np.diff(v_hist) / 0.1
                    # Calculate jerk history (third derivative)
                    jerk_hist = np.diff(a_hist) / 0.1

                    a_oft_jerk.append(np.sum(oft_jerks)/len(oft_jerks))
                    # a_emato_jerk.append(np.sum(emato_jerks)/len(emato_jerks))                

                    # Store the data in a dictionary
                    simulation_data = {
                        "car_type": car_type_list,
                        "g_type": g_type_list,
                        "r_type": r_type_list,
                        "sim_time": sim_time_list,
                        "traveled_s": traveled_s_list,
                        "traveled_l": traveled_l_list,
                        "total_fc": total_fc_list,
                        "total_fe_ds": total_fe_ds_list,
                        "if_use_frenet": use_frenet_list,
                        "s_d":average_s_d_list,
                        "v":average_v_list,
                        "oft_jerk":a_oft_jerk
                    }

                    # Save the data as a .pkl file
                    file_route = 'data/frenet/tests/'
                    # file_name = file_route +f'{car_type}_{g_type}_{r_type}_{if_use_frenet}.pkl'
                    file_name = file_route + 'total.pkl'
                    with open(file_name, 'wb') as f:
                        pickle.dump(simulation_data, f)

                    # Optionally print all the results
                    for i in range(len(sim_time_list)):
                        print(f"--- Simulation {i+1} ---")
                        print(f"Car Type: {car_type_list[i]}")
                        print(f"G Type: {g_type_list[i]}")
                        print(f"R Type: {r_type_list[i]}")
                        print(f"Simulated Time: {sim_time_list[i]}")
                        print(f"Traveled S: {traveled_s_list[i]}")
                        print(f"Traveled L: {traveled_l_list[i]}")
                        print(f"Total FC: {total_fc_list[i]}")
                        print(f"Total FE DS: {total_fe_ds_list[i]}")
                        print(f"If Frenet:{use_frenet_list[i]}")
                        print(f"Average s_d: {average_s_d_list[i]}")
                        print(f"Average v: {average_v_list[i]}")
                        print(f"Average oft jerk: {a_oft_jerk[i]}")
                        print("------------------------\n")
"""
To do:
1. Analysis candidates
2. NLP setting for 2d
3. Rendering better with [attitute] and [speed]

"""