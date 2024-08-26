
from time import time
import numpy as np
from emato.alg.quintic2d import Quintic_2d
from emato.param.param import TruckParam
from emato.obj.road.road import Road
from emato.obj.car.car import Car, Car_J
from emato.obj.car.traffic_car import TrafficCar
from emato.util.util import get_leading_profile, local_approx, get_gradient_profile, get_traj_theta
from emato.alg.nlp import NLP, NLP_J
from emato.alg.bnlp import BNLP, BNLP_R
from emato.alg.quintic import Quintic_1d, Quintic_1d_R
from emato.util.recorder import Recorder
from emato.util.plot import plot_cars, plot_traffic_traj
from emato.alg.emato import EMATO

import matplotlib.pyplot as plt

def main(if_plot, r_type):
    sim_time = 0
    total_time = 138
    time_step = 0.1
    recorder = Recorder()
    if_plot = if_plot
    param = TruckParam(cycle='HWFET')
    param.gd_profile_type = "steep"
    param.prediction_time = 5
    param.N = round(param.prediction_time/param.dt)+1
    param.rsafe = 4
    param.desired_v = 70/3.6
    param.desired_d = 0
    # sim_time = param.ts
    global_time = 0
    check_lb = -50
    check_ub = 80

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
    for bs in np.arange(0, 1800, 60):
        tcars.append(TrafficCar(lane=0, sd = [bs, 0], speed = 50/3.6, road = road))
    for bs in np.arange(30, 1800, 90):
        tcars.append(TrafficCar(lane=1, sd = [bs, 0], speed = 56/3.6, road = road))
    for bs in np.arange(40, 1800, 70):
        tcars.append(TrafficCar(lane=2, sd = [bs, 0], speed = 60/3.6, road = road))

    """"
    Ego car initialization
    """
    # Dynmical car
    start_s = 195
    # car_e's s coordinate is the path arc coordinate
    # Rendering car
    tcar_e = TrafficCar(lane=1, sd = [start_s,0],speed = 12.0, road = road)
    es,ed = tcar_e.s, tcar_e.d
    xc = [es, 70/3.6, 0, ed, 0, 0]
    tcar_e.set_xc(xc)

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
        

    # leading_sd_end = leading_car.future_sd_points[-1] if leading_car is not None else None
        
    """
    Solver setup
    """
    emato = EMATO(param)
    solver = Quintic_2d(param, road, traffic_traj_sd_list, \
                        leading_car,\
                        xc,desired_v=param.desired_v, desired_d = param.desired_d)

    fig, ax = plt.subplots(figsize=(10, 5))
    window_width = 10
    

    # Data recording set up
    traveled_l = 0
    traveled_s = 0
    total_fc = 0
    average_v = 0 # along arcwarts **************************************

    # Simulation loop
    # for i in range(0, round(total_time/param.dt)):
    while traveled_s < 2200:
        ax.clear()
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
        # print("**************** len of oft: ", len(oft.s))
        # print("***************** N : ", param.N)
        # assert len(oft.s) == param.N

        emato.update(oft)
        res = emato.solve()
        # print("res[:,N]: ", res[:param.N])
        # print("res[N,2N]: ",res[param.N : 2*param.N])

        v = res[param.N : param.N *2]
        at = res[param.N*2: param.N *3]
        o = param.o; c = param.c 
        ofr = o[0] + o[1]*v + o[2]*v**2+o[3]* v**3 + o[4]*v**2+(c[0]+c[1]*v + c[2]*v**2)*at
        ofc = np.cumsum(ofr * param.dt)

        assert ofc[-1] < oft.fc[-1], "getting worse fc!!"
        print('***************************ofc: {}, oft: {} **************'.format(ofc[-1], oft.fc[-1]))
        
        """ Pseudo function
        
        """
        if if_plot:
            # for ft in invalid_candidates:
            #     ft.check_collision(traffic_traj_sd_list, param.rsafe)
            #     if ft.if_xy_collision:
            #         # print("check in 2d.py ", ft.if_xy_collision)
            #         plt.plot(ft.x,ft.y, 'red', alpha = 0.2)
            #     elif ft.if_over_curvy:
            #         plt.plot(ft.x,ft.y, 'orange', alpha = 0.5)
            #     #     print(max(ft.cur))
            #     elif ft.if_over_jerky:
            #         plt.plot(ft.x,ft.y, 'yellow', alpha = 0.5)

            for ft in feasible_candidates:
                plt.plot(ft.x, ft.y, 'lightgreen', alpha = 0.6)



                # print("In feasible set check if xy collision",ft.if_xy_collision)
                # sd_collision, xy_collision = ft.check_collision(traffic_traj_sd_list, param.rsafe)
                # assert not sd_collision, "sd collision happened in feasible set!!"
                # assert not xy_collision, "xy collision happened in feasible set!!"
                

            # assert len(acc_invalids) is not 0, "len acc invalid not 0"
            for ft in acc_invalids:
                plt.plot(ft.x,ft.y, 'gray')
            for ft in acc_feasibles:
                plt.plot(ft.x,ft.y, 'blue')
            
            # Select an traj with an optimal policy


            plt.plot(oft1.x, oft1.y, 'purple', alpha = 1)
            plt.plot(oft2.x, oft2.y, 'orange', alpha = 1)
            plt.plot(oft3.x, oft3.y, 'red', alpha = 1)


            # plot_traffic_traj(ax,tcars)
            plot_cars(ax,tcars,if_plot_future=False)
            plot_cars(ax,[tcar_e],if_plot_future=False)
            if leading_car is not None:
                plt.plot(leading_car.x, leading_car.y, '.')
            road.plot_road(ax)
            assert len(solver.r_complex1s)> 0, "check reward list"

            wt = 8
            ax.set_xlim([oft.x[0]-wt*window_width, oft.x[0]+window_width*wt])
            ax.set_ylim([oft.y[0]-wt*window_width, oft.y[0]+window_width*wt])
            ax.set_aspect('equal')
            # ax.set_title(f"Highway time: {sim_time:.2f} [s] Altitude: {profile_altitude[int(tcar_e.s)]:.2f}")
            ax.set_title(f"Highway time: {sim_time:.2f} [s] Altitude: {profile_altitude[int(tcar_e.s)]:.2f} [m]  V: {oft.v[0]*3.6:.3f} km/h fr: {oft.fr[0]:.3f} ml/s")
            # ax.set_title(f"Highway time: {sim_time:.2f} [s] d dist: {abs(tcar_e.d-leading_car.d):.2f} [m] ")
            print("time : ", sim_time)
            # for inft in acc_invalids:
                # print("r_jerk: {}, r_collision: {}, r_cur: {}, max_jerk: {}".format(inft.r_jerk,inft.r_collision, inft.r_curvature, np.max(inft.flon_jerk + inft.flat_jerk)) )
           
            print("r_jerk: {}, r_v: {}, r_collision: {}, r_cur: {}, max_jerk: {}".format(oft.r_jerk, oft.r_v, oft.r_collision, oft.r_curvature, np.max(oft.flon_jerk + oft.flat_jerk)) )
            print("oft s_d: ", oft.s_d[0:5]*3.6)
            print("oft v: ", oft.v[0:5]*3.6)
            print("end s_d: {}, desired_v: {}".format(oft.s_d[-1]*3.6, param.desired_v*3.6))

            plt.pause(0.5)    
            # plt.show()

        # assert 1==2, "sdfad"


        leading_dist = np.inf
        leading_car = None
        traffic_traj_sd_list = []
        # for tcar in tcars:
        #     tcar.traffic_update_position(param.dt)
        #     tcar.calc_traffic_future_sd_points(param.prediction_time+param.dt, param.dt)
        #     if tcar.s - tcar_e.s > check_lb and tcar.s - tcar_e.s < check_ub:
        #         traffic_traj_sd_list.append(tcar.get_future_sd_points())
        #         if abs(tcar.d - tcar_e.d) < 2:
        #             if tcar.s - tcar_e.s < leading_dist and tcar.s - tcar_e.s > 5:
        #                 leading_car = tcar
        #                 leading_dist = tcar.s - tcar_e.s
            
        #     tcar.set_xyyaw_with_sd()

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

        xc = oft.s[1],oft.s_d[1],oft.s_dd[1],oft.d[1],oft.d_d[1],oft.d_dd[1]
        tcar_e.set_xc(xc)
        tcar_e.set_xyyaw(oft.x[1], oft.y[1], oft.yaw[1])

        traveled_s = oft.s[1] - start_s
        traveled_l += oft.dl[0]
        total_fc += oft.fr[1] * param.dt

        print(" ********* Data display")
        print("simed time: ", sim_time)
        print("traveled s: {}, l: {}, fc: {} ".format(traveled_s,traveled_l,total_fc))
        print("total fe_ds: {}, instant fe_ds: {}, instant fe: {}".format(total_fc/traveled_s,oft.fe_ds[0], oft.fe[0]))
        print("fuel rate: ", oft.fr[0:5])
        print("r_jerk: {}, r_v: {}, r_fe: {}, r_complex2: {}, r_collision: {}, r_cur: {}, max_jerk: {}".format(oft.r_jerk, oft.r_v, oft.r_fe, oft.r_complex2,oft.r_collision, oft.r_curvature, np.max(oft.flon_jerk + oft.flat_jerk)) )
        
        
        # print("s: ", oft.s)
        # print("d: ", oft.d)
        # print("s_d: ", oft.s_d)
        # print("d_d: ",oft.d_d)
        # curv index
        ci = (oft.s/0.1).astype(int)
        
        # print("x:", oft.x)
        # print("y:", oft.y)
        # print("dx: ", oft.dx)
        # print("dy: ", oft.dy)
        # print("dl: ", oft.dl)
        # print("l:", oft.l)
        # print("v:", oft.v)
        # print("at: ", oft.at)
        # print("ar: ", oft.ar    )
        # print("av: ", oft.a)
        # print("jerk: ", oft.jerk)
        # print("road curv: ",road.tk[ci])
        # print("curve: ", oft.cur)
        

        sim_time += param.dt



if __name__ == '__main__':
    main(if_plot=True, r_type=2)

"""
To do:
1. Analysis candidates
2. NLP setting for 2d
3. Rendering better with [attitute] and [speed]

"""