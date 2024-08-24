from time import process_time, time
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
import pickle
import matplotlib.pyplot as plt

def main(if_plot, solver_type):
    sim_time = 0
    total_time = 100.0
    time_step = 0.1
    recorder = Recorder()
    if_plot = if_plot
    param = TruckParam(cycle='HWFET')
    param.gd_profile_type = "rolling"
    param.prediction_time = 5
    param.N = round(param.prediction_time/param.dt)
    param.rsafe = 3
    param.desired_v = 70/3.6
    param.desired_d = 0
    # sim_time = param.ts
    global_time = 0
    check_lb = -10
    check_ub = 50

    """
    Road initialization
    """
    wx = [0.0, 350.0, 700.0, 1300.0, 1500.0]
    wy = [0.0, 500.0, 150.0, 65.0, 0.0]
    road = Road(wx, wy, num_lanes=3)
    profile_altitude, profile_gradient = get_gradient_profile(feature=param.gd_profile_type)
    bs = 300 # base s
    """
    Simulated traffic cars 
    """
    tcar1 = TrafficCar(lane=0, sd= [bs+30,0], speed=55/3.6, road = road)
    tcar2 = TrafficCar(lane=0, sd = [bs-20,0], speed=55/3.6 ,road = road)
    tcar3 = TrafficCar(lane=1, sd = [bs+80,0], speed=50/3.6,road = road)
    tcar4 = TrafficCar(lane=1, sd = [bs+120,0], speed=50/3.6 ,road = road)
    
    tcar5 = TrafficCar(lane=2, sd = [bs+0,0], speed=50/3.6 ,road = road)
    tcar6 = TrafficCar(lane=2, sd = [bs+80,0],speed=52/3.6,road = road)
    tcar7 = TrafficCar(lane=2, sd = [bs+120,0],speed=53/3.6 ,road = road)
    tcar8 = TrafficCar(lane=1, sd = [bs+150,0], speed=50/3.6 ,road = road)
    tcar9 = TrafficCar(lane=1, sd = [bs+180,0], speed=50/3.6 ,road = road)
    tcar10 = TrafficCar(lane=2, sd = [bs+130,0],speed=67/3.6 ,road = road)
    tcar11 = TrafficCar(lane=1, sd = [bs+210,0], speed=55/3.6 ,road = road)
    tcar12 = TrafficCar(lane=0, sd = [bs+170,0], speed=50/3.6 ,road = road)
    tcar13 = TrafficCar(lane=0, sd = [bs+200,0], speed=53/3.6 ,road = road)
    
    tcars = [tcar1, tcar2, tcar3, tcar4, tcar5, tcar6, \
            tcar7, tcar8, tcar9, tcar10, tcar11,tcar12,tcar13]



    """"
    Ego car initialization
    """
    # Dynmical car
    start_s = 0
    car_e = Car_J(param)
    # car_e's s coordinate is the path arc coordinate
    car_e.set_state(0, 60/3.6, 0.0, jerk = 0.0, theta=profile_gradient[int(start_s)], fc=0.0)
    # Rendering car
    tcar_e = TrafficCar(lane=1, sd = [bs-20,0],speed = 12.0, road = road)
    es,ed = tcar_e.s, tcar_e.d
    xc = [es, 70/3.6, 0, ed, 0, 0]
    tcar_e.set_xc(xc)

    """
    Solver set up
    """
    # Traffic trajectory setup, define which cars are cared by distance
    
    s_list = []
    d_list = []
    cared_index_list = []
    traffic_traj_sd_list = []
    for i in range(len(tcars)):
        if tcars[i].s - es > check_lb and tcars[i].s - es < check_ub:
            cared_index_list.append(i)
            tcars[i].calc_traffic_future_sd_points(param.prediction_time+param.dt,param.dt)
            print("***** len: ", len(tcars[i].get_future_sd_points()))
            traffic_traj_sd_list.append(tcars[i].get_future_sd_points())
            tcars[i].set_xyyaw_with_sd()

        
    # solver setup
    solver = Quintic_2d(param, road, traffic_traj_sd_list, xc,desired_v=param.desired_v, desired_d = param.desired_d)

    fig, ax = plt.subplots(figsize=(10, 5))
    window_width = 10
    road.plot_road(ax)
    

    for i in range(0, round(total_time/param.dt)):
        ax.clear()
        road.plot_road(ax)
        # ***** Update 
        solver.update(xc, traffic_traj_sd_list)
        # Solve
        feasible_candidates, invalid_candidates = solver.solve()
        # ********************
        lenf = len(feasible_candidates); len_in = len(invalid_candidates)
        print(" Total candidates {}, Feasible {}, invalid {} ".format(lenf+len_in, lenf, len_in))
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
            
        # plt.plot(oft.x, oft.y, 'green')
        
        # Select an traj with an optimal policy
        print("how many feasible?" ,len(feasible_candidates))
        # print("r_complex1s list",solver.r_complex1s)
        print("solver.r_complex2s, ",solver.r_complex2s)
        oft_index = np.argmin(solver.r_complex2s)
        print("which is optimal?" ,oft_index)
        oft = solver.feasible_candidates[oft_index]

        plt.plot(oft.x, oft.y, 'green', alpha = 1)


        plot_traffic_traj(ax,tcars)
        plot_cars(ax,tcars,if_plot_future=False)
        plot_cars(ax,[tcar_e],if_plot_future=False)

        assert len(solver.r_complex1s)> 0, "check reward list"

        ax.set_xlim([oft.x[0]-5*window_width, oft.x[0]+window_width*5])
        ax.set_ylim([oft.y[0]-5*window_width, oft.y[0]+window_width*5])
        ax.set_aspect('equal')
        ax.set_title(f"Highway Simulation at t={sim_time:.2f} s")
        print("time : ", sim_time)
        plt.pause(0.1)    
        # plt.show()

        # assert 1==2, "sdfad"



        traffic_traj_sd_list = []
        for tcar in tcars:
            tcar.traffic_update_position(param.dt)
            tcar.calc_traffic_future_sd_points(param.prediction_time+param.dt, param.dt)
            if tcar.s - tcar_e.s > check_lb and tcar.s - tcar_e.s < check_ub:
                traffic_traj_sd_list.append(tcar.get_future_sd_points())
            tcar.set_xyyaw_with_sd()

        xc = oft.s[1],oft.s_d[1],oft.s_dd[1],oft.d[1],oft.d_d[1],oft.d_dd[1]
        tcar_e.set_xc(xc)
        tcar_e.set_xyyaw(oft.x[1], oft.y[1], oft.yaw[1])

        sim_time += param.dt



if __name__ == '__main__':
    main(if_plot=False, solver_type='NLP_J')

"""
To do:
1. Analysis candidates
2. NLP setting for 2d
3. Rendering better with [attitute] and [speed]

"""