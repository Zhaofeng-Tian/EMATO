from time import time
import numpy as np
from emato.alg.frenet import Quintic_2d
from emato.param.param import TruckParam
from emato.obj.road.road import Road
from emato.obj.car.car import Car, Car_J
from emato.obj.car.traffic_car import TrafficCar
from emato.util.util import get_leading_profile, local_approx, get_gradient_profile, get_traj_theta
from emato.alg.nlp import NLP, NLP_J
from emato.alg.bnlp import BNLP, BNLP_R
from emato.alg.quintic import Quintic_1d, Quintic_1d_R
from emato.util.recorder import Recorder
from emato.util.plot import plot_cars
import pickle
import matplotlib.pyplot as plt

def main(if_plot, solver_type):
    total_time = 100.0
    time_step = 0.1
    recorder = Recorder()
    if_plot = if_plot
    param = TruckParam(cycle='HWFET')
    param.gd_profile_type = "rolling"
    sim_time = param.ts
    global_time = 0

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
    tcar1 = TrafficCar(lane=0, sd= [bs+30,0], speed=70/3.6, road = road)
    tcar2 = TrafficCar(lane=0, sd = [bs+0,0], speed=70/3.6 ,road = road)
    tcar3 = TrafficCar(lane=1, sd = [bs+20,0], speed=66/3.6,road = road)
    tcar4 = TrafficCar(lane=1, sd = [bs+45,0], speed=65/3.6 ,road = road)
    tcar5 = TrafficCar(lane=2, sd = [bs+0,0], speed=60/3.6 ,road = road)
    tcar6 = TrafficCar(lane=2, sd = [bs+50,0],speed=60/3.6,road = road)
    tcar7 = TrafficCar(lane=2, sd = [bs+100,0],speed=60/3.6 ,road = road)
    tcars = [tcar1, tcar2, tcar3, tcar4, tcar5, tcar6, tcar7]

    """"
    Ego car initialization
    """
    # Dynmical car
    start_s = 0
    car_e = Car_J(param)
    # car_e's s coordinate is the path arc coordinate
    car_e.set_state(0, 60/3.6, 0.0, jerk = 0.0, theta=profile_gradient[int(start_s)], fc=0.0)
    # Rendering car
    tcar_e = TrafficCar(lane=1, sd = [bs-100,0],speed = 12.0, road = road)
    """
    Solver set up
    """
    # Traffic trajectory setup, define which cars are cared by distance
    es,ed = tcar_e.s, tcar_e.d
    s_list = []
    d_list = []
    cared_index_list = []
    traffic_traj_sd_list = []
    for i in range(len(tcars)):
        if tcars[i].s - es > -50 and tcars[i].s - es < 50:
            cared_index_list.append(i)
            tcars[i].calc_future_sd_points(param.prediction_time,param.dt)
            traffic_traj_sd_list.append(tcars[i].get_future_sd_points())
    print("Traffic traj list:{} len {}: ".format(traffic_traj_sd_list,len(traffic_traj_sd_list)))
        
    # solver setup
    solver = Quintic_2d(param, road, traffic_traj_sd_list, xc)










 
    # if solver_type == 'NLP_J':
    #     solver = NLP_J(param, (traj_sl, traj_vl, traj_al), xc=car_e.get_x())
    # elif solver_type == 'BNLP_R':
    #     solver = BNLP_R(param, (traj_sl, traj_vl, traj_al), xc=car_e.get_x())    
    # elif solver_type == 'Quintic_1d':
    #     solver = Quintic_1d(param,(traj_sl, traj_vl, traj_al), xc=car_e.get_x())
    # elif solver_type == 'Quintic_1d_R':
    #     solver = Quintic_1d_R(param,(traj_sl, traj_vl, traj_al), xc=car_e.get_x())
    # elif solver_type == 'BNLP':
    #     solver = BNLP(param, (traj_sl, traj_vl, traj_al), xc=car_e.get_x())
    # # print("check quintic solver ", solver2.T)
    # # assert 1==2, "check quintic"

    # fig, ax = plt.subplots(figsize=(10, 5))
    # window_width = 8

    # # Plot the road as two straight lines outside the main loop
    # lane_width = 3.5  # Example lane width, adjust if needed
    # ax.plot([wx[0], wx[1]], [-lane_width/2, -lane_width/2], color='gray', linewidth=2)
    # ax.plot([wx[0], wx[1]], [lane_width/2, lane_width/2], color='gray', linewidth=2)

    # for i in range(int(10 * param.ts), int(10 * param.te)):
    # # for i in range(int(10 * param.ts), int(10 * param.ts + 10)):
    #     """
    #     A. Trajectory solving
    #     """
    #     xc = car_e.get_x()
    #     solver.update(xc, (traj_sl, traj_vl, traj_al))
        
        

        
    #     # car_e.set_at_av_ab(at, av, ab)
    #     if solver.id == 'NLP':
    #         print("solver is NLP")
    #         at, av, ab, jerk = solver.solve()
    #         print("****************** at: {}, av: {}, ab: {}, jerk: {} *************".format(at,av,ab,jerk))
    #         car_e.set_at_av_ab_jerk(at, av, ab,jerk)
    #     if solver.id == 'Quintic':
    #         av, jerk = solver.solve()
    #         car_e.set_av_jerk(av,jerk)
    #     print(" ******************** solve time: {} *******************".format(solver.get_solve_info()['solve_time']))
    #     traj_s = solver.get_traj_s()
    #     # print("traj_s of ego: ", traj_s)
    #     """
    #     B. Rendering 
    #     """
    #     if if_plot:
    #         ax.clear()
    #         # Redraw the road lines
    #         ax.plot([wx[0], wx[1]], [-lane_width/2, -lane_width/2], color='gray', linewidth=2)
    #         ax.plot([wx[0], wx[1]], [lane_width/2, lane_width/2], color='gray', linewidth=2)

    #         # rendering setting:
    #         tcar_l.set_1d_traj(future_s_points=traj_sl)
    #         tcar_e.set_1d_traj(future_s_points=traj_s)

    #         cars = [tcar_e, tcar_l]
    #         plot_cars(ax, cars)
    #         ax.set_xlim([cars[0].pose_history[-1][0] - window_width, cars[0].pose_history[-1][0] + window_width * 20])
    #         ax.set_ylim([cars[0].pose_history[-1][1] - window_width, cars[0].pose_history[-1][1] + window_width])
    #         ax.set_aspect('equal')
    #         ax.set_title(f"Highway Simulation at t={sim_time:.2f} s")
    #         plt.pause(0.1)

    #     """
    #     C. Data recording
    #     """
    #     recorder.recordj(car_l.get_state(), car_e.get_state(),sim_time)
    #     recorder.record_solve_info(solver.get_solve_info())

    #     """
    #     C. Step process
    #     """
    #     sim_time += param.dt
    #     global_time += param.dt

    #     # 1. Ego step:
    #     car_e.step()
    #     theta_new = profile_gradient[int(car_e.get_s())]
    #     print("time {}, s: {}".format(i * param.dt, car_e.get_s()))
    #     car_e.set_theta(theta_new)
    #     print("car states: ", car_e.get_state())
    #     assert car_e.v < 50, "wrong speed!!"

    #     # 2. Leading step:
    #     car_l.step()
    #     traj_sl, traj_vl, traj_al = local_approx(profile_leading, time_target=sim_time, delta_time=param.prediction_time, dt=param.dt)
    #     jerkl = profile_leading['jerk_profile'][round(sim_time/param.dt)]
    #     car_l.set_state(traj_sl[0], traj_vl[0], traj_al[0],jerk = jerkl ,theta=profile_gradient[int(traj_sl[0])], fc=car_l.fc)

    # """
    # Save data
    # """

    # #  plt.show()

    # recorder.plot_trajectory()

    # data_to_save = {
    #     'recorder': recorder,
    #     'param': param
    # }
    # file_name = f"{param.car_type}_{param.cycle}_{param.gd_profile_type}_{param.solver_type}_{param.w1}_{param.w2}_{param.w3}_gd_{param.use_gd_prediction}_ptime_{param.prediction_time}.pkl"
    # # Serialize the data and save it to a file
    # # file_name = self.save_route + file_name
    # route_head = 'emato/data/'
    # file_name = route_head + file_name
    # with open(file_name, 'wb') as file:
    #     pickle.dump(data_to_save, file)
    

if __name__ == '__main__':
    main(if_plot=False, solver_type='NLP_J')