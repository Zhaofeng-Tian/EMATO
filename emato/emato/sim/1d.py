
from time import time
import numpy as np
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
    param = TruckParam(cycle='HWFET')
    print("param.dt: ",param.dt)
    sim_time = param.ts
    global_time = 0

    wx = [0.0, 10000]
    wy = [0.0, 0.0]
    road = Road(wx, wy, num_lanes=1)
    recorder = Recorder()
    if_plot = if_plot

    # Setting global parameters
    param.gd_profile_type = "rolling"
    param.w1 = 0.1
    param.w2 = 5
    param.w3 = 2
    param.dinit = 50
    param.dmin = 10
    param.dmax = 200
    param.prediction_time = 1
    print("param.dt: ",param.dt)
    param.N = int(param.prediction_time / param.dt)

    # Section. A: Initialization
    profile_leading = get_leading_profile(param.cycle)
    profile_altitude, profile_gradient = get_gradient_profile(feature=param.gd_profile_type)
    traj_sl, traj_vl, traj_al = local_approx(profile_leading, time_target=sim_time, delta_time=param.prediction_time, dt=param.dt)
    jerkl = profile_leading['jerk_profile'][round(sim_time/param.dt)]
    # sl = profile_leading['s_profile'][round(sim_time/param.dt)]
    # print("jerkl: ", jerkl)
    # print(" compare traj_sl[0]: {} and sl: {} ".format(traj_sl[0], sl))
    # assert 1==2, " check results"
    param.dinit = traj_sl[0] - param.dmin - traj_vl[0] * param.accT

    car_l = Car_J(param)
    car_e = Car_J(param)
    car_l.set_state(traj_sl[0], traj_vl[1], traj_al[2], jerk = 0.,theta=profile_gradient[int(traj_sl[0])], fc=0.0)
    car_e.set_state(traj_sl[0] - param.dinit, traj_vl[1], traj_al[2], jerk = jerkl, theta=profile_gradient[int(traj_sl[0] - param.dinit)], fc=0.0)

    tcar_l = TrafficCar(lane=0, sd=[car_l.s, 0], speed=0.0, road=road)
    tcar_e = TrafficCar(lane=0, sd=[car_e.s, 0], speed=0.0, road=road)
    # return np.array([self.s, self.v, self.av, self.jerk, self.at, self.ar, self.fr, self.fc, self.theta])
    if solver_type == 'NLP_J':
        solver = NLP_J(param, (traj_sl, traj_vl, traj_al), xc=car_e.get_x())
    elif solver_type == 'NLP':
        solver = NLP(param, (traj_sl, traj_vl, traj_al), xc=car_e.get_x())
    elif solver_type == 'BNLP_R':
        solver = BNLP_R(param, (traj_sl, traj_vl, traj_al), xc=car_e.get_x())    
    elif solver_type == 'Quintic_1d':
        solver = Quintic_1d(param,(traj_sl, traj_vl, traj_al), xc=car_e.get_x())
    elif solver_type == 'Quintic_1d_R':
        solver = Quintic_1d_R(param,(traj_sl, traj_vl, traj_al), xc=car_e.get_x())
    elif solver_type == 'BNLP':
        solver = BNLP(param, (traj_sl, traj_vl, traj_al), xc=car_e.get_x())
    # print("check quintic solver ", solver2.T)
    # assert 1==2, "check quintic"

    fig, ax = plt.subplots(figsize=(10, 5))
    window_width = 8

    # Plot the road as two straight lines outside the main loop
    lane_width = 3.5  # Example lane width, adjust if needed
    ax.plot([wx[0], wx[1]], [-lane_width/2, -lane_width/2], color='gray', linewidth=2)
    ax.plot([wx[0], wx[1]], [lane_width/2, lane_width/2], color='gray', linewidth=2)

    for i in range(int(10 * param.ts), int(10 * param.te)):
    # for i in range(int(10 * param.ts), int(10 * param.ts + 10)):
        """
        A. Trajectory solving
        """
        xc = car_e.get_x()
        solver.update(xc, (traj_sl, traj_vl, traj_al))
        
        

        
        # car_e.set_at_av_ab(at, av, ab)
        if solver.id == 'NLP':
            print("solver is NLP")
            at, av, ab, jerk = solver.solve()
            print("****************** at: {}, av: {}, ab: {}, jerk: {} *************".format(at,av,ab,jerk))
            car_e.set_at_av_ab_jerk(at, av, ab,jerk)
        if solver.id == 'Quintic':
            av, jerk = solver.solve()
            car_e.set_av_jerk(av,jerk)
        print(" ******************** solve time: {} *******************".format(solver.get_solve_info()['solve_time']))
        traj_s = solver.get_traj_s()
        # print("traj_s of ego: ", traj_s)
        """
        B. Rendering 
        """
        if if_plot:
            ax.clear()
            # Redraw the road lines
            ax.plot([wx[0], wx[1]], [-lane_width/2, -lane_width/2], color='gray', linewidth=2)
            ax.plot([wx[0], wx[1]], [lane_width/2, lane_width/2], color='gray', linewidth=2)

            # rendering setting:
            tcar_l.set_1d_traj(future_s_points=traj_sl)
            tcar_e.set_1d_traj(future_s_points=traj_s)

            cars = [tcar_e, tcar_l]
            plot_cars(ax, cars, if_plot_future=True)
            ax.set_xlim([cars[0].pose_history[-1][0] - window_width, cars[0].pose_history[-1][0] + window_width * 20])
            ax.set_ylim([cars[0].pose_history[-1][1] - window_width, cars[0].pose_history[-1][1] + window_width])
            ax.set_aspect('equal')
            ax.set_title(f"Highway Simulation at t={sim_time:.2f} s")
            plt.pause(0.1)

        """
        C. Data recording
        """
        recorder.recordj(car_l.get_state(), car_e.get_state(),sim_time)
        recorder.record_solve_info(solver.get_solve_info())

        """
        C. Step process
        """
        sim_time += param.dt
        global_time += param.dt

        # 1. Ego step:
        car_e.step()
        theta_new = profile_gradient[int(car_e.get_s())]
        print("time {}, s: {}".format(i * param.dt, car_e.get_s()))
        car_e.set_theta(theta_new)
        print("car states: ", car_e.get_state())
        assert car_e.v < 50, "wrong speed!!"

        # 2. Leading step:
        car_l.step()
        traj_sl, traj_vl, traj_al = local_approx(profile_leading, time_target=sim_time, delta_time=param.prediction_time, dt=param.dt)
        jerkl = profile_leading['jerk_profile'][round(sim_time/param.dt)]
        car_l.set_state(traj_sl[0], traj_vl[0], traj_al[0],jerk = jerkl ,theta=profile_gradient[int(traj_sl[0])], fc=car_l.fc)

    """
    Save data
    """

    #  plt.show()
    print(" Jerk sum squre: ", np.sum(np.array(recorder.Tjerke)**2)/len(recorder.Tjerke))
    print(" A sum squre: ", np.sum(np.array(recorder.Tave)**2) /len(recorder.Tave))
    print(" Fr sum squre: ", np.sum(np.array(recorder.Tfre))/len(recorder.Tfre) )
    print("Average jerk: ", np.sum(recorder.Tjerke)/ len(recorder.Tjerke))
    print("Average speed: ", np.sum(recorder.Tve)/ len(recorder.Tve))
    print("Fc / road length: ml/ meters ", recorder.Tfce[-1]/(recorder.Tse[-1]-recorder.Tse[0]) )
    recorder.plot_trajectory()

    data_to_save = {
        'recorder': recorder,
        'param': param
    }
    file_name = f"{param.car_type}_{param.cycle}_{param.gd_profile_type}_{param.solver_type}_{param.w1}_{param.w2}_{param.w3}_gd_{param.use_gd_prediction}_ptime_{param.prediction_time}.pkl"
    # Serialize the data and save it to a file
    # file_name = self.save_route + file_name
    route_head = 'emato/data/'
    file_name = route_head + file_name
    with open(file_name, 'wb') as file:
        pickle.dump(data_to_save, file)
    

if __name__ == '__main__':
    main(if_plot=True, solver_type='BNLP')

