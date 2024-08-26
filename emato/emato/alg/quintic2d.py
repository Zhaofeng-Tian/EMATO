from emato.obj.poly.quintic import QuinticPolynomial, Traj
import numpy as np
from emato.util.util import *
import time
from emato.obj.poly.frenet_traj import FrenetTraj




class Quintic_2d:
    def __init__(self,param, road,traj_traffic_list, leading_car, xc, desired_v, desired_d):
        """
        xc should be [(s, s', s''), (d, d', d'')]
        that is [(cs, cs_d, cs_dd) (cd, cd_d, cd_dd)]
        """
        self.id = 'Quintic'
        self.param = param
        self.k = param.k
        self.c = param.c
        self.o = param.o
        self.road = road
        self.traj_traffic_list = traj_traffic_list
        self.leading_car = leading_car
        self.xc = xc
        self.ptime = param.prediction_time
        self.dt = self.param.dt
        self.rsafe = param.rsafe
        _, self.gd_profile = get_gradient_profile(feature=param.gd_profile_type)
        self.desired_v = desired_v
        self.desired_d = desired_d
        self.feasible_candidates = []
        self.invalid_candidates = []
        self.acc_feasible_candidates = []
        self.acc_invalid_candidates = []
        self.r_jerks = []
        # self.r_collisions = []
        # self.r_curvatures = []
        self.r_vs = []
        self.r_ds = []
        self.r_complex1s = [] # original frenet objective
        self.r_complex2s = [] # original + fuel efficiency
        self.r_complex3s = [] # only fuel efficiency
        self.solve_time = None

    def solve(self):
        # self.feasible_candidates = []
        # self.invalid_candidates = []
        # 6-tuple states
        cs,cs_d,cs_dd,cd,cd_d,cd_dd = self.xc
        ts = time.time()
        for gd in np.arange(-self.road.road_width/2+0.5, self.road.road_width/2-0.5, 1) :
            # for gs in np.arange(cs+self.desired_v*self.ptime - 30, cs+self.desired_v*self.ptime+30, 1):
            for gs in np.arange(cs+20, cs+self.desired_v*self.ptime+30, 5):
                xg = [gs,self.desired_v,0,gd,0,0]

                # Create Frenet traj obj
                ft = FrenetTraj(self.road, self.k, self.o, self.c)

                # Setup
                ft.setup(self.xc, xg, road = self.road, \
                        ptime = self.ptime, dt = self.dt, \
                        gd_profile=self.gd_profile)

                # Calc reward
                ft.calc_reward(self.traj_traffic_list, self.rsafe, self.param.jerkmax, self.param.kmax, self.desired_v, self.desired_d)
                if ft.r_invalid >= 1000:
                    self.invalid_candidates.append(ft)
                    
                else:
                    self.feasible_candidates.append(ft)
                    self.r_jerks.append(ft.r_jerk)
                    self.r_vs.append(ft.r_v)
                    self.r_ds.append(ft.r_d)
                    self.r_complex1s.append(ft.r_complex1)
                    self.r_complex2s.append(ft.r_complex2)
                    self.r_complex3s.append(ft.r_complex3)

        # Now check acc policy
        if self.leading_car is not None:
            cl_s, gd = self.leading_car.future_sd_points[-1]
            acc_s = cl_s - self.param.dmin - self.param.accT*self.leading_car.speed
            
            for gs in np.arange(acc_s-5, cl_s-self.param.dmin, 1):
                xg = [gs,self.leading_car.speed,0,gd,0,0]
                ft = FrenetTraj(self.road, self.k, self.o, self.c)
                ft.setup(self.xc, xg, road = self.road, \
                        ptime = self.ptime, dt = self.dt, \
                        gd_profile=self.gd_profile)
                
                ft.calc_reward(self.traj_traffic_list, self.rsafe, self.param.jerkmax, self.param.kmax, self.desired_v, self.desired_d)
                
                
                if ft.r_invalid >= 1000:
                    self.invalid_candidates.append(ft)
                    self.acc_invalid_candidates.append(ft)
                else:
                    self.feasible_candidates.append(ft)
                    self.acc_feasible_candidates.append(ft)
                    
                    self.r_jerks.append(ft.r_jerk)
                    self.r_vs.append(ft.r_v)
                    self.r_ds.append(ft.r_d)
                    self.r_complex1s.append(ft.r_complex1)
                    self.r_complex2s.append(ft.r_complex2)
                    self.r_complex3s.append(ft.r_complex3)
                """
                To do: try use s,d and x,y to calc arc length and check collision, curvature, and fuel rate etc.
                """
        te = time.time()
        self.solve_time = te-ts
        print("**************** solve time: ", self.solve_time, " seconds *********************")
        return self.feasible_candidates, self.invalid_candidates, self.acc_feasible_candidates, self.acc_invalid_candidates

    def update(self, xc, traffic_traj_sd_list, leading_car):
        self.xc = xc
        self.traj_traffic_list = traffic_traj_sd_list

        self.feasible_candidates = []
        self.invalid_candidates = []
        self.acc_feasible_candidates = []
        self.acc_invalid_candidates = []
        self.r_jerks = []
        # self.r_collisions = []
        # self.r_curvatures = []
        self.r_vs = []
        self.r_ds = []
        self.r_complex1s = [] # original frenet objective
        self.r_complex2s = [] # original + fuel efficiency
        self.r_complex3s = [] # only fuel efficiency

        # Leading car 
        self.leading_car = leading_car

    def get_solve_info(self):
        return {'solve_time': self.solve_time, 'executed_its': 1}