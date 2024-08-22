from emato.obj.poly.quintic import QuinticPolynomial, Traj
import numpy as np
from emato.util.util import *
import time

class Quintic_1d:
    def __init__(self,param, traj_l,xc):
        self.id = 'Quintic'
        self.param = param
        self.traj_sl, self.traj_vl, self.traj_al = traj_l
        self.xc = xc

        _, self.gd_profile = get_gradient_profile(feature=param.gd_profile_type)
        # self.s, self.v, self.av, self.jerk, self.at, self.ar, self.fr, self.fc, self.theta
        self.xs = [xc[0], xc[1], xc[2]]


        assert xc[0] < self.traj_sl[0], "collision happened, check code!"
        # ACC estimated end state
        se = self.traj_sl[-1] - (self.param.accT * self.traj_vl[-1] + self.param.dmin)
        ve = self.traj_vl[-1]
        ae = self.traj_al[-1]
        self.xe = [se, ve, ae]

        # assert 1==2, " sdfaasd "
        # self.T = np.arange(0, round(self.param.prediction_time/ self.param.dt), self.dt)
        self.T = self.param.prediction_time
        self.poly_list = []
        self.traj_list = []
        self.r_list = []
        self.rv_list = []
        self.rj_list = []
        self.rfc_list = []

        self.r_index = None
        self.rv_index = None
        self.rj_index = None
        self.rfc_index = None
        self.solve_time = np.inf


    def solve(self):
        self.poly_list = []
        self.traj_list = []
        self.r_list = []
        self.rv_list = []
        self.rj_list = []
        self.rfc_list = []

        self.r_index = None
        self.rv_index = None
        self.rj_index = None
        self.rfc_index = None
        self.solve_time = np.inf
        ts = time.time()
        # for i in range(len(self.T)):
        poly = QuinticPolynomial(self.xs[0],self.xs[1],self.xs[2],\
                                self.xe[0],self.xe[1],self.xe[2],self.T, self.param.dt)
        self.poly_list.append(poly)
        for i in range(len(self.poly_list)):
            p = self.poly_list[i]
            traj = Traj(self.param,p,self.gd_profile,dv=self.xe[1])
            self.traj_list.append(traj)
            self.r_list.append(traj.r)
            self.rv_list.append(traj.rv)
            self.rj_list.append(traj.rj)
            self.rfc_list.append(traj.rfc) 
        self.r_index = self.r_list.index(min(self.r_list))
        self.rv_index = self.rv_list.index(min(self.rv_list))
        self.rj_index = self.rj_list.index(min(self.rj_list))
        self.rfc_index = self.rfc_list.index(min(self.rfc_list))
        self.res = self.traj_list[self.rfc_index]
        te = time.time()
        self.solve_time = te - ts

        # print("Traj_s: ", self.res.Traj_s)
        # print("Traj_v: ", self.res.Traj_v)
        # print("Traj_a: ", self.res.Traj_a)
        # print("Traj_jerk: ", self.res.Traj_jerk)

        assert np.all(np.diff(self.res.Traj_s) >= 0), "Trajectory of s infeasible"



        return self.res.Traj_a[0], self.res.Traj_jerk[0]

    def update(self, xc, traj_l):
        self.xs = [xc[0], xc[1], xc[2]]
        self.traj_sl, self.traj_vl, self.traj_al = traj_l 

        # print(" car state got: ", xc)
        # print(" leading traj_sl: ", self.traj_sl)
        # ACC estimated end state
        se = self.traj_sl[-1] - (self.param.accT * self.traj_vl[-1] + self.param.dmin)
        ve = self.traj_vl[-1]
        ae = self.traj_al[-1]
        self.xe = [se, ve, ae]
        # print("*********xs *************: ", self.xs)
        # print("*********xe *************: ", self.xe)
    def get_traj_s(self):
        return self.res.Traj_s

    def get_solve_info(self):
        return {'solve_time': self.solve_time, 'executed_its': 1}

"""
Relaxed 
"""
class Quintic_1d_R:
    def __init__(self,param, traj_l,xc):
        self.id = 'Quintic'
        self.param = param
        self.traj_sl, self.traj_vl, self.traj_al = traj_l
        self.xc = xc
        # Relaxed boundary condition
        self.relaxT = 1.0
        self.relaxv = 2.0

        _, self.gd_profile = get_gradient_profile(feature=param.gd_profile_type)
        # self.s, self.v, self.av, self.jerk, self.at, self.ar, self.fr, self.fc, self.theta
        self.xs = [xc[0], xc[1], xc[2]]


        assert xc[0] < self.traj_sl[0], "collision happened, check code!"
        # ACC estimated end state
        se = self.traj_sl[-1] - (self.param.accT * self.traj_vl[-1] + self.param.dmin)
        ve = self.traj_vl[-1]
        ae = self.traj_al[-1]
        self.xe = [se, ve, ae]

        # assert 1==2, " sdfaasd "
        # self.T = np.arange(0, round(self.param.prediction_time/ self.param.dt), self.dt)
        self.T = self.param.prediction_time
        self.poly_list = []
        self.traj_list = []
        self.r_list = []
        self.rv_list = []
        self.rj_list = []
        self.rfc_list = []

        self.r_index = None
        self.rv_index = None
        self.rj_index = None
        self.rfc_index = None
        self.solve_time = np.inf


    def solve(self):
        self.poly_list = []
        self.traj_list = []
        self.r_list = []
        self.rv_list = []
        self.rj_list = []
        self.rfc_list = []

        self.r_index = None
        self.rv_index = None
        self.rj_index = None
        self.rfc_index = None
        self.solve_time = np.inf
        ts = time.time()
        
        # self.fig, self.a = plt.subplots(2,5)


        # for s in np.arange(self.xe[0]-self.xe[1]*self.relaxT, self.xe[0]+self.xe[1]*self.relaxT, 1):
        #     for v in np.arange(self.xe[1]-self.relaxv, self.xe[1]+ self.relaxv,1):
        #         poly = QuinticPolynomial(self.xs[0],self.xs[1],self.xs[2],\
        #                                 s,v,self.xe[2],\
        #                                 self.T, self.param.dt)
        #         self.poly_list.append(poly)
        # for s in np.arange(self.xe[0]-10, self.xe[0]+ 5, 1):
        """
        for s in np.arange(self.xe[0]-self.xe[1]*self.relaxT, self.xe[0]+self.xe[1]*self.relaxT, 1):
            for v in np.arange(self.xe[1]-self.relaxv, self.xe[1]+ self.relaxv,0.2):
                poly = QuinticPolynomial(self.xs[0],self.xs[1],self.xs[2],\
                                        s,v,0,\
                                        self.T, self.param.dt)
                self.poly_list.append(poly)
        """
        poly = QuinticPolynomial(self.xs[0],self.xs[1],self.xs[2],\
                                self.xe[0],self.xe[1],self.xe[2],\
                                # self.xe[0], 0., 0.,\
                                self.T, self.param.dt)
        self.poly_list.append(poly)
        # assert 1==2, "sdfad"
        for i in range(len(self.poly_list)):
            p = self.poly_list[i]
            traj = Traj(self.param,p,self.gd_profile,dv=self.xe[1])
            self.traj_list.append(traj)
            self.r_list.append(traj.r)
            self.rv_list.append(traj.rv)
            self.rj_list.append(traj.rj)
            self.rfc_list.append(traj.rfc) 
        self.r_index = self.r_list.index(min(self.r_list))
        self.rv_index = self.rv_list.index(min(self.rv_list))
        self.rj_index = self.rj_list.index(min(self.rj_list))
        self.rfc_index = self.rfc_list.index(min(self.rfc_list))
        self.res = self.traj_list[self.rfc_index]
        self.res = self.traj_list[self.rj_index]
        te = time.time()
        self.solve_time = te - ts

        # plot_traj(self.a, self.traj_list[self.rfc_index], al = 1, single_color='green', if_single_color= True)


        # print("Traj_s: ", self.res.Traj_s)
        # print("Traj_v: ", self.res.Traj_v)
        # print("Traj_a: ", self.res.Traj_a)
        # print("Traj_jerk: ", self.res.Traj_jerk)

        assert np.all(np.diff(self.res.Traj_s) >= 0), "Trajectory of s infeasible"



        return self.res.Traj_a[0], self.res.Traj_jerk[0]

    def update(self, xc, traj_l):
        self.xs = [xc[0], xc[1], xc[2]]
        self.traj_sl, self.traj_vl, self.traj_al = traj_l 

        # print(" car state got: ", xc)
        # print(" leading traj_sl: ", self.traj_sl)
        # ACC estimated end state
        se = self.traj_sl[-1] - (self.param.accT * self.traj_vl[-1] + self.param.dmin)
        ve = self.traj_vl[-1]
        ae = self.traj_al[-1]
        self.xe = [se, ve, ae]
        # print("*********xs *************: ", self.xs)
        # print("*********xe *************: ", self.xe)
    def get_traj_s(self):
        return self.res.Traj_s

    def get_solve_info(self):
        return {'solve_time': self.solve_time, 'executed_its': 1}