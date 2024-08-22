from emato.obj.poly.quintic import QuinticPolynomial, Traj
import numpy as np
from emato.util.util import *
import time

class FrenetTraj:
    def __init__(self):
        # all trajectory, not number
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.clon= 0.0
        self.clat = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.dx = []
        self.dy = []
        self.yaw = []
        # arc coordinate not lon
        self.dl = []
        self.l = []
        self.v = []
        self.a = []
        self.jerk = []
        self.c = []



class Quintic_2d:
    def __init__(self,param, road,traj_traffic_list, xc, desired_v):
        """
        xc should be [(s, s', s''), (d, d', d'')]
        that is [(cs, cs_d, cs_dd) (cd, cd_d, cd_dd)]
        """
        self.id = 'Quintic'
        self.param = param
        self.road = road
        self.traj_traffic_list = traj_traffic_list
        self.xc = xc
        self.ptime = param.prediction_time
        self.dt = self.param.dt
        self.rsafe = 3
        _, self.gd_profile = get_gradient_profile(feature=param.gd_profile_type)

        self.ftraj_list = []

        def solve(self):
            self.ftraj_list = []
            cs,cs_d,cs_dd,cd,cd_d,cd_dd = self.xc
            # goal d, goal s
            for gd in np.arange(-self.road.road_width/2+ 2, self.road.road_width/2-2, 0.5) :
                for gs in np.arange(desired_v*self.ptime - 30, desired_v*self.ptime+30, 5):
                    ft = FrenetTraj()
                    # (xs, vxs, axs, xe, vxe, axe, time,dt):
                    Tlat = QuinticPolynomial(cd, cd_d, cd_dd, gd, 0,0, self.ptime, self.dt)
                    Tlon = QuinticPolynomial(cs,cs_d,cs_dd, gs, desired_v,0, self.ptime, self.dt)
                    ft.t, ft.d, ft.d_d, ft.d_dd, ft.d_ddd = Tlat.get_traj()
                    _, ft.s, ft.s_d, ft.s_dd, ft.s_ddd = Tlon.get_traj()
                    # for i in range(len(ft.t)):
                    #     x,y,_ = self.road.frenet_to_global(ft.s[i],ft.d[i])
                    #     ft.x.append(x), 
                    ft.x, ft.y, _ = self.road.frenet_to_global(ft.s, ft.d)
                    ft.dx = np.diff(ft.x); dx = np.append(dx, dx[-1])
                    ft.dy = np.diff(ft.x); dy = np.append(dy, dy[-1])
                    ft.yaw = np.arctan2(dy, dx)
                    ft.dl = np.hypot(dx,dy)
                    # Calculate velocity along the arc
                    ft.v = ft.dl / self.dt
                    # Calculate acceleration along the arc
                    ft.a = np.diff(ft.v) / self.dt
                    ft.a = np.append(ft.a, ft.a[-1])  # Extend to match length
                    # Calculate jerk along the arc
                    ft.jerk = np.diff(ft.a) / self.dt
                    ft.jerk = np.append(ft.jerk, ft.jerk[-1])  # Extend to match length



                    """
                    To do: try use s,d and x,y to calc arc length and check collision, curvature, and fuel rate etc.
                    """

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
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